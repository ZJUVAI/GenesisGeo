# 证明图构建与频繁子图挖掘实现说明

本说明文档梳理本仓库中几何“证明图”构建与频繁子图挖掘（FSM）的任务要求、实现细节、参数与用法、测试与性能注意事项。对应代码在 `src/newclid/data_discovery/proof_graph.py` 中。

## 0. 近期更新速览（与脚本/输出对齐）
- 新增“规则仅挖掘”模式（rules-only）：在合并图上仅以规则节点建图与扩展，最终再基于嵌入回溯重建 schema；显著减小搜索图规模，同时保留可读化输出。
- 分叉挖掘（branched）鲁棒性修复：
  - 若模式中所有规则均已完整，先行“预 finalize”产出结果，避免在后续扩展中因限额而丢失可输出模式。
  - A 阶段（补齐规则）若没有实际进展，则不会提前 return，而是继续进入 FRF/attach 流程，减少“0 结果”情况。
  - 变量闭包检查改为“逐嵌入过滤”，更新支持度后再判定，避免一刀切误杀。
- 输出落盘统一：
  - 路径挖掘：`data_discovery/data/path_mining.json`
  - 分叉挖掘 fact_rule：`data_discovery/data/branched_mining.json`
  - 分叉挖掘 rule_only：`data_discovery/data/rules_only_mining.json`
  - 统一包含：meta（图统计/参数/用时）、top-k 摘要、完整 patterns（含 schema），不再写入 input.json 相关信息。
- 日志改进：在总节点/边统计后，追加 per-problem 平均节点/边数，便于设置 `max_nodes`/`max_edges`。

并行与脚本更新（Scheme B）
- 新增“种子级并行挖掘”通道：
  - 子线程仅负责“搜索”（按 seed 扩展 + emit 原始模式），不做 schema 转换/过滤。
  - 主线程集中进行 schema 转换、两层去重（结构签名 + 规范化 schema）、过滤（可选丢弃 unknown，变量闭包兜底）与写入（流式或批量）。
- 脚本参数更新（`Newclid/scripts/run_gspan_branched_demo.py`）：
  - `--parallel-mining {0,1}`：1 启用并行（默认 1）；0 使用旧的单次矿工运行。
  - `--mining-workers N`：并行时的工作线程数（默认 10）。
  - `--stream-write {0,1}`：1 主线程消费即写 NDJSON 并最终汇总；0 聚合后一次写。
  - `--drop-unknown-in-schema {0,1}`：是否直接丢弃包含 `unknown()` 的 schema（默认 1）。
- debug_limit_expansions 语义：
  - 旧路径（非并行）：全局扩展步数上限。
  - 并行路径：为“每个种子/线程内”的本地上限（仍可防止单个种子爆炸，但总扩展≈多线程之和）。若需要严格的全局上限，可引入线程安全的“全局预算”计数器（后续可选）。
- 日志顺序修正：并行模式下不再调用矿工的整体 run_*，避免其日志先于并行管线日志输出；挖掘时长统计在并行收集结束后记录。

## 1. 任务要求（Checklist）
- 图模型
  - 分层有向二分图：奇数层为 fact 节点，偶数层为 rule 节点。
  - 边不区分类型；仅存储有向边 (src, dst)。
- 数据来源与解析
  - 从 JSON 的 `results[*].proof.analysis`、`results[*].proof.numerical_check`、`results[*].proof.proof` 构建图。
  - `analysis` 与 `numerical_check` 中的子句统一解析为 fact：形如 `pred args [NNN]`。
  - `proof` 中每一步统一解析：`concl_pred args [NEW_ID] RULE [PID] [PID] ...`。
- 一致性与约束
  - 不做去重（同一个 [NNN] 多次出现时如一致则记录为重复，否则冲突并告警）。
  - 去重作用域与 [NNN] 局部 ID：
    - [NNN] 的作用域仅在单个 problem_id 内；同一题目内，analysis / numerical_check / proof 出现的相同 [NNN] 被视为指向同一“本题的局部实体”。
    - 同题同 [NNN] 若 predicate/args 完全一致，则允许重复登记；若不一致，保留首次登记并输出冲突告警（不抹平差异、不全局改写）。
    - 不进行跨题合并：不同 problem_id 即便标签、args 完全相同，也不会在合并图 G* 中被折叠或连边；仅保留其各自的节点与本题边。
    - 规则步索引/局部 ID 均为题内局部概念；缺失前提仅在本题内创建占位 fact（不会跨题借用任何节点）。
    - 支持度统计口径：以覆盖到的去重后的 problem_id 计数（每题最多计 1 次）；而 occurrence/embeddings 可在同一题内包含多处出现，用于举例展示但不影响支持度。
  - FSM 阶段仅使用 fact 的标签（predicate）与 rule 的 code；fact 的 args 暂不参与 FSM（仅在可读化输出时使用）。
  - 任意“无法解析/跳过/占位/冲突”必须通过 logger 和/或 print 发出提示，不允许静默忽略。
- 频繁子图挖掘（FSM）
  - 支持度按覆盖的 problem_id 数量计数。
  - 结构约束在输出阶段过滤（不影响扩展过程）：
    - 起点/终点均为 fact。
    - 最少规则节点数（min_rule_nodes）。
    - 最少边数（min_edges）。
  - 首版提供两种变体：
    - 路径挖掘（F->R->F->...），便于初始探索与性能控制。
    - 分叉子图挖掘（允许 rule 入度≥2 等结构），适配“多前提/多汇聚”的定理模式。
  - 扩展变体新增：
    - 规则仅挖掘（rules-only）：在规则-规则邻接图上搜索，再重构前提与唯一结论用于 schema 输出。

## 2. 数据模型与主要类
- ProofGraph
  - 存储节点与边、问题内局部 ID 映射、规则步映射等。
  - 公有字段：
    - `nodes: Dict[node_id, {type, label/code, args, problem_id, layer, ...}]`
    - `edges: List[(src_id, dst_id)]`
    - `fact_id_map: {problem_id: {id_local: fact_node_id}}`
    - `rule_step_map: {problem_id: {step_index: rule_node_id}}`
  - 关键方法：
    - `parse_facts_from_text(problem_id, text)`：解析 fact 子句并登记 fact 节点（layer=1）。
    - `parse_proof_step(line)`：解析单条 proof 步骤（结论 + 规则 + 前提 IDs）。
    - `add_rule_step(...)`：创建 rule 节点、连接前提→规则、规则→结论，缺失前提创建占位 fact。
    - `from_single_result(result_obj)` / `from_results_json(path)` / `from_results_obj(obj)`：构建整图入口。
- MergedGraph
  - 将所有题目的子图合并为一张大图 G*，节点标签规范化为 `F:{predicate}` 与 `R:{code}`；保留 `orig_node_id` 与 `problem_id`，不跨题连边。
- GSpanMiner
  - 构造合并图，并提供两种挖掘入口：
    - `run()`：路径挖掘变体（仅简单路径）。
  - `run_branched(...)`：分叉子图挖掘（规则完整性 + FRF 原子扩展 + 可选生产者接入），含上述鲁棒性修复。
  - `run_rules_only(...)`：规则仅挖掘；内部构建规则邻接 r1→r2（若存在 r1→F 且 F→r2，且同题），在仅含规则的图上扩展；输出阶段用 `pattern_to_schema_rules_only(...)` 还原前提与唯一结论。
  - 支持度阈值 `min_support` 可为绝对值（int）或比例（float 0~1）。

## 3. 解析规则
- 去标签：`<analysis>...</analysis>`、`<proof>...</proof>` 等用正则剥离，仅保留内容。
- fact 子句正则：`^\s*(?P<pred>\w+)\s+(?P<args>.*?)\s*\[(?P<id>\d+)\]\s*$`。
- proof 步骤：通过首个 `[NNN]` 锚定结论 ID；左侧拆出 `pred args`，右侧第一个 token 为 `rule_code`，其后找到全部 `[PID]`。
- 缺失前提：创建占位 fact（`label=unknown, args=[], layer=1`），并告警。
- 结论冲突：若同一局部 ID 先前已登记且内容不一致，保持原记录并告警忽略新值。

## 4. 路径挖掘（run）
- 思路：
  - 以所有 F->R 边作为种子，合并同标签对，聚合支持度（按 problem_id）。
  - 仅做 forward 扩展，保持简单路径（不重用节点），直到上限 `max_nodes`。
  - 输出阶段过滤：起止为 fact、规则节点数与边数达标、支持度达标。
  - 嵌入（embeddings）用于可读化；在扩展阶段不截断，以免丢失支持度；仅在结果输出时截断到 `sample_embeddings`。
- 输出：
  - `labels: [F:*, R:*, F:*, ...]`
  - `edges: [(0,1),(1,2),...]`（路径）
  - `support, pids, embeddings`
- 可读化：
  - `pattern_to_schema(pattern)` 取路径两端 fact，变量化其 args：`pred(X1,X2,...) => pred'(Y1,Y2,...)`。

## 5. 分叉子图挖掘（run_branched）
- 目标：支持“多前提/多汇合”结构，如 `f1,f2 -> r1 -> f3`；`f3,f4 -> r2 -> f5`。
- 初始种子：FRF 种子，生成时就补齐该 rule 的其余前提（保证“规则完整性”从一开始成立）。
- 扩展类型（替换旧三步）：
  - 规则完整性补齐：若模式中存在未完整的 rule，则优先且仅做“补齐该 rule 的全部前提”。
  - FRF 原子扩展：从某个 fact 一次性扩到其相邻 rule 再到该 rule 的结论 fact，并同时补齐该 rule 的其余前提。
  - 生产者接入（可选）：attach-producer(F)，从某个已在模式中的 fact F 向上接入其生产者 rule r（r->F）及其全部前提；默认仅上溯 1 层。
- 去重与访问控制：按 `(labels_tuple, sorted_edges)` 作为签名进行去重和避免重复扩展。
- 输出过滤：
  - 存在至少一个入度=0 的 fact（源前提）。
  - 所有出度=0 的节点都是 fact，且唯一结论 fact。
  - 规则节点数 ≥ `min_rule_nodes`；边数 ≥ `min_edges`；支持度 ≥ `min_support`。
  - 可选 `min_rule_indeg2_count`：至少有 N 个 rule 的入度≥2。
  - 可选变量闭包兜底（enable_var_closure_check）：逐嵌入过滤，结论 fact 的变量必须包含于所有源前提的变量并集（默认关闭）。
- 输出：
  - `nodes: [{idx, label}]`、`labels`、`edges`、`support`、`pids`、`embeddings`。
- 可读化：
  - `pattern_to_schema_branched(pattern)`：
    - premise = 所有入度=0 的 fact 的合取；
    - conclusion = 所有出度=0 的 fact 的合取；
    - 输出形如：`P1 ∧ P2 => C1 ∧ C2`。

提示（鲁棒性）：
- 当 A 阶段无法推进时继续尝试 FRF/attach，可显著降低在大上限（`max_nodes/max_edges`）下“0 结果”的概率。
- 当所有规则已完整时立即 finalize，防止在后续扩展中因资源限制而错失可输出模式。

## 5.1 规则仅挖掘（run_rules_only）
- 图构建：在同一 problem_id 内，若存在 `r1 -> F` 且 `F -> r2`，则在规则图上连一条 `r1 -> r2` 的边；仅保留规则节点与上述边，显著缩减搜索空间。
- 搜索：在规则图上进行与分叉挖掘相似的扩展/去重与支持度聚合（按题目计数）。
- finalize：对每个模式，基于其嵌入在原始合并图 G* 中回溯，重建所有前提 fact 与唯一结论 fact，用于 schema 与输出过滤（保持与分叉挖掘一致的判定，如“唯一结论 fact”）。
- 输出：与分叉挖掘基本一致（labels 仅含 `R:*`），`pattern_to_schema_rules_only(...)` 给出 `P1 ∧ ... => C` 的可读化表达。

## 6. 参数说明与建议
- 通用：
  - `min_support`：绝对数或比例（float）。提高可显著加速。
  - `min_rule_nodes`、`min_edges`：抑制平凡模式，增大可提升质量、也会降低数量。
  - `max_nodes`（与分叉版的 `max_edges`）：控制爆炸，建议先小后大。
  - `sample_embeddings`：仅影响输出体量，不建议在扩展阶段截断（实现已保证仅在结果阶段截断）。
- 分叉特有：
  - `min_rule_indeg2_count`：≥1 则强制至少一个 rule 入度≥2，有助于筛选“多前提”模式。
  - `prune_low_support_labels`：按标签（F:*, R:*）的题目覆盖度进行全局剪枝。
  - `prune_by_rule`：按规则 code（R:code）的题目覆盖度进行全局剪枝。
  - `attach_producer` 与 `max_producer_depth`：是否启用与向上接入层数上限。
  - `skip_unknown`：扩展/接入时跳过 F:unknown 前提以减少节点噪声。
  - `enable_var_closure_check`：输出阶段的变量闭包兜底检查（默认关闭）。
 - 规则仅挖掘：
   - 推荐与分叉版共享相同的结构/支持度阈值；在较大图上优先选择 rules-only 以快速获得高质量候选，再回溯生成 schema。

## 7. 快速开始（一键脚本 + 集中超参）
- 提供开箱即用脚本，超参集中在脚本内的 `CONFIG` 字典中：
  - 路径挖掘：`Newclid/run_gspan_demo.py`
  - 分叉/规则仅：`Newclid/scripts/run_gspan_branched_demo.py`（通过 `--mode` 选择）
- 直接运行即可按安全默认值执行；如需覆盖，脚本也提供可选的命令行参数。

示例：
```sh
/usr/bin/env python3 Newclid/run_gspan_demo.py
/usr/bin/env python3 Newclid/scripts/run_gspan_branched_demo.py --mode fact_rule
/usr/bin/env python3 Newclid/scripts/run_gspan_branched_demo.py --mode rule_only
```

两脚本默认使用较小数据集 `r07_expanded_problems_results_lil.json`，避免卡住；可通过 `--json` 指定完整数据集。

常用可覆盖参数（二者交集）：
- `--json`: 数据源 JSON 路径（默认 lil 数据集）
- `--min-support`: 支持度（int 或 float 比例字符串，如 "2" 或 "0.2"）
- `--min-rule-nodes`: 最少规则节点数
- `--min-edges`: 最少边数
- `--max-nodes`: 节点上限（路径/分叉）
- `--sample-embeddings`: 每个模式展示的代表性嵌入数
- `--top-k`: 打印前 K 个模式
- `--quiet`: 静默解析模式

分叉/规则仅脚本特有参数：
- `--min-rule-indeg2-count`: 至少 N 个 rule 的入度≥2（默认 1）
- `--max-edges`: 边数上限
- `--debug-limit-expansions`: 扩展步数硬上限（防爆炸）
- `--debug-log-every`: 扩展日志频率（步）
- `--time-budget-seconds`: 全局时间预算（秒，缺省为不限）
- `--prune-low-support-labels`: 1/0，按标签的全局题目覆盖度进行剪枝（默认 1）
- `--prune-by-rule`: 1/0，按规则 code 的全局题目覆盖度进行剪枝（默认 1）
- `--attach-producer`: 1/0，是否允许接入生产者规则（默认 1）
- `--max-producer-depth`: 生产者接入向上层数（默认 1）
- `--skip-unknown`: 1/0，是否跳过 F:unknown 前提（默认 1）
- `--enable-var-closure-check`: 1/0，是否启用变量闭包兜底（默认 0）
 - `--mode {fact_rule,rule_only}`：分叉/规则仅模式（仅在 `scripts/run_gspan_branched_demo.py` 提供）

推荐默认（已写入脚本 CONFIG，中小规模数据能稳定跑完，不会卡住）：
- 路径：`min_support=2, min_rule_nodes=2, min_edges=3, max_nodes=9, sample_embeddings=1, top_k=10, quiet=True`
- 分叉：`min_support=2, min_rule_nodes=3, min_edges=3, min_rule_indeg2_count=1, max_nodes=10, max_edges=16, sample_embeddings=2, debug_limit_expansions=30000, debug_log_every=5000, prune_low_support_labels=True, prune_by_rule=True, attach_producer=True, max_producer_depth=1, skip_unknown=True, enable_var_closure_check=False, time_budget_seconds=None, top_k=10, quiet=True`

## 8. 运行方法（命令）
- 路径挖掘 demo：
  ```sh
  /usr/bin/env python3 Newclid/run_gspan_demo.py \
    --json Newclid/src/newclid/data_discovery/r07_expanded_problems_results.json \
    --min-support 2 --min-rule-nodes 2 --min-edges 3 --top-k 5 --sample-embeddings 1
  ```
- 分叉挖掘 demo（fact_rule）：
  ```sh
  /usr/bin/env python3 Newclid/scripts/run_gspan_branched_demo.py --mode fact_rule \
    --json Newclid/src/newclid/data_discovery/r07_expanded_problems_results.json \
    --min-support 2 --min-rule-nodes 3 --min-edges 3 --min-rule-indeg2-count 1 \
    --max-nodes 10 --max-edges 16 --sample-embeddings 2 --top-k 10
  ```
- 规则仅挖掘 demo（rule_only）：
  ```sh
  /usr/bin/env python3 Newclid/scripts/run_gspan_branched_demo.py --mode rule_only \
    --json Newclid/src/newclid/data_discovery/r07_expanded_problems_results.json \
    --min-support 2 --min-rule-nodes 3 --min-edges 3 \
    --max-nodes 10 --max-edges 16 --sample-embeddings 2 --top-k 10
  ```

## 9. 输出与落盘
- 统一落盘目录：`data_discovery/data/`
- 文件名：
  - 路径挖掘：`path_mining.json`
  - 分叉挖掘（fact_rule）：`branched_mining.json`
  - 规则仅挖掘（rule_only）：`rules_only_mining.json`
- 文件结构（示例字段）：
  - `meta`: { graph_stats, params, timings, dataset_name, ... }
  - `top_k`: [ { pattern_id, support, labels, schema, ... } ]
  - `patterns`: 完整列表，每个含 `labels/nodes/edges/support/pids/embeddings/schema`
- 说明：不包含 input.json 的复制/大小信息；JSON 为 pretty 格式便于阅读与版本管控。

## 10. 单元测试
- 路径挖掘：`Newclid/tests/test_gspan.py`
- 分叉挖掘：`Newclid/tests/test_gspan_branched.py`
- 运行：
  ```sh
  /usr/bin/env python3 -m unittest Newclid/tests/test_gspan.py -v
  /usr/bin/env python3 -m unittest Newclid/tests/test_gspan_branched.py -v
  # 或一次性发现
  /usr/bin/env python3 -m unittest discover -s Newclid/tests -v
  ```

## 11. 性能与注意事项
- 性能关键点：
  - 支持度阈值越低、结构上限越大，搜索空间越大；优先提高 `min_support`。
  - 扩展阶段不截断嵌入，防止“支持度早丢失→过早剪枝”，仅在结果阶段截断输出样本。
  - 合并图上建议开启 `--quiet` 以减少日志开销。
- 可能的改进：
  - 为分叉挖掘引入完整 gSpan 的最小 DFS 码去同构，避免标签一致但结构不同的折叠。
  - 建立按标签分桶与反向邻接索引，加速扩展候选检索。
  - 加入基于 pid 的代表采样或 beam 策略控制扩展宽度（默认未启用）。

可见性增强：
- 解析完成后，在总 nodes/edges 统计后打印 per-problem 平均节点/边（problems, avg_nodes, avg_edges），可据此调整 `max_nodes`/`max_edges`。

## 12. 已知局限与后续计划

## 附：文件与目录速览（与本功能相关）
- 代码实现：`Newclid/src/newclid/data_discovery/proof_graph.py`
- 一键脚本：
  - `Newclid/run_gspan_demo.py`（路径挖掘，集中超参）
  - `Newclid/scripts/run_gspan_branched_demo.py`（分叉/规则仅挖掘，集中超参 + 限流/剪枝/时间预算）
- 测试与演示：
  - `Newclid/tests/run_gspan_demo.py`（旧版 CLI 演示）
  - `Newclid/tests/run_gspan_branched_demo.py`（新版 CLI 演示，含 debug 限流）
  - `Newclid/tests/test_gspan.py`、`Newclid/tests/test_gspan_branched.py`、`Newclid/tests/test_gspan_branched_step.py`
- 路径挖掘仅产出简单路径；
- 分叉挖掘当前使用 `(labels, sorted_edges)` 作为签名，非严格同构判定；
- 计划升级：
  - 引入 directed gSpan 的 canonical DFS code 与 rightmost 扩展；
  - 更丰富的 schema 转换（含中间 fact/多个结论的表达）；
  - 导出/可视化接口（GraphML/JSON）以及 FSM 结果持久化。

---
如需进一步扩展或特定模式的约束（例如强制存在某些 rule code 序列），可在输出阶段增加附加过滤器，或在扩展过程中加入轻量剪枝（保持与“输出阶段约束为主”的约定一致）。
