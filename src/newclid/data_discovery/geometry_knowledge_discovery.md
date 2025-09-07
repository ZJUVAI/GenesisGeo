# 几何知识发现

## 0. 文件与目录速览（与本功能相关）
- 代码实现：`Newclid/src/newclid/data_discovery/proof_graph.py`
- 一键脚本：
  - `Newclid/scripts/run_gspan_demo.py`（路径挖掘，集中超参）
  - 根入口 `run_gspan_branched_demo.py`（调用 `Newclid/scripts/run_gspan_branched_demo.py`，集中超参 + 限流/剪枝/时间预算）
- 测试与演示：
  - `Newclid/tests/test_gspan.py`、`Newclid/tests/test_gspan_branched.py`、`Newclid/tests/test_gspan_branched_step.py`
- 路径挖掘仅产出简单路径；
- 分叉挖掘当前使用 `(labels, sorted_edges)` 作为签名，非严格同构判定；
- 计划升级：
  - 引入 directed gSpan 的 canonical DFS code 与 rightmost 扩展；
  - 更丰富的 schema 转换（含中间 fact/多个结论的表达）；
  - 导出/可视化接口（GraphML/JSON）以及 FSM 结果持久化。

---

## 1. 概述与背景

- 当前存在的问题 / 待解决目标
  1. AlphaGeometry中的LLM主要充当辅助构造的向导，而非新几何原理的发现者。
  2. 包括AG在内的几何定理证明系统依赖于静态数据集或预设知识（defs & rules）。
  3. 人类直观的非形式化证明与机器可验证的形式化证明之间的鸿沟是一个主要瓶颈。
- 几何发现思路：分析带辅助点的生成数据，发掘其中利用辅助点、经常出现、有新颖性与非平凡性质的证明过程构成新定理；如果把题目的证明过程看成是DAG，那么找到经常出现的子过程就是发掘依赖图中的频繁子图
  ```
  "analysis": "<analysis> coll c d k [000] ; coll c g l [001] ; cong c k d k [002] ; cong c l g l [003] ; coll g k n [004] ; coll d l n [005] ; </analysis>",  
  "numerical_check": "<numerical_check> ncoll g k l [006] ; sameside k c d l c g [007] ; sameclock d g n k n l [008] ; sameclock d g l g z l [009] ; sameclock g k z k l n [010] ; </numerical_check>",  
  "proof": "<proof> eqratio c k c l d k g l [011] a00 [002] [003] ; para d g k l [012] r27 [000] [001] [011] [006] [007] ; eqangle d g g l l z g l [013] a01 [014] [012] ; eqangle d l g l g z g l [015] a01 [016] ; simtri d g l z l g [017] r34 [013] [015] [009] ; eqratio d l g l g z g l [018] r52 [017] ; eqangle d g g n k l k n [019] a01 [004] [012] ; eqangle d n g n l n k n [020] a01 [005] [004] ; simtri d g n l k n [021] r34 [019] [020] [008] ; eqratio d n g n l n k n [022] r52 [021] ; eqangle g k k n k z k l [023] a01 [004] [014] ; eqangle g z k z l n k l [024] a01 [005] [014] [016] ; simtri g k z n k l [025] r34 [023] [024] [010] ; eqratio g k k n k z k l [026] r52 [025] ; eqratio g z k z l n k l [027] r52 [025] ; eqratio d l d n g k g n [028] a00 [018] [022] [026] [027] ; </proof>"  
  ```
- 预期贡献
  1. 自主几何引理发现与泛化：系统应能够超越辅助构造的简单建议，自主识别、泛化和存储可应用于未来问题的创新性几何引理。
      1. 更高效率：相比每次通过AG进行辅助构造的猜测，直接从库中提取新发现定理可以更高效地解决问题
  2. 弥合几何形式化与非形式化证明之间的鸿沟：通过整合人类非形式化证明作为指导，使系统能够将人类的直观思维转化为严谨的几何形式化步骤。
      1. 可以为下一阶段的自动形式化积累数据集（或者直接生成非形式化-形式化文本对的数据集）
  3. 构建自我持续的几何学课程：设计一个反馈循环，使证明尝试的成功或失败能够指导进化器优先泛化哪些类型的引理和解决哪些请求，从而创建一个自我强化的几何学习过程。

## 2. 目标与阶段里程碑

- 验证思路：目前的ddar规则集中，r07（Thales Theorem I）的结论可以通过添加辅助点来证明，包含r07的题目如果添加相应的辅助点，就可以不依赖r07求解。如果对一批这样的求解结果利用频繁子图搜索算法重新找出r07，就可以验证这个思路是可行的
  ```
  r07 Thales Theorem I
  para A B C D, coll O A C, ncoll O A B, coll O B D => eqratio3 A B C D O O
  ```

## 3. 数据与证明图构建

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

### 3.1 解析规则
- 去标签：`<analysis>...</analysis>`、`<proof>...</proof>` 等用正则剥离，仅保留内容。
- fact 子句正则：`^\s*(?P<pred>\w+)\s+(?P<args>.*?)\s*\[(?P<id>\d+)\]\s*$`。
- proof 步骤：通过首个 `[NNN]` 锚定结论 ID；左侧拆出 `pred args`，右侧第一个 token 为 `rule_code`，其后找到全部 `[PID]`。
- 缺失前提：创建占位 fact（`label=unknown, args=[], layer=1`），并告警。
- 结论冲突：若同一局部 ID 先前已登记且内容不一致，保持原记录并告警忽略新值。

## 4. 方法

- 频繁子图挖掘（FSM）总体设定
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

### 4.1 路径挖掘（run）
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

### 4.2 分叉子图挖掘（run_branched）
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
  - 可选变量闭包兜底（enable_var_closure_check）：逐嵌入过滤，结论 fact 的变量必须包含于所有源前提的变量并集（脚本默认开启，可在 CONFIG 关闭）。
- 输出：
  - `nodes: [{idx, label}]`、`labels`、`edges`、`support`、`pids`、`embeddings`。
- 可读化：
  - `pattern_to_schema_branched(pattern)`：
    - premise = 所有入度=0 的 fact 的合取；
    - conclusion = 所有出度=0 的 fact 的合取；
    - 输出形如：`P1 ∧ P2 => C1 ∧ C2`。

提示（鲁棒性）：
- 当 A 阶段无法推进时继续尝试 FRF/attach，可显著降低在大上限（`max_nodes`）下“0 结果”的概率。
- 当所有规则已完整时立即 finalize，防止在后续扩展中因资源限制而错失可输出模式。

### 4.3 规则仅挖掘（run_rules_only）
- 图构建：在同一 problem_id 内，若存在 `r1 -> F` 且 `F -> r2`，则在规则图上连一条 `r1 -> r2` 的边；仅保留规则节点与上述边，显著缩减搜索空间。
- 搜索：在规则图上进行与分叉挖掘相似的扩展/去重与支持度聚合（按题目计数）。
- finalize：对每个模式，基于其嵌入在原始合并图 G* 中回溯，重建所有前提 fact 与唯一结论 fact，用于 schema 与输出过滤（保持与分叉挖掘一致的判定，如“唯一结论 fact”）。
- 输出：与分叉挖掘基本一致（labels 仅含 `R:*`），`pattern_to_schema_rules_only(...)` 给出 `P1 ∧ ... => C` 的可读化表达。

### 4.4 参数说明与建议
- 通用：
  - `min_support`：绝对数或比例（float）。提高可显著加速。
  - `min_rule_nodes`、`min_edges`：抑制平凡模式，增大可提升质量、也会降低数量。
  - `max_nodes`：控制爆炸，建议先小后大。
  - `rule_only_max_nodes_ratio`：仅在 rule_only 模式生效。将每题规则节点最大值乘以该比例并上取整，作为有效 `max_nodes`；不大于 0 时忽略，沿用显式 `max_nodes`。
  - `sample_embeddings`：仅影响输出体量，不建议在扩展阶段截断（实现已保证仅在结果阶段截断）。
- 分叉特有：
  - `min_rule_indeg2_count`：≥1 则强制至少一个 rule 入度≥2，有助于筛选“多前提”模式。
  - `prune_low_support_labels`：按标签（F:*, R:*）的题目覆盖度进行全局剪枝。
  - `prune_by_rule`：按规则 code（R:code）的题目覆盖度进行全局剪枝。
  - `attach_producer` 与 `max_producer_depth`：是否启用与向上接入层数上限。
  - `skip_unknown`：扩展/接入时跳过 F:unknown 前提以减少节点噪声。
  - `enable_var_closure_check`：输出阶段的变量闭包兜底检查（脚本默认 True，可在脚本内 CONFIG 关闭）。
 - 规则仅挖掘：
   - 推荐与分叉版共享相同的结构/支持度阈值；在较大图上优先选择 rules-only 以快速获得高质量候选，再回溯生成 schema。

## 5. 实现与代码结构

### 5.1 近期更新速览（与脚本/输出对齐）
- 新增“规则仅挖掘”模式（rules-only）：在合并图上仅以规则节点建图与扩展，最终再基于嵌入回溯重建 schema；显著减小搜索图规模，同时保留可读化输出。
- 分叉挖掘（branched）鲁棒性修复：
  - 若模式中所有规则均已完整，先行“预 finalize”产出结果，避免在后续扩展中因限额而丢失可输出模式。
  - A 阶段（补齐规则）若没有实际进展，则不会提前 return，而是继续进入 FRF/attach 流程，减少“0 结果”情况。
  - 变量闭包检查改为“逐嵌入过滤”，更新支持度后再判定，避免一刀切误杀。
- 输出落盘统一（仅分叉/规则仅）：
  - 分叉挖掘（fact_rule）：`data_discovery/data/branched_mining.json`
  - 规则仅挖掘（rule_only）：`data_discovery/data/rules_only_mining.json`
  - 统一包含：`created_at/proof_graph/merged_graph/params/timing_sec/patterns_summary_topN/patterns` 等结构化字段（不再写入 input.json 相关信息）。
- 新的 MiningPipeline 后处理总线：将 schema 生成与多步过滤、去重、最终“同结论按前提集合最小化”与审计写入封装为 `MiningPipeline` 类，脚本仅负责超参与调用。
- aconst/rconst 语义修正：其最后一个参数是数值常量（非点/变量），不计入依赖/变量闭包；在 schema 渲染时保留字面量、不参与变量重命名。
- 日志改进：在总节点/边统计后，追加 per-problem 平均节点/边数，便于设置 `max_nodes`。

并行与脚本更新（多进程 seeds_mproc）
- 新增“种子级并行挖掘”通道：
  - 子进程仅负责搜索（按 seed 扩展并通过 emit 推送原始模式对象），不做 schema/过滤。
  - 主进程集中完成 schema 转换、两层去重（结构签名 + 规范化 schema）、过滤（可选丢弃 unknown、变量闭包兜底与依赖过滤）与写入（流式或批量）。
- 脚本参数（`Newclid/scripts/run_gspan_branched_demo.py`）：支持常见命令行参数，但推荐直接编辑脚本内 CONFIG 以控制引擎/阈值/剪枝/时间预算等。
- 全局扩展预算（重要变化）：
  - seeds_mproc 下，`debug_limit_expansions` 作为“全体子进程共享”的严格上限实现，使用 `multiprocessing.Semaphore` 作为跨进程预算；每次可导致结构增长的扩展尝试前均会尝试消耗 1 个配额，耗尽后所有 worker 将不再产生新扩展。
  - `single` 与 `seeds` 引擎下则为“当前一次运行/当前种子内”的本地上限。
- 死锁修复与可靠退出：
  - 工作队列采用 `JoinableQueue` + 阻塞式 `get()`；在启动前预投递与 worker 数相同的哨兵 `None`，worker 消费到哨兵后 `task_done()` 并退出。
  - 主进程对每次 `put` 调用都有匹配的 `task_done()`；`jobs.join()` 不再挂起。
  - 结果通道使用 `qout`；每个 worker 结束时会向 `qout` 发送一次 `None` 作为完成信号，主进程据此统计 worker 退出并收尾。
  - 加入 `maxsize` 以防止输出洪泛，emit 端采用带超时的 `put` 避免阻塞。
- 日志顺序：不再调用整体 `run_*`，而是在主进程汇总后统一记录耗时。
入口脚本：仓库根目录提供薄封装 `run_gspan_branched_demo.py`，可直接 `python run_gspan_branched_demo.py` 运行，相当于调用 `Newclid/scripts/run_gspan_branched_demo.py`。

### 5.2 数据模型与主要类
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
  - `run_branched(...)`：分叉子图挖掘（规则完整性 + FRF 原子扩展 + 可选生产者接入），含鲁棒性修复。
  - `run_rules_only(...)`：规则仅挖掘；内部构建规则邻接 r1→r2（若存在 r1→F 且 F→r2，且同题），在仅含规则的图上扩展；输出阶段用 `pattern_to_schema_rules_only(...)` 还原前提与唯一结论。
  - 支持度阈值 `min_support` 可为绝对值（int）或比例（float 0~1）。

### 5.3 运行方法（命令）
```sh
/usr/bin/env python3 Newclid/scripts/run_gspan_demo.py
/usr/bin/env python3 run_gspan_branched_demo.py
```
超参与模式从脚本内 `CONFIG` 修改。

默认数据集：
- 路径挖掘默认 `r07_expanded_problems_results_lil.json`；
- 分叉/规则仅默认 `r07_expanded_problems_results.json`。

### 5.4 输出与落盘
- 统一落盘目录：`data_discovery/data/`
- 文件名：
  - 分叉挖掘（fact_rule）：`branched_mining.json`
  - 规则仅挖掘（rule_only）：`rules_only_mining.json`
- 文件结构（示例字段）：
  - `created_at`: 生成时间（ISO 格式）
  - `proof_graph`: 原图统计（每题与合并图）
  - `merged_graph`: 合并图统计与标签覆盖度
  - `params`: 运行参数快照（来自脚本内 CONFIG）
  - `timing_sec`: 各阶段耗时
  - `pattern_count`: 模式总数
  - `patterns_summary_topN`: 概览（前 N 条）
  - `patterns`: 完整列表（含 `labels/nodes/edges/support/pids/embeddings/schema/rendered` 等）
- 说明：
  - 路径挖掘 demo 不写入文件，仅打印到 stdout；
  - 不包含 input.json 的复制/大小信息；结果 JSON 采用“混合美化”排版：
  - 绝大多数对象/数组使用缩进打印，便于阅读与 diff；
  - `labels`、`rules`、`fr_edges` 等短数组保持单行输出，避免冗长；
  - 审计文件使用 NDJSON（每行一条）。
  实现参考 `MiningPipeline._dump_pretty_mixed(...)`。

#### 5.4.1 后处理管线（MiningPipeline）与审计文件
- 入口：`MiningPipeline(pg, miner, args, out_dir).run(...)`。
- 处理阶段：
  1) schema 生成（基于 `pattern_to_schema_*`），并保留 `schema_before_dependency` 字段用于对照；
   2) 过滤 unknown：若在配置中启用“丢弃 unknown”，则直接丢弃包含 `unknown()` 的 schema；
  3) 依赖过滤（可选）：仅对首个嵌入进行三分支判定：
     - 若所有前提点集均是结论点及其依赖并集的子集（all_subset），丢弃；
     - 若均不是子集（all_not_subset），丢弃；
     - 若部分是子集，则保留并裁剪掉不满足子集条件的前提，重建 `schema_after`；
     注：`aconst/rconst` 的最后一个参数被视为字面量，不参与点集/依赖判断；
  4) 变量闭包兜底（可选）：结论参数集合需为前提参数集合的子集，且同样忽略 `aconst/rconst` 的最后一个参数；
  5) 两层去重：
     - 结构签名去重：按 `(labels, sorted(edges))`；
     - 规范化 schema 去重：变量重命名与前提无序排序后的键；
  6) 最终“同结论按前提集合最小化”：对相同结论组内，若某条的前提集合严格包含另一条，则丢弃前者（胜者为前提更少的那条）。规范化时 `aconst/rconst` 的最后一个参数保留字面量、不变量化。
- 审计输出（与结果分开，NDJSON，一行一条）：
  - step1：`audit_{mode}_step1_after_unknown.ndjson`（unknown 过滤后样本）；
  - step2-kept：`audit_{mode}_step2_dep_kept.ndjson`（依赖过滤保留/裁剪的明细，含 schema_before/schema_after、union_rely、各前提判定）；
  - step2-dropped：`audit_{mode}_step2_dep_dropped.ndjson`（依赖过滤丢弃原因与细节）；
  - step3：`audit_{mode}_step3_after_varclosure.ndjson`（变量闭包后样本）；
  - step4：`audit_{mode}_step4_final_dedup.ndjson`（两层去重后样本，含 `schema_before_dependency`）；
  - step5-dropped：`audit_{mode}_step5_subset_min_dropped.ndjson`（“前提集合最小化”阶段被丢弃的记录，包含赢家/输家对照）。

额外文件（rule_only）：
- `unknown_problem.txt`：当某些题目缺少必要的 rule-only 邻接信息而被跳过时，记录其 problem_id，便于后续检查数据完整性。

#### 5.4.2 结果可视化载荷：rendered 字段
- 目的：给出与 schema 变量重命名一致的“子图快照”，便于人工检查与复现。
- 位置：出现在 `patterns[*]` 与 `top_k[*]` 的每条记录中。
- 共同：都包含 `schema_vars`（当前 schema 的变量映射，如 `{"a": "X1", ...}`）。
- fact_rule 模式：
  - 结构：`{ nodes: [{gid, label}], edges: [[u,v],...], schema_vars }`
  - 节点 `label`：
    - fact：按 schema 变量名渲染，如 `pred(X1,X2,...)`；`aconst/rconst` 的最后一个参数保持字面量（不变量化）。
    - rule：`R:code`。
  - 边：与模式中的 `edges` 一致（以 0..N-1 的局部下标表示）。
- rule_only 模式：
  - 结构：`{ rules: [{gid,label}], facts: [{gid,label,role}], fr_edges: [[f,r],...], rf_edges: [[r,f],...], schema_vars }`
  - rules：`label` 为 `R:code`。
  - facts：从“首个嵌入”回溯重建，按 schema 变量渲染 `pred(X...)`，并给出 `role`：`premise | internal | conclusion`。
  - 边：`fr_edges`（前提→规则）、`rf_edges`（规则→fact）。
  - 说明：此模式下 fact 不是搜索图的一部分，因而通过嵌入映射回原合并图获取。

### 5.5 多进程实现要点与顺序（seeds_mproc）
- 任务分发：`JoinableQueue` 承载种子；启动 worker 前先投递与 worker 数相等的哨兵 `None`。
- Worker 循环：阻塞 `get()`；遇到 `None` 调用 `task_done()` 并退出；正常任务在 `finally` 中调用 `task_done()`，确保与 `put` 配对，避免 `jobs.join()` 卡住。
- 结果归集：`qout` 传输模式对象；每个 worker 结束前向 `qout` 发送一次 `None` 作为完成信号，主进程统计 `finished == workers` 后收尾。
- 全局扩展预算：在主进程构造 `Semaphore(limit)` 并传入 `expand_*_from_seed(..., global_budget=...)`，在关键扩展点尝试 `acquire(False)` 以原子消耗配额；用尽即停止产生新扩展。
- 关闭顺序：主进程消费完成信号 → `jobs.join()` → `Process.join()`；并关闭审计句柄。

### 5.6 单元测试
- 路径挖掘：`Newclid/tests/test_gspan.py`
- 分叉挖掘：`Newclid/tests/test_gspan_branched.py`
- 运行：
  ```sh
  /usr/bin/env python3 -m unittest Newclid/tests/test_gspan.py -v
  /usr/bin/env python3 -m unittest Newclid/tests/test_gspan_branched.py -v
  /usr/bin/env python3 -m unittest Newclid/tests/test_gspan_branched_step.py -v
  # 或一次性发现
  /usr/bin/env python3 -m unittest discover -s Newclid/tests -v
  ```

### 5.7 性能与注意事项
- 性能关键点：
  - 支持度阈值越低、结构上限越大，搜索空间越大；优先提高 `min_support`。
  - 扩展阶段不截断嵌入，防止“支持度早丢失→过早剪枝”，仅在结果阶段截断输出样本。
  - 合并图上建议在配置中开启 `quiet` 以减少日志开销。
- 可能的改进：
  - 为分叉挖掘引入完整 gSpan 的最小 DFS 码去同构，避免标签一致但结构不同的折叠。
  - 建立按标签分桶与反向邻接索引，加速扩展候选检索。
  - 加入基于 pid 的代表采样或 beam 策略控制扩展宽度（默认未启用）。

可见性增强：
- 解析完成后，在总 nodes/edges 统计后打印 per-problem 平均节点/边（problems, avg_nodes, avg_edges），可据此调整 `max_nodes`。

## 6. 局限与后续计划
- 已知局限与后续计划保持与现有实现一致。

---

### 附录：Schema 批量提取与评估（scripts/schema_eval.py）

目的与定位：
- 从分叉挖掘输出 `branched_mining.json` 中批量提取 `schema` 与 `schema_before_dependency`，
  统一对齐 `scripts/translate_rule_to_problem.py` 的接口格式，并批量翻译→求解→写出结果。

脚本与位置：
- 主脚本：`Newclid/scripts/schema_eval.py`
  - 运行风格参照 demo 脚本：默认参数写在脚本顶部 `Args` 类；在仓库根目录执行：`python scripts/schema_eval.py`。
  - 全程 `logging` 输出实时进度；不做去重。

输入：
- `src/newclid/data_discovery/data/branched_mining.json`（分叉挖掘输出，含 patterns[*].schema/schema_before_dependency）

标准化与翻译：
- 规范化 schema：将“∧”统一替换为逗号；保持 `premises => conclusions` 结构。
- 每个输入 JSON 对应派生的两行规则与拆分产物（便于审计）：
  - `<basename>.schema.rules.txt` 与 `<basename>.schema.split.txt`
  - `<basename>.schema_before.rules.txt` 与 `<basename>.schema_before.split.txt`
  位于：`src/newclid/data_discovery/data/` 目录。

求解与证明：
- 对每条翻译出的 `Problem: ...`，使用 `newclid` 的 `GeometricSolverBuilder` 构建并 `run()`；
- 通过 `proof_writing.write_proof_steps(solver.proof)` 捕获证明文本并写入 JSON（stdout 捕获）。

成功判定标准：
- translate_fail：`translate_success == False`；
- solver_fail：`translate_success == True` 且所有 problems 的 `solver_success == False`；
- ok（成功）：`translate_success == True` 且存在至少一个 `solver_success == True`。

输出与落盘（一个输入对应一个输出）：
- 结果文件：
  - `<basename>.schema.results.json`
  - `<basename>.schema_before.results.json`
  每条记录包含：`title/kind/rule/translate_success/problems[*]`。
  位置：`src/newclid/data_discovery/data/`。

命令行输出：
- 打印每个 kind 的保存路径与记录条数；`Args.topn_print` 仍可用于内部调试（当前默认不打印 Top-N 分类预览）。

运行方法：
```sh
/usr/bin/env python3 scripts/schema_eval.py
```

实现说明与健壮性：
- 译器装载：通过 `importlib` 动态加载脚本中的 `process_rules`；若加载失败仅跳过翻译步骤并记录日志。
- 求解器缺失：若运行环境无法导入 `newclid` 求解器，则问题级别结果标记为 `solver_unavailable`，不影响文件落盘。

---
如需进一步扩展或特定模式的约束（例如强制存在某些 rule code 序列），可在输出阶段增加附加过滤器，或在扩展过程中加入轻量剪枝（保持与“输出阶段约束为主”的约定一致）。
