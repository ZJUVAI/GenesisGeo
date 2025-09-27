# 几何知识发现

## 0. 目录速览


- 1. 概述与背景 — 阐述当前几何自动证明的瓶颈、以频繁子图发掘新引理/定理的核心思路与预期贡献。
- 2. 目标与阶段里程碑 — 通过两阶段可行性验证与规则集拆分评估方案有效性，给出基础/完整规则集的解题率对比。
- 3. 数据与证明图构建 — 将证明文本解析为“fact-rule”分层有向二分图，定义节点/边、去重与一致性策略及作用域。
- 4. 方法 — 提出三类频繁子图挖掘器（路径/分叉/规则仅），配套支持度与结构约束、可读化 schema 与变量闭包检查。
- 5. 实现与代码结构 — 汇总近期功能更新、核心类与数据模型、并行管线与落盘格式，以及运行入口与日志/审计机制。
- 6. 剩余目标及日程规划 — 列出实验与论文的待办事项、时间表与阶段产出，以指导后续推进。

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

### 2.1 方案可行性验证 I

验证思路：目前的ddar规则集中，r07（Thales Theorem I）的结论可以通过添加辅助点来证明，包含r07的题目如果添加相应的辅助点，就可以不依赖r07求解。如果对一批这样的求解结果利用频繁子图搜索算法重新找出r07，就可以验证这个思路是可行的

  ```
  r07 Thales Theorem I
  para A B C D, coll O A C, ncoll O A B, coll O B D => eqratio3 A B C D O O
  ```

### 2.2 方案可行性验证 II

将通过“选择小规则集 → 评估小规则集解题能力 → 生成数据 → 定理挖掘 → 更新规则集并重新评估”的流程，验证方案的实际效果。为此，先对现有规则集进行整理：在当前 31 条规则中，可按“基础规则/派生定理”进行如下划分。

#### 基础规则：公理与定义(16条)

基础规则是系统的逻辑基石，定义了最核心的几何概念与关系，本身无需证明。

```
r28 Overlapping parallels
para A B A C => coll A B C
r34 AA Similarity of triangles (Direct)
eqangle B A B C Q P Q R, eqangle C A C B R P R Q, sameclock A B C P Q R => simtri A B C P Q R
r35 AA Similarity of triangles (Reverse)
eqangle B A B C Q R Q P, eqangle C A C B R Q R P, sameclock A B C P R Q => simtrir A B C P Q R
r49 Recognize center of cyclic (circle)
circle O A B C, cyclic A B C D => cong O A O D
r50 Recognize center of cyclic (cong)
cong O A O B, cong O C O D, cyclic A B C D, npara A B C D => cong O A O C
r51 Midpoint splits in two
midp M A B => rconst M A A B 1/2
r52 Properties of similar triangles (Direct)
simtri A B C P Q R => eqangle B A B C Q P Q R, eqratio B A B C Q P Q R
r53 Properties of similar triangles (Reverse)
simtrir A B C P Q R => eqangle B A B C Q R Q P, eqratio B A B C Q P Q R
r54 Definition of midpoint
cong M A M B, coll M A B => midp M A B
r56 Properties of midpoint (coll)
midp M A B => coll M A B
r60 SSS Similarity of triangles (Direct)
eqratio B A B C Q P Q R, eqratio C A C B R P R Q, sameclock A B C P Q R => simtri A B C P Q R
r61 SSS Similarity of triangles (Reverse)
eqratio B A B C Q P Q R, eqratio C A C B R P R Q, sameclock A B C P R Q => simtrir A B C P Q R
r62 SAS Similarity of triangles (Direct)
eqratio B A B C Q P Q R, eqangle B A B C Q P Q R, sameclock A B C P Q R => simtri A B C P Q R
r63 SAS Similarity of triangles (Reverse)
eqratio B A B C Q P Q R, eqangle B A B C Q R Q P, sameclock A B C P R Q => simtrir A B C P Q R
r101 Similarity to Congruence (Direct)
simtri A B C P Q R, cong A B P Q => contri A B C P Q R
r102 Similarity to Congruence (Reverse)
simtrir A B C P Q R, cong A B P Q => contrir A B C P Q R
```

- 定义性规则：定义几何概念的内涵与外延。
  - 中点定义（r51, r54, r56）：完整给出“何为中点”（r54）及其共线（r56）与二等分（r51）性质。
  - 相似性的性质（r52, r53）：定义 simtri（相似三角形）这一谓词的含义，一旦确立相似关系，可推出对应角相等与对应边成比例。
  - 相似到全等（r101, r102）：可视作 congtri 的定义途径之一：若两三角形相似且有一对对应边相等，则它们全等。
  - 圆心定义（r49, r50）：基于“圆上各点到圆心距离相等”的定义，用于识别与使用圆心。
- 公理化规则：源于欧几里得几何的基本公理。
  - 重叠平行线（r28）：平行公理的直接逻辑推论（过直线外一点仅有一条与已知直线平行的直线）。

#### 核心判定规则：建立等价关系的基石

此类规则虽在严格意义上可证，但在几何推理系统中承担着判定三角形相似的核心职责，是运用比例与角度关系的关键工具。

- AA 相似（r34, r35）：角-角相似准则。
- SSS 相似（r60, r61）：边-边-边相似准则。
- SAS 相似（r62, r63）：边-角-边相似准则。

将这些判定准则视作基础层级是合理的，因为它们是多数复杂定理证明的起点。

#### 派生定理：由基础规则构建的几何知识

派生定理在系统中数量最多，代表欧几里得几何中那些著名且需多步证明的定理。在 DDARN 引擎中将其固化为单步规则，能显著提升推理效率，避免每次从零展开冗长推导。

- 泰勒斯定理（r07, r27, r41, r42）：泰勒斯定理（平行线分线段成比例）及其逆定理，均可通过构造相似三角形（利用 AA 相似）证明。
- 圆几何定理（r03, r04, r19, r58, r59）：包括圆周角定理及其逆、等弦对等角、直角三角形斜边是外接圆直径等，可通过添加圆心、连接半径构造等腰三角形并利用基础角关系来证明。
- 著名三角形定理：
  - 角平分线定理（r11, r12）：可通过添加辅助平行线构造相似三角形来证明。
  - 勾股定理（r57）：经典做法是过直角顶点向斜边作高，利用产生的三个相似三角形（AA 相似）进行边长比例推导。
  - 垂心定理（r43）与内心定理（r46）：关于“四心”的更高级结论，证明常需综合运用多种规则，例如通过证明四点共圆获得新的角度关系。
- 高等几何定理：
  - 帕普斯定理（r44）：更高等的定理，射影几何框架下更简洁；在欧氏框架下证明也依赖反复应用相似与泰勒斯定理。

接下来，我们将检查提取到的基础规则集，记录对 benchmark jgex-231 的题目求解率；利用当前数据生成功能，基于基础规则集生成一批数据，提取其中包含辅助点的数据，并调用子图挖掘管线进行求解。

- 基础规则集的求解完成率：158/231
- 完全规则集的求解完成率：202/231

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

（完整的 data_discovery 代码与数据清单见文末“附录A”。）

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

- 求解输出增强：在 `scripts/run_batch.py` 生成的结果中，每题对象新增两项以便审计与后续绘图/重放：
  - `point_lines`: 形如 `point a x y` 的行列表（按点名排序）；
  - `points`: 结构化数组 `[{"name": str, "x": float, "y": float}]`。

并行与脚本更新（多进程 seeds_mproc）
- 新增“种子级并行挖掘”通道：
  - 子进程仅负责搜索（按 seed 扩展并通过 emit 推送原始模式对象），不做 schema/过滤。
  - 主进程集中完成 schema 转换、两层去重（结构签名 + 规范化 schema）、过滤（可选丢弃 unknown、变量闭包兜底与依赖过滤）与写入（流式或批量）。
- 脚本参数（`Newclid/scripts/mine_schemas.py`）：支持常见命令行参数，推荐编辑脚本顶部的 CONFIG 常量以控制引擎/阈值/剪枝/时间预算等。
- 全局扩展预算（重要变化）：
  - seeds_mproc 下，`debug_limit_expansions` 作为“全体子进程共享”的严格上限实现，使用 `multiprocessing.Semaphore` 作为跨进程预算；每次可导致结构增长的扩展尝试前均会尝试消耗 1 个配额，耗尽后所有 worker 将不再产生新扩展。
  - `single` 与 `seeds` 引擎下则为“当前一次运行/当前种子内”的本地上限。
- 死锁修复与可靠退出：
  - 工作队列采用 `JoinableQueue` + 阻塞式 `get()`；在启动前预投递与 worker 数相同的哨兵 `None`，worker 消费到哨兵后 `task_done()` 并退出。
  - 主进程对每次 `put` 调用都有匹配的 `task_done()`；`jobs.join()` 不再挂起。
  - 结果通道使用 `qout`；每个 worker 结束时会向 `qout` 发送一次 `None` 作为完成信号，主进程据此统计 worker 退出并收尾。
  - 加入 `maxsize` 以防止输出洪泛，emit 端采用带超时的 `put` 避免阻塞。
- 日志顺序：不再调用整体 `run_*`，而是在主进程汇总后统一记录耗时。
推荐入口脚本：
- `Newclid/scripts/mine_schemas.py`：挖掘（分叉/规则仅），写出结构化 patterns 与载荷
- `Newclid/scripts/filt_schemas.py`：读取挖掘结果进行筛选与审计，生成终态 JSON 与 NDJSON 审计
- `Newclid/scripts/render_mined_schemas.py`：将已筛选或原始挖掘结果进行可视化渲染

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
/usr/bin/env python3 Newclid/scripts/mine_schemas.py
/usr/bin/env python3 Newclid/scripts/filt_schemas.py
/usr/bin/env python3 Newclid/scripts/render_mined_schemas.py
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

#### 5.4.1 后处理职责拆分：SchemaMiner 与 SchemaFilter
- 入口：`MiningPipeline(pg, miner, args, out_dir).run(...)`。
- 最新职责边界：
  - SchemaMiner：核心挖掘与 schema 生成类（`src/newclid/data_discovery/schema_miner.py`）。由 `scripts/mine_schemas.py` 驱动，负责从输入结果构建证明图、运行分叉/规则仅挖掘、生成 `schema` 与 `schema_before_dependency`，并收集 `rendered/point_lines/points` 等载荷，写出 `branched_mining.json` 或 `rules_only_mining.json`。
  - SchemaFilter：独立类（`src/newclid/data_discovery/schema_filter.py`），负责 unknown 过滤、依赖过滤、变量闭包检查、两层去重与“同结论按前提集合最小化”，并产出各阶段审计输出。
  - 脚本入口：`scripts/filt_schemas.py`，以常量配置方式读取挖掘结果并调用 `SchemaFilter` 产出终态 JSON 与审计文件。
- 审计输出（由 SchemaFilter 生成，NDJSON，每行一条）：
  - step1：`audit_{mode}_step1_after_unknown.ndjson`
  - step2-kept：`audit_{mode}_step2_dep_kept.ndjson`
  - step2-dropped：`audit_{mode}_step2_dep_dropped.ndjson`
  - step3：`audit_{mode}_step3_after_varclosure.ndjson`
  - step4：`audit_{mode}_step4_final_dedup.ndjson`
  - step5-dropped：`audit_{mode}_step5_subset_min_dropped.ndjson`

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

#### 5.4.3 可视化增强使用示例（配合过滤环节）

当你完成“按点覆盖/按 rely_on 二次筛选”后，可直接基于筛选产物进行可视化，并按 union_rely 对前提节点分色、在图上标注 union_rely 集与 schema 文本，同时通过 legend 模式将长标签外移以避免拥挤。

- 入口类：`src/newclid/data_discovery/schema_visualizer.py` 中的 `SchemaVisualizer`
- 推荐调用（按二次筛选产物 partition_by_rely_on.json 分桶渲染）：

```python
from newclid.data_discovery.schema_visualizer import SchemaVisualizer

viz = SchemaVisualizer()
stats = viz.render_from_rely_on_file(
  partition_json_path="src/newclid/data_discovery/data/partition_by_rely_on.json",
  buckets=[
    "candidate_schemas",        # 通过第二次筛选的候选
    "candidate_schemas_type2",  # 互不包含情形
    "discarded_schemas",        # 被丢弃（并集等于/真子集等情况）
    "error_schemas",            # 点集不闭合等错误
  ],
  base_out_dir="src/newclid/data_discovery/data/schema_fig",
  max_items_per_bucket=50,          # 0 表示不限制
  overwrite=True,
  label_mode="legend",             # 将冗长标签移至右侧 Legend，提高可读性
  highlight_kind=True,              # 前提/中间/结论差异化底色/描边
  enable_union_rely_styling=True,   # 基于 union_rely 对前提进行二次分色
  style_opts={"figsize": (14, 9), "font_size": 8},
)
print(stats)
```

渲染效果说明：
- 规则节点为方形，fact 节点为圆形；唯一结论 fact 使用蓝色描边；
- 前提 fact 若其点集包含于 union_rely，底色为浅绿色；否则为浅橙色（便于与 union_rely 的差集对比）；
- 左上角会显示当前 schema 文本，左下角显示 `union_rely = { ... }`；
- `label_mode="legend"` 时，节点上使用短名（F1/F2/R1/C），右上角 Legend 显示短名与完整标签映射，避免节点内文字拥挤；
- 也可通过 `label_mode="short"` 显示谓词短名，或 `"full"` 直接显示完整 `pred(X,...)` 标签；
- 通过 `style_opts` 可覆盖画布大小与字体等；如需关闭“分类着色”，将 `highlight_kind=False`。

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

## 6. 剩余目标及日程规划

### 实验部分

- [x] __(9.11) 能够测试现有的schema是否满足要求（通过Yuclid实现）__
- [ ] __(9.12)调试挖掘代码：辅助点部分的正确性 + 超参（效率/数量）__
- [ ] (9.13)验证方案II
- [ ] (9.14)实验目标设计 & 消融实验设计
- [ ] __(9.15)主体实验管线__
- [ ] (9.18)主要实验结果
- [ ] (9.24)全部实验结果

### 论文部分

- [ ] (9.14)调研相关工作 relate works
- [ ] (9.16)method
- [ ] (9.18)experiment & abstract
- [ ] __(9.20)初稿__
- [ ] (9.25)完成提交

### 日程表

| 日期范围      | 任务内容           | 完成情况     |
|--------------|--------------------|-------------|
| 0908-0914    | 调试 + 跑通实验管线  |             |
|              | 设定实验目标        |             |
|              | 论文摘要            |            |
| 0915-0918    | 实验主要结果完成     |            |
|              | 完成论文初稿        |             |
| 0918-0921    | 补充实验结果        |             |
|              | 提交摘要           |             |
| 0922-0925    | 最终检查           |             |
|              | 投稿               |             |


## 附录

### 附录A：data_discovery 代码与数据目录清单

本附录基于 main ↔ discovery 的差异与当前目录扫描，统计与 data_discovery 功能有关的新增代码与生成数据。若后续有增删，将在附录处持续更新。

#### A.1 代码（按功能分类，列至文件级）

- 图构建与数据解析
  - `src/newclid/data_discovery/proof_graph.py`：解析 analysis/numerical_check/proof，构建题内证明图与合并图
  - `src/newclid/data_discovery/data_processor.py`：结果 JSON 读取/规范化与数据预处理工具
  - `src/newclid/data_discovery/solver_utils.py`：求解与外部工具适配的辅助函数
  - `src/newclid/data_discovery/expand_problem.py`：题目扩展/辅助点相关预处理
  - 测试
    - `src/newclid/data_discovery/test_graph_construction.py`：图构建单测
    - `tests/test_proof_graph.py`：证明图相关单测

- 规则抽取与管理
  - `src/newclid/data_discovery/rule_extractor.py`：从模式/图中抽取 schema/规则
  - `src/newclid/data_discovery/rules_manager.py`：规则集装载/切换/统计
  - 规则配置
    - `src/newclid/default_configs/rules_basic.txt`（新增，基础规则集）
    - `src/newclid/default_configs/rules_backup.txt`（新增，备份）
    - `src/newclid/default_configs/rules.txt`（修改，完整规则集）

- 挖掘/筛选/可视化
  - `src/newclid/data_discovery/schema_miner.py`：挖掘与 schema 生成（分叉/规则仅）
  - `src/newclid/data_discovery/schema_filter.py`：schema 筛选与审计（unknown/依赖/变量闭包/去重/最小化）
  - `src/newclid/data_discovery/schema_visualizer.py`：基于 rendered 的图可视化
  - 运行脚本（推荐在脚本顶部 CONFIG 设参）
    - `scripts/mine_schemas.py`：挖掘入口（分叉/规则仅），落盘 `branched_mining.json` / `rules_only_mining.json`
    - `scripts/filt_schemas.py`：筛选与审计入口，产出终态 JSON 与 NDJSON 审计
    - `scripts/render_mined_schemas.py`：可视化入口
    - `scripts/run_gspan_demo.py`：路径挖掘 demo（stdout）
    - `scripts/check_discovery.py`：发现流程自检/报告
  - 测试
    - `tests/test_gspan.py`、`tests/test_gspan_branched.py`、`tests/test_gspan_branched_step.py`：挖掘相关单测

- 批处理与评估
  - `scripts/schema_eval.py`：批量 schema 评估入口（对 schema 与 schema_before 进行两次高层处理）
  - `src/newclid/data_discovery/schema_evaluator.py`：Schema 批评估的实现（类/函数）
  - `src/newclid/data_discovery/iterative_rules_pipeline.py`：迭代规则集/评估的管线
  - 批量运行
    - `scripts/run_batch.py`：集中参数的批处理入口（建议使用此脚本）
    - `src/newclid/data_discovery/run_batch.py`：兼容保留（建议改用 scripts 版本）

- 数据与样例（输入/参考）
  - `src/newclid/data_discovery/r07_expanded_problems_results_lil.json`：r07 扩展结果（轻量版）
  - 其他（用于开发/统计）
    - `src/newclid/data_discovery/discovery_aux_data.jsonl`
    - `src/newclid/data_discovery/lil_data.jsonl`
    - `src/newclid/data_discovery/rules_with_discovery.txt`

- 其他工具
  - `scripts/translate_rule_to_problem.py`：规则到题目转写工具

- 包装与入口
  - `src/newclid/data_discovery/__init__.py`：模块导出
  - `src/newclid/data_discovery/summary_and_todo.md`：阶段记录（文档）

#### A.2 生成数据产物（结果与审计）

统一落盘目录：`src/newclid/data_discovery/data/`（审计为 NDJSON）。

| 文件 | 角色 | 生成脚本/来源 | 说明 |
|---|---|---|---|
| `branched_mining.json` | 分叉挖掘结果 | `scripts/mine_schemas.py` | 结构化结果（patterns、统计、params、timing 等） |
| `rules_only_mining.json` | 规则仅挖掘结果 | `scripts/mine_schemas.py` | 规则图挖掘，回溯重建 facts |
| `branched_mining.schema.results.json` | schema 评估结果（schema） | `scripts/schema_eval.py` | “一个输入一个输出”策略 |
| `branched_mining.schema_before.results.json` | schema 评估结果（schema_before） | `scripts/schema_eval.py` | 与 schema 分开落盘 |
| `branched_mining.schema.rules.txt` | 规则导出（schema） | `scripts/schema_eval.py`/后处理 | 人审/导入用 |
| `branched_mining.schema.split.txt` | schema 可读拆分/翻译报告 | `scripts/schema_eval.py` | 含 translate 失败原因与原 schema |
| `branched_mining.schema_before.split.txt` | schema_before 可读拆分/翻译报告 | `scripts/schema_eval.py` | 同上（before） |
| `audit_fact_rule_engine_expansions.ndjson` | 分叉挖掘引擎审计 | `scripts/mine_schemas.py` | 扩展尝试日志（seeds/expansions） |
| `audit_fact_rule_engine_seeds.ndjson` | 分叉挖掘引擎审计 | `scripts/mine_schemas.py` | 初始种子审计 |
| `audit_fact_rule_step1_after_unknown.ndjson` | 分叉筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | unknown 过滤后样本 |
| `audit_fact_rule_step2_dep_kept.ndjson` | 分叉筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 依赖过滤保留/裁剪明细 |
| `audit_fact_rule_step2_dep_dropped.ndjson` | 分叉筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 依赖过滤丢弃原因 |
| `audit_fact_rule_step3_after_varclosure.ndjson` | 分叉筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 变量闭包后样本 |
| `audit_fact_rule_step4_final_dedup.ndjson` | 分叉筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 两层去重后样本 |
| `audit_fact_rule_step5_subset_min_dropped.ndjson` | 分叉筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | “前提集合最小化”丢弃记录 |
| `audit_rule_only_engine_expansions.ndjson` | 规则仅引擎审计 | `scripts/mine_schemas.py` | 扩展尝试日志（seeds/expansions） |
| `audit_rule_only_engine_seeds.ndjson` | 规则仅引擎审计 | `scripts/mine_schemas.py` | 初始种子审计 |
| `audit_rule_only_step1_after_unknown.ndjson` | 规则仅筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | unknown 过滤后样本 |
| `audit_rule_only_step2_dep_kept.ndjson` | 规则仅筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 依赖过滤保留/裁剪明细 |
| `audit_rule_only_step2_dep_dropped.ndjson` | 规则仅筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 依赖过滤丢弃原因 |
| `audit_rule_only_step3_after_varclosure.ndjson` | 规则仅筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 变量闭包后样本 |
| `audit_rule_only_step4_final_dedup.ndjson` | 规则仅筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | 两层去重后样本 |
| `audit_rule_only_step5_subset_min_dropped.ndjson` | 规则仅筛选审计 | `scripts/filt_schemas.py`（SchemaFilter） | “前提集合最小化”丢弃记录 |
| `tmp_fact_rule_patterns.ndjson` | 中间模式快照 | 挖掘阶段 | 原始模式流式缓存 |
| `sch_split_test.txt` | 调试/单测小样 | 工具调试 | 翻译/拆分快速校验 |

补充：输入/参考数据（位于 `src/newclid/data_discovery/`）
- `r07_expanded_problems_results_lil.json`
- 其他：`discovery_aux_data.jsonl`、`lil_data.jsonl`、`rules_with_discovery.txt`

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
