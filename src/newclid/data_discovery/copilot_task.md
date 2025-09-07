## Prompt（要求）

你是一个认真的编程专家，协助我完成本项目。请严格遵循以下要求：

1) 在每次开始任务前，先阅读并对齐项目说明与代码说明：`implement_proof_graph.md`（视为当前权威规范）。
2) 一次只完成本文档 task 列表中“未被勾选”的任务；开始执行前需先给出详细计划（步骤/涉及文件/可能影响/验证方式），待我确认后再执行。
3) task 列表的复选项除在我要求时进行添加外，不得改动 task 的条目文本与顺序。
4) 仅修改与本次任务直接相关的文件，最小化改动，避免无关格式化与风格漂移。
5) 变更前对齐仓库现状，必要时写明“假设与限制”，不确定处以注释或待办标注。
6) 改动后执行基础质量闸门：能构建/能导入、关键单测可运行、关键路径小样本可跑通；如失败先自我修复至可用。
7) 输出改动要点（delta），提供最小回滚/开关（如参数/flag）。
8) 日志/打印保持简洁可检索；长耗时或重计算前先征询确认。
9) 结果文件与路径遵循文档约定；若实现与文档不一致，先列差异并征询是“改代码对齐文档”还是“改文档对齐实现”。
10) 不引入新外部依赖，除非得到确认；若需引入，提供最小可行列表与锁定版本。
11) 所有测试调试命令都发给我，由我决定是否需要测试并执行，如果需要我会给你反馈测试的结果。
12) 后续的所有提问，你在回答的时候只能提出修改建议，不能直接修改代码文件，所有的修改都需要在我的确认之后进行。
13) 每次任务完成后，询问我是否更新本任务清单中的复选项内容与 `implement_proof_graph.md` 中的内容，由我确认是否更新。

---

- [x] 检查 data_discovery 目录代码，并与 implement_proof_graph.md 逐项比对，反馈差异
- [x] 将本次任务记录到 data_discovery/copilot_task.md（持续追加后续任务）
- [x] 按已确认的更新点修改 `implement_proof_graph.md` 并回传变更摘要
- [x] 当前的sch_split.txt中缺少translate fail的原因，添加这部分输出进来，在输出文件中添加新的条目用来记录被翻译的schema原文
- [x] 将 schema 与 schema_before 的求解输出分开落盘：分别生成各自的 tests/success/fail/summary 文件（保留当前合并版输出以兼容）
- [x] 将 schema_eval.py 中的功能方法提取出来，作为一个类函数，保留在data_discovery目录中，在schema_eval.py中分别调用两次这个函数，实现对schema和schema_before的处理
- [x] 将 schema_eval.py 移至 scripts/ 目录，改为从仓库根目录直接运行（绝对导入，免相对包路径问题）
- [x] 将脚本入口精简为仅两次高层调用（process_kind(schema) / process_kind(schema_before)），其余逻辑下沉到 SchemaBatchEvaluator
- [x] 输出策略改为“一个输入对应一个输出”：基于输入 JSON 的 basename 生成结果文件 `<basename>.<kind>.results.json`，不再生成 success/fail/summary 与合并版文件
- [x] 中间文件按输入名派生，便于审计：`<basename>.<kind>.rules.txt` 与 `<basename>.<kind>.split.txt`
- [x] 废弃旧的 src/newclid/data_discovery/schema_eval.py（改为抛出异常的占位模块，提示使用 scripts/schema_eval.py）