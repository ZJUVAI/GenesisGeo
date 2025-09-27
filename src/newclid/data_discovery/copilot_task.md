## Prompt（要求）

你是一个认真的编程专家，协助我完成本项目。请严格遵循以下要求：

1) 在每次开始任务前，先阅读并对齐项目说明与代码说明：`geometry_knowledge_discovery.md`（视为当前权威规范）。
2) 你可以阅读全部任务，了解我的整体规划，但是一次只完成本文档 task 列表中“未被勾选”的任务中的第一条；开始执行前需先给出详细计划（步骤/涉及文件/可能影响/验证方式），待我确认后再执行。
3) task 列表的复选项除在我要求时进行添加外，不得改动 task 的条目文本与顺序。
4) 仅修改与本次任务直接相关的文件，最小化改动，避免无关格式化与风格漂移。
5) 变更前对齐仓库现状，必要时写明“假设与限制”，不确定处以注释或待办标注。
6) 改动后执行基础质量闸门：能构建/能导入、关键单测可运行、关键路径小样本可跑通；如失败先自我修复至可用。
7) 输出改动要点（delta）。
8) 日志/打印保持简洁可检索；长耗时或重计算前先征询确认。
9) 结果文件与路径遵循文档约定；若实现与文档不一致，先列差异并征询是“改代码对齐文档”还是“改文档对齐实现”。
10) 不引入新外部依赖，除非得到确认；若需引入，提供最小可行列表与锁定版本。
11) 所有测试调试命令都发给我，由我决定是否需要测试并执行，如果需要我会给你反馈测试的结果。
12) 后续的所有提问，你在回答的时候只能提出修改建议，不能直接修改代码文件，所有的修改都需要在我的确认之后进行。
13) 每次任务完成后，询问我是否更新本次完成的复选项内容具体说明与 `geometry_knowledge_discovery.md` 中的内容，由我确认是否更新。
14) data的内容是作为记录使用，你不需要在意。

---

date: 0907

- [x] 检查 data_discovery 目录代码，并与 geometry_knowledge_discovery.md 逐项比对，反馈差异
- [x] 将本次任务记录到 data_discovery/copilot_task.md（持续追加后续任务）
- [x] 按已确认的更新点修改 `geometry_knowledge_discovery.md` 并回传变更摘要
- [x] 当前的sch_split.txt中缺少translate fail的原因，添加这部分输出进来，在输出文件中添加新的条目用来记录被翻译的schema原文
- [x] 将 schema 与 schema_before 的求解输出分开落盘：分别生成各自的 tests/success/fail/summary 文件（保留当前合并版输出以兼容）
- [x] 将 schema_eval.py 中的功能方法提取出来，作为一个类函数，保留在data_discovery目录中，在schema_eval.py中分别调用两次这个函数，实现对schema和schema_before的处理
- [x] 将 schema_eval.py 移至 scripts/ 目录，改为从仓库根目录直接运行（绝对导入，免相对包路径问题）
- [x] 将脚本入口精简为仅两次高层调用（process_kind(schema) / process_kind(schema_before)），其余逻辑下沉到 SchemaBatchEvaluator
- [x] 输出策略改为“一个输入对应一个输出”：基于输入 JSON 的 basename 生成结果文件 `<basename>.<kind>.results.json`，不再生成 success/fail/summary 与合并版文件
- [x] 中间文件按输入名派生，便于审计：`<basename>.<kind>.rules.txt` 与 `<basename>.<kind>.split.txt`
- [x] 废弃旧的 src/newclid/data_discovery/schema_eval.py（改为抛出异常的占位模块，提示使用 scripts/schema_eval.py）
- [x] 重新组织geometry_knowledge_discovery.md中的内容，重命名为geometry_knowledge_discovery.md。主题改为几何知识发现，从介绍背景出发，指出需要实现的目标，当前的第一步是完成r07规则的重新发现，然后介绍具体的方法，不添加额外信息的情况下重新组织结构，先制定新的大纲，我确认后开始执行。

date: 0908

- [x] 更新文档，添加第二步：基础规则集扩展的任务描述及规划
- [x] 修改 run_batch.py，将其放入 scripts 目录中，其他路径不做修改；仿照 run_gspan_branched_demo.py 的脚本逻辑，将需要设定的超参集中于脚本文件开头，命令行只需执行 python run_batch.py
- [x] 检查rules_basic.txt，确认jgex可以全build，记录题目完成率
- [x] 利用当前generate.sh脚本，修改调用的generate.py文件，改用rules_basic.txt生成一批数据(100k) -> aux: 15k

date: 0909

- [x] 统计现在与data_discovery功能有关的新代码与数据，结合git的记录，文档的内容进行整理，将代码文件的目录整理写在geometry_knowledge_discovery.md中
- [x] 在现有的schema筛选部分代码中添加一步过滤，在生成schema_before后，先过滤掉sameclock sameside这两个premises，然后以过滤后的schema作为输入进行后续的过滤

date: 0910

- [x] 将translate_rule_to_problem.py中的函数提取出来，改成一个类似于tests/test_solve.py的简单测试函数，根据branched_mining.json中的文件格式，给一个简单的测试，方便我进行调试检查我希望能够完整迁移translate_rule_to_problem.py中translate_premise整个函数以及其中调用的若干translate_*函数及辅助函数，实现输入类似eqratio(X1,X2,X3,X2,X4,X2,X5,X2) => eqratio3(X1,X3,X4,X5,X2,X2)的schema格式，输出x5 = free x5; x4 = free x4; x3 = free x3; x2 = free x2; x1 = eqratio x1 x5 x2 x4 x2 x3 x2 x2 ? eqratio3 x1 x3 x4 x5 x2 x2这样的dsl语言题目。整个测试代码文件可以复制所有需要的功能独立完成上述流程，这样我通过调试这个测试代码就可以知道translate_rule_to_problem.py中哪里存在需要修复的bug

date: 0911

- [x] 确认当前的solve部分代码，找到画图与确定题目中点的坐标部分的代码，之后更新solver_utils.py，在当前求解部分将这些点的坐标添加进来，类似于
    ```
    point a -0.40746902984670474446 -0.66665500610231753775
    point b -0.11560335689440992546 -0.59371690741631888422
    point g1 -0.44257354475251259318 -0.52618242659683756024
    point g2 -0.17244267242498786952 -0.36627136967481344065
    point m -0.40604502710432055501 -0.38607339777242094536
    ```
    的格式，作为新的一项存储在输出文件中对应的题目项下
- [x] 更新子图挖掘的代码，在输出的schema中保留一份各点的坐标信息，格式如下：
    ```
        "point_lines": [
        "point a -0.49216369270784277 0.0322773499542357",
        "point b -0.34837646691822327 0.795407168565341",
        "point c -1.104024433863752 1.044882521415065",
        "point m -0.420270079813033 0.41384225925978835",
        "point n -0.27917315628915673 0.38725709025424193",
        "point o -0.8656498763618392 0.4977597314993747"
      ],
      "points": [
        {
          "name": "a",
          "x": -0.49216369270784277,
          "y": 0.0322773499542357
        },
        {
          "name": "b",
          "x": -0.34837646691822327,
          "y": 0.795407168565341
        },
        {
          "name": "c",
          "x": -1.104024433863752,
          "y": 1.044882521415065
        },
        {
          "name": "m",
          "x": -0.420270079813033,
          "y": 0.41384225925978835
        },
        {
          "name": "n",
          "x": -0.27917315628915673,
          "y": 0.38725709025424193
        },
        {
          "name": "o",
          "x": -0.8656498763618392,
          "y": 0.4977597314993747
        }
      ],
    ```
    更详细的格式可以参考outputs/jgex_ag_231_results.json中的内容，对每个schema只需要记录一个代表的坐标值即可，将这部分信息补充在输出的json文件中
- [x] 梳理当前run_gspan_branched_demo.py及其依赖的代码文件，精简其中的实现与注释内容，并与当前文档中描述进行比对，对文档内容进行精简和更新
- [x] 在scripts目录下新创建一个脚本文件，目标是将src/newclid/data_discovery/data/branched_mining.json文件中的schema和schema_before_dependency，结合对应的点坐标信息，重新整理成若干个txt的题目，每个题目的形式与src/newclid/data_discovery/data/schema_tests/example.txt的格式相同，文件名称为schema_{id}.txt和schema_before_{id}.txt，其中id是从0000开始，按顺序到9999的编号，这批文件最终存储在src/newclid/data_discovery/data/schema_tests/目录下。该脚本文件应该不需要依赖其他库中的代码文件。
- [x] 在scripts中写一个脚本代码，可以将指定文件（如outputs/data/geometry_clauses13_samples100k.jsonl）中的fl_problem项提取出来，整理成类似src/newclid/data_discovery/data/r07_problems.txt这种格式，整理后写成一个txt文件，存放在data_discovery/data/目录下该脚本文件应该不需要依赖其他库中的代码文件。

date: 0912

- [x] 统计目前的子图挖掘部分代码是如何进行schema筛选的，然后将筛选步骤的代码移出重新整理成一个类函数，并用单独的脚本文件（写在scripts目录下）进行调用，每个阶段单独保存输出文件,类函数代码名称为schema_filter.py；脚本文件名字取filt_schemas.py，同时还要更新make_schema_tests.py的格式，使得新的输出schema和不同的审计文件都可以后续经过make_schema_tests.py管线，make_schema_tests.py的cli选项都放在代码开头部分，通过手动在文件内修改超参来实现，我希望运行脚本文件只需要在命令行中输入python make_schema_tests.py
  备注：已新增 `src/newclid/data_discovery/schema_filter.py` 与 `scripts/filt_schemas.py` 并通过实际数据验证可用；`mining_pipeline.py` 已去除筛选/审计仅保留挖掘与schema生成。`make_schema_tests.py` 的常量化改造暂未执行，后续单独处理。
- [x] 跑通yuclid的流程，目前的情况：473 success + 26 fail to solve + 160 error(ncoll + eqratio3)

date: 0915

- [x] 检查mining_pipeline.py代码，输出的结果确实是正确的，并非一开始猜测的推理结果有误。

date: 0916

- [x] 在现在的schema_filter.py中添加一个新的函数，它用来进行结果的检查及过滤，方法如下：首先确认schema中每个premise的点集的并集和conclusion的点集，如果并集是点集的真子集，存放在error_schemas中；如果并集等于点集，则存放在discarded_schemas中，余下的存放在candidate_schemas中。
- [x] 设计一个python脚本，实现schema的图可视化方案，能够根据rendered中的信息将schema进行可视化，输出格式可以为图片或图形格式的文本，你来给我提供一些可能的参考
- [x] 根据每个schema中的rely_on信息，对candidate_schemas进行第二次筛选，规则如下：首先确认schema中conclusion中点及其rely_on组成union_rely集合，与premise中的点集的并集pre_union进行比较，会得到四种结果，根据不同的结果对schema进行分类：1. pre_union等于union_rely，此时schema被分类到discarded_schemas；2. pre_union是union_rely的真子集，此时schema被分类到discarded_schemas；3. union_rely是pre_union的真子集，此时schema被分类到candidate_schemas；4. union_rely和pre_union互不包含，此时schema被分类到candidate_schemas_type2中。新的过滤环节只输入partition_by_point_coverage.json中的candidate_schemas进行处理，输出到partition_by_rely_on.json中，同样保留rendered等附带信息。

date: 0917
- [x] 将第二次筛选过的schema接入visualize_schemas.py中进行可视化，输出到schema_fig目录下
- [x] 整理目前的挖掘-筛选-可视化代码，使其整体脉络更清晰，并更新文档
- [x] 进一步优化可视化环节，包括如下几部分：1. 目前schema的内容还是会超出圆圈的范围，尝试调整布局方式，使得或者圆圈中内容在外部指代或者如何调整字体大小，给出一个可行方案；2. 在图中添加union_rely的信息；3. 为fact node添加更多中标识方法，除了现在的绿色和蓝色表示premise和conclusion外，我还希望提供更多可供操作的接口，这样后续我可以对premise根据其点集是否包含在union_rely中进行区分；4. 完成1 2 3后开始对schema图中premise进行进一步细分的颜色标识区分。
- [x] 将filt/mine/visualize_schemas.py和schema_filter/miner/visualizer.py的内容更新到文档中，并去除文档中mining_pipeline.py和run_gspan_branched_demo.py的内容

date: 0918
- [x] 更新schema_visualizer.py，使得所有fact node都添加颜色标识，判断依据还是单个premise的点集是否包含在union_rely中，包含则为绿色，不包含则为橙色，conclusion为蓝色，用在schema中的前提（入度为0）边框加粗

date: 0919
- [x] 在二次筛选后再添加一个筛选，对于candidate_schemas和candidate_schemas_type2进行筛选，规则如下：对每个schema，检查其对应的图结构中的rule node连接的fact node，如果有一个rule node连接的全部fact node都满足其点集包含在union_rely中，则该schema被分类到discarded_schemas中，否则分类到final_candidate_schemas中
- [x] 将三次筛选的结果接入可视化功能
- [x] 修改第三次筛选的代码，使其输出的json中格式与第一次/第二次筛选的输出一致，先整理第一次第二次筛选后json文件的条目，然后给出第三次筛选的json文件条目修改方案。
- [x] 将第二阶段筛选（"stage" 为 "rely_on"）中的candidate_schemas和candidate_schemas_type2两类合并成candidate_schemas输出，第三阶段筛选后的绘图也像第二阶段一样在命令行中添加进度显示消息，之后整理第三阶段筛选（"stage"为"pruning"）的规则，我来指导如何进行修改

date: 0924
- [x] 检查当前scripts/run_batch.py管线，查看如何输出辅助构造信息（Auxiliary Constructions），将该功能添加进来后可以将辅助构造信息加在输出的json中
- [x] 整理现有的proof_graph的代码，整理成一个类函数，说明现在的功能及节点中存储的信息，之后按我的要求添加更多存储信息（辅助点信息、依赖关系）

date: 0925
- [x] 将schema_visualizer.py中的可视化功能复制到新的proof_graph中，整理现有的绘制规则，在我的要求下进行修改
- [x] 整理现在的类函数代码和所有脚本文件，梳理代码，去除多余的函数，修改后我来进行测试，确认没问题之后进行下一步开发
- [x] 参考schema_miner.py和mine_schemas.py重新设计一个子图提取的管线，只处理包含辅助点的题目；给定输入proof_graph后，从结论节点出发，向上挖掘，最终得到的子图为入度为0的节点都是fact node，出度为0的节点为结论节点。这部分功能保存在一个python文件中，在scripts目录下再保留一个脚本文件用来调用这个功能，输入就是json格式的数据，存放在类似r07_expanded_problems.results.json中，输出的格式也是json格式文件和得到的子图的可视化渲染结果。
- [x] 调整绘图的细节：1. 像schema_visualizer.py中一样，在图片左上角添加一个“前提-> 结论”的标签； 2. 现在的很多结果中节点的间距，特别是层与层之间的间距太小，需要调整，可以将画布调大一些。
- [x] 现在的proof_graph_visualizer中的绘图功能中对节点的染色规则需要更新：先整理目前的绘制规则，然后按如下需求重新设定规则：对结论节点绘制成蓝色；其余所有fact node中，包含辅助点的fact node染成橙色，否则染成绿色；对前提（入度为0）的节点进行边框加粗。

date: 0926
- [x] 完全重做extract_aux_subgraph.py以及调用的类函数，类函数改成aux_extractor.py，脚本函数改成extract_aux_graph.py，这个类函数只用来判断证明过程中aux_points是否是空的，只保留非空的证明结果，然后extract_aux_graph.py输入json文件，进行筛选后输出到一个带_aux后缀的json文件中，并且完全参照plot_proof_graphs.py的使用方法对这个带aux后缀的json文件进行可视化，最终输出到proof_graphs目录下
- [x] 添加一个新的类函数代码graph_pruner.py和一个脚本代码prune_graphs.py，按如下规则进行证明图的修剪：对每个规则节点进行判断，如果指向它的fact node全部都是题目的前提（在图中对应加粗边框的fact node），且与它相连的所有fact node都是不包含辅助点的fact node（在图中对应绿色节点的前提），则删去这个规则节点及与周围fact node相连的边；将它指向的fact node改为题目的前提，并检查是否有前提节点没有与任何规则节点相连，如果有则也删去这个fact node，迭代删减直到没有符合条件的规则节点出现位置。输出结果方面，我需要输出的格式和输入的格式尽可能保持一致，仍然能通过proof_graph_visualizer进行可视化，并且包含aux_points的信息。

date: 0927
- [ ] 参考schema_visualizer.py，为proof_graph_visualizer添加legend模式
- [ ] 将现在的aux_

- [ ] 从结果出发，并对aux结果打标 设置support=1 同时修改绘图管线 之后进行筛选
- [ ] 跑一下rule_translate的题目集，然后绘制一下结果图，和现在的schema进行比对，找一下规律
- [ ] 创建一个工具脚本，通过重新用ddar求解翻译后的题目，比对前后premises的差别，判断translate代码的正确性并进行调试
- [ ] 调试translate代码，逐个检查失败翻译的情况，确认失败原因后进行对应的修正（无法翻译的需要解决翻译的问题，无法求解的需要分析原因，可能是正确的结果也可能是翻译错误）
- [ ] 将修改好后的翻译代码改成一个工具类放在data_discovery目录下
- [ ] 检查现有的挖掘代码，只保留分叉图挖掘的模式，根据现有结果，验证schema筛选环节的正确性，并进行相应调试
- [ ] 整理现有的工具代码，实现 ‘挖掘-翻译验证-整理’ 的部分管线流程打通
- [ ] 添加一个从生成数据到题目集的脚本代码，调用处理好的翻译功能，对齐数据格式
- [ ] 将新数据中的辅助点数据转换成题目集后调用子图挖掘管线，实现 ‘数据-题目-挖掘-翻译验证-整理’ 的全流程管线
- [ ] 在schema_eval.py后添加规则导出功能
- [ ] 测试添加规则的新规则集求解jgex的完成率
- [ ] 规划第三步的大型测试
