import logging
from typing import List, Dict, Any
from newclid.proof_scout.core.proof_graph import ProofGraph


class GraphPruner:
    """
    实现独占子图规则提取与剥离算法 (Exclusive Subgraph Extraction & Pruning)。
    """

    def __init__(self, verbose: bool = False):
        self.logger = logging.getLogger("GraphPruner")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    def prune_and_extract(self, pg: ProofGraph) -> List[Dict[str, Any]]:
        """
        执行算法流程：
        1. 筛选 Alive 节点 (非 Aux 的 Fact)。
        2. 遍历检查是否依赖 Aux，且是否构成独占子图。
        3. 提取子图并根据独占性更新 Alive 集合。

        Returns:
            sub_PGs: 提取出的子图数据列表。
        """
        # 确保图的邻接表已构建
        if not hasattr(pg, "_adj_in"):
            pg.build_adjacency()

        extracted_subgraphs: List[Dict[str, Any]] = []

        # 1. 初始化待检查队列 (Alive)
        # 筛选: IsFact(node) and not HasAux(node)
        alive_ids = list(pg.nodes.keys())

        # 按 layer 升序排序
        alive_ids.sort(key=lambda nid: pg.nodes[nid]["layer"])

        # 使用 Set 加速查找，alive_ids 列表用于保持遍历顺序
        alive_set = set(alive_ids)

        self.logger.info(f"Initialized Alive set with {len(alive_ids)} nodes.")

        sub_idx = 0  # 子图索引，用于标识提取的子图

        # 2. 迭代检查每个事实节点 F0
        # 注意：我们需要遍历 alive_ids 的副本，因为 alive_set 可能会在过程中缩减
        for f0_id in alive_ids:
            if pg.nodes[f0_id]["type"] != "fact" or pg.nodes[f0_id]["is_aux"]:
                continue  # 仅处理不带辅助点的事实节点

            # === Step 2.1: 寻找活着的祖先 (Ancestors Discovery) ===
            # 在全图中反向搜索 F0 的所有上游节点
            ancestors = pg.get_ancestors(f0_id)
            for ancestor in list(ancestors):
                # 只保留活着的事实节点
                if ancestor not in alive_set:
                    ancestors.remove(ancestor)

            # 若无祖先 (Layer 0 Fact)，跳过
            if not ancestors:
                continue

            # === Step 2.2: 基于辅助点信息判定提取资格 (Extraction Eligibility) ===
            # F0 必然有一个直接规则前驱 R0 (因为它是 Fact 且有祖先)
            # 注意：ProofGraph 中 Fact 的前驱只能是 Rule (除了 Layer 0)
            rule = pg.get_predecessors(f0_id)
            if not rule:
                continue  # 理论上不应发生

            r0_id = rule[0]  # Fact 通常只有一个生成规则
            r0_node = pg.nodes.get(r0_id)

            should_extract = False
            if r0_node["is_aux"] is True:
                should_extract = True

            # === Step 2.3: 独占性检查 (Isolation Check) ===
            # 检查 Ancestors + {F0} 是否对 "当前还活着的外部世界" 封闭
            is_isolated = True

            # 定义当前子图涉及的所有节点
            subgraph_nodes_set = ancestors.copy()
            subgraph_nodes_set.add(f0_id)

            for ancestor_id in ancestors:
                # 获取 A 的所有连接点 (前驱 + 后继)
                neighbors = pg.get_predecessors(ancestor_id) + pg.get_successors(
                    ancestor_id
                )

                for n_id in neighbors:
                    # 关键：只关注当前还“活着”的外部节点
                    # 这里的“活着”指的是 alive_set 中的 Fact 节点
                    # 注意：图中的 Rule 节点不直接在 alive_set 中，但如果 Rule 连接到了活着的 Fact，
                    # 那么该 Rule 的上游也就间接连接到了活着的 Fact。
                    # 简化逻辑：如果在 Neighbors 中发现了 alive_set 中的节点，且该节点不在当前子图中，则不隔离。

                    if n_id in alive_set:
                        # 如果这个活着的邻居不是 Ancestors 内部节点，也不是目标 F0
                        if n_id not in subgraph_nodes_set:
                            is_isolated = False
                            break

                if not is_isolated:
                    break

            # === Step 2.4: 执行操作 ===

            # 操作 A: 提取规则
            if should_extract:
                # *** 关键修改：调用新的 create_subgraph 方法 ***
                sub_pg: ProofGraph = pg.create_subgraph(subgraph_nodes_set)
                sub_pg.problem_id = (
                    pg.problem_id + "sub_" + str(sub_idx)
                )  # 保留原图的 problem_id
                sub_idx += 1  # 更新子图索引

                # 将子图对象及其元数据添加到列表中
                extracted_subgraphs.append(
                    {
                        "subgraph_object": sub_pg,  # 新的 ProofGraph 对象
                        "target_node_id": f0_id,  # 目标节点 ID
                        "node_count": len(subgraph_nodes_set),  # 方便快速查看统计信息
                    }
                )

                self.logger.info(
                    f"Extracted subgraph for {f0_id}, Node count: {len(subgraph_nodes_set)}, Isolated: {is_isolated}"
                )

            # 操作 B: 剥离子图 (仅修改搜索空间 alive_set，不修改物理图)
            if is_isolated:
                # 将 Ancestors 中属于 Fact 的部分从 Alive Set 中移除
                # 因为它们已经作为独立子图的一部分被处理了，不需要再作为其他分支的起点或干扰项
                removed_count = 0
                for a_id in ancestors:
                    if a_id in alive_set:
                        alive_set.remove(a_id)
                        removed_count += 1
                self.logger.info(
                    f"Pruned {removed_count} ancestors from Alive set for {f0_id}"
                )

        return extracted_subgraphs


# ---------------------------------------------------------
# 测试用例 (Optional)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 使用与 proof_graph.py 相同的示例数据
    sample_data = {
        "problem_id": 8,
        "llm_input_renamed": "<problem> a : ; b : ; c : ; d : coll b c d [000] cong b d c d [001] ; e : coll a c e [002] cong a e c e [003] ? simtri a b c e d c </problem>",
        "llm_output_renamed": "<aux> x00 f : coll a b f [004] cong a f b f [005] ; </aux> <numerical_check> sameclock a b c c e d [006] ; </numerical_check> <proof> eqangle a c b c c e c d [007] AR [002] [000] ; eqratio a f a e b f c e [008] AR [005] [003] ; eqratio a b a c a f a e [009] r105 [004] [002] [008] ; eqratio a f b f b d c d [010] AR [005] [001] ; eqratio a b a f b c b d [011] r105 [004] [000] [010] ; eqratio a c b c c e c d [012] AR [003] [001] [009] [011] ; simtri a b c e d c [013] r62 [007] [012] [006] ; </proof>",
    }

    # 1. 构建图
    pg = ProofGraph(verbose=True)
    pg.build_from_json(sample_data)

    # 记得构建邻接表
    pg.build_adjacency()

    # 2. 运行 Pruner
    pruner = GraphPruner(verbose=True)
    sub_graphs = pruner.prune_and_extract(pg)
    print(f"\nTotal subgraphs extracted: {len(sub_graphs)}")
    for i, sub in enumerate(sub_graphs):
        print(
            f"Subgraph {i + 1}: Target {sub.get('target_node_id')}, Nodes: {sub.get('node_count')}, Isolated: {sub.get('is_isolated')}"
        )

    for sub in sub_graphs:
        sub_pg = sub["subgraph_object"]
        sub_pg.print_graph()
        sub_pg.to_json_data()
        print(sub_pg.export_to_rule_format())
