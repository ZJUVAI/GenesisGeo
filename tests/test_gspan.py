import os
import sys
import logging
import unittest

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from newclid.data_discovery.proof_graph import ProofGraph, GSpanMiner  # noqa: E402


class GSpanTests(unittest.TestCase):
    def build_small_pg(self) -> ProofGraph:
        g = ProofGraph(verbose=False, log_level=logging.INFO)
        # problem A
        resA = {
            "problem_id": "A",
            "proof": {
                "analysis": "<analysis> coll f f0 h0 [000] ; coll a c f0 [001] ; coll c f e0 [002] ; cong a f0 c f0 [003] ; cong c e0 f e0 [004] ; coll a e0 h0 [005] ; coll g s v [006] ; midp b0 q v [007] ; coll g s u [008] ; cong g u u v [009] ; </analysis>",
                "numerical_check": "<numerical_check> ncoll f e0 f0 [010] ; sameside e0 c f f0 a c [011] ; sameclock a f h0 e0 f0 h0 [012] ; sameclock g q v u b0 v [013] ; ncoll f f0 g0 [014] ; sameside f0 a c g0 a f [015] ; sameclock f e0 g0 e0 f0 g0 [016] ; </numerical_check>",
                "proof": "<proof> eqratio a f0 c f0 f e0 c e0 [017] a00 [003] [004] ; para a f e0 f0 [018] r27 [001] [002] [017] [010] [011] ; eqangle a f f h0 e0 f0 f0 h0 [019] a01 [000] [018] ; eqangle a h0 f h0 e0 h0 f0 h0 [020] a01 [005] [000] ; simtri a f h0 e0 f0 h0 [021] r34 [019] [020] [012] ; eqratio a f f h0 e0 f0 f0 h0 [022] r52 [021] ; eqratio a h0 f h0 e0 h0 f0 h0 [023] r52 [021] ; eqratio a f0 a g0 c f0 f g0 [024] a00 [003] [025] ; para c f f0 g0 [026] r27 [001] [027] [024] [014] [015] ; eqangle f e0 f g0 f0 g0 e0 f0 [028] a01 [027] [002] [018] [026] ; eqangle f g0 e0 f0 e0 g0 e0 g0 [029] a01 [027] [018] ; simtri f e0 g0 f0 g0 e0 [030] r34 [028] [029] [016] ; eqratio f g0 e0 f0 e0 g0 e0 g0 [031] r52 [030] ; coll q v b0 [032] r56 [007] ; eqangle g v q v u v v b0 [033] a01 [006] [032] ; coll g u v [034] Same [008] [006] ; midp u g v [035] r54 [034] [009] ; rconst g v u v 2/1 [036] r51 [035] ; rconst q v b0 v 2/1 [037] r51 [007] ; eqratio g v q v u v v b0 [038] a00 [036] [037] ; simtri g q v u b0 v [039] r62 [033] [038] [013] ; eqratio g q q v u b0 v b0 [040] r52 [039] ; midp g0 a f [041] r54 [027] [025] ; rconst a f f g0 2/1 [042] r51 [041] ; eqratio a h0 g q e0 h0 u b0 [043] a00 [022] [023] [031] [040] [042] [037] ; </proof>"
            },
        }
        # problem B（结构相似）
        resB = {
            "problem_id": "B",
            "proof": {
                "analysis": "<analysis> coll a i b0 [000] ; coll a o c0 [001] ; coll i o a0 [002] ; cong a c0 o c0 [003] ; cong i a0 o a0 [004] ; coll o b0 d0 [005] ; cong a b0 i b0 [006] ; coll a a0 d0 [007] ; </analysis>",
                "numerical_check": "<numerical_check> ncoll i a0 c0 [008] ; sameside a0 i o c0 a o [009] ; ncoll o b0 c0 [010] ; sameside b0 a i c0 a o [011] ; ncoll o a0 b0 [012] ; sameside a0 i o b0 a i [013] ; sameclock a o d0 a0 b0 d0 [014] ; sameclock o a0 c0 a0 b0 c0 [015] ; sameclock i o d0 b0 d0 c0 [016] ; sameclock i o b0 a0 k0 c0 [017] ; sameclock o a0 b0 o c0 k0 [018] ; sameclock o b0 k0 b0 c0 k0 [019] ; </numerical_check>",
                "proof": "<proof> eqratio a c0 i a0 o c0 o a0 [020] a00 [003] [004] ; para a i a0 c0 [021] r27 [001] [002] [020] [008] [009] ; eqangle i o i b0 a0 k0 a0 c0 [022] a01 [000] [023] [021] ; eqangle i o o b0 a0 k0 c0 k0 [024] a01 [023] [025] ; simtri i o b0 a0 k0 c0 [026] r34 [022] [024] [017] ; eqratio i o i b0 a0 k0 a0 c0 [027] r52 [026] ; eqratio a b0 a c0 i b0 o c0 [028] a00 [006] [003] ; para i o b0 c0 [029] r27 [000] [001] [028] [010] [011] ; eqangle i o o d0 b0 c0 b0 d0 [030] a01 [005] [029] ; eqratio a b0 i b0 o a0 i a0 [031] a00 [006] [004] ; para a o a0 b0 [032] r27 [000] [002] [031] [012] [013] ; eqangle a o a d0 a0 b0 a0 d0 [033] a01 [007] [032] ; eqangle a o o d0 a0 b0 b0 d0 [034] a01 [005] [032] ; simtri a o d0 a0 b0 d0 [035] r34 [033] [034] [014] ; eqratio a o o d0 a0 b0 b0 d0 [036] r52 [035] ; eqangle o a0 o k0 a0 b0 o c0 [037] a01 [001] [023] [032] ; eqangle o b0 a0 b0 c0 k0 o c0 [038] a01 [001] [032] [025] ; simtri o a0 b0 k0 o c0 [039] r34 [037] [038] [018] ; eqratio o a0 o k0 a0 b0 o c0 [040] r52 [039] ; eqangle o a0 o c0 b0 c0 a0 b0 [041] a01 [001] [023] [032] [029] ; eqangle o c0 a0 b0 a0 c0 a0 c0 [042] a01 [001] [032] ; simtri o a0 c0 b0 c0 a0 [043] r34 [041] [042] [015] ; eqratio o c0 a0 b0 a0 c0 a0 c0 [044] r52 [043] ; eqangle o b0 o k0 c0 k0 b0 c0 [045] a01 [023] [029] [025] ; eqangle o k0 b0 c0 b0 k0 b0 k0 [046] a01 [023] [029] ; simtri o b0 k0 c0 k0 b0 [047] r34 [045] [046] [019] ; eqratio o k0 b0 c0 b0 k0 b0 k0 [048] r52 [047] ; midp c0 a o [049] r54 [001] [003] ; rconst a o c0 o 2/1 [050] r51 [049] ; midp a0 i o [051] r54 [002] [004] ; rconst i o a0 o 2/1 [052] r51 [051] ; eqratio i o o d0 b0 c0 b0 d0 [053] a00 [036] [040] [044] [048] [050] [052] ; simtri i o d0 c0 b0 d0 [054] r62 [030] [053] [016] ; eqratio i o i d0 b0 c0 c0 d0 [055] r52 [054] ; eqratio i o o b0 a0 k0 c0 k0 [056] r52 [026] ; eqratio o b0 o k0 c0 k0 b0 c0 [057] r52 [047] ; midp b0 a i [058] r54 [000] [006] ; rconst a i b0 i 2/1 [059] r51 [058] ; eqratio a i i d0 a0 c0 c0 d0 [060] a00 [027] [055] [056] [040] [057] [044] [059] [052] ; </proof>"
            },
        }
        g.from_single_result(resA)
        g.from_single_result(resB)
        return g

    def test_path_mining_and_schema(self):
        pg = self.build_small_pg()
        miner = GSpanMiner(pg, min_support=2, min_rule_nodes=1, min_edges=2, sample_embeddings=2)
        patterns = miner.run()
        # 至少应发现 F:coll->R:r54->F:midp 或 F:midp->R:r51->F:rconst 的路径
        self.assertTrue(len(patterns) >= 1)
        # 生成可读 schema
        expr, vmap = miner.pattern_to_schema(patterns[0])
        self.assertIsInstance(expr, str)
        self.assertTrue("=>" in expr)
        self.assertTrue(len(vmap) >= 1)


if __name__ == "__main__":
    unittest.main()
