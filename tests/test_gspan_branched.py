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


class GSpanBranchedTests(unittest.TestCase):
    def build_pg_with_branch(self) -> ProofGraph:
        g = ProofGraph(verbose=False, log_level=logging.INFO)
        # 构造两个题目，均包含：f1,f2 -> r1 -> f3；f3,f4 -> r2 -> f5
        res1 = {
            "problem_id": "P1",
            "proof": {
                "analysis": "<analysis> coll a b c [000] ; coll d e f [001] ; coll x y z [002] ; coll u v w [003] ; </analysis>",
                "numerical_check": "",
                "proof": "<proof> midp m a b [010] r54 [000] [000] ; midp n d e [011] r54 [001] [001] ; para m n x y [012] r27 [010] [011] ; para m n u v [013] r27 [010] [011] ; cong x y u v [014] a00 [012] [013] ; cong x y z y [015] r52 [014] ; cong u v w v [016] r52 [014] ; eqangle m n x y u v w v [017] r62 [012] [016] [016] ; </proof>",
            },
        }
        # 简化：我们只需要形成两次 r54 前提，再共同指向一个规则（近似分叉环境），第二个题目重复结构
        res2 = {
            "problem_id": "P2",
            "proof": {
                "analysis": "<analysis> coll a b c [000] ; coll d e f [001] ; coll x y z [002] ; coll u v w [003] ; </analysis>",
                "numerical_check": "",
                "proof": "<proof> midp m a b [010] r54 [000] [000] ; midp n d e [011] r54 [001] [001] ; para m n x y [012] r27 [010] [011] ; para m n u v [013] r27 [010] [011] ; cong x y u v [014] a00 [012] [013] ; cong x y z y [015] r52 [014] ; cong u v w v [016] r52 [014] ; eqangle m n x y u v w v [017] r62 [012] [016] [016] ; </proof>",
            },
        }
        g.from_single_result(res1)
        g.from_single_result(res2)
        return g

    def test_branched_mining(self):
        pg = self.build_pg_with_branch()
        miner = GSpanMiner(pg, min_support=2, min_rule_nodes=1, min_edges=2, sample_embeddings=3)
        patterns = miner.run_branched(min_rule_indeg2_count=0, max_edges=10)
        self.assertTrue(len(patterns) >= 1)
        expr, vmap = miner.pattern_to_schema_branched(patterns[0])
        self.assertIsInstance(expr, str)
        self.assertIn("=>", expr)
        self.assertTrue(len(vmap) >= 0)


if __name__ == "__main__":
    unittest.main()
