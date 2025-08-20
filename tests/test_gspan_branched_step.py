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


class GSpanBranchedStepTests(unittest.TestCase):
    def build_pg_small(self) -> ProofGraph:
        # 一个最小可分叉数据：f0->rA->f1；另有 f2->rA->f1（同一 rule 汇合），两题重复
        g = ProofGraph(verbose=False, log_level=logging.WARNING)
        res1 = {
            "problem_id": "S1",
            "proof": {
                "analysis": "<analysis> coll A B C [000] ; coll D E F [001] ; </analysis>",
                "numerical_check": "",
                "proof": "<proof> midp M A B [010] r54 [000] [000] ; midp N D E [011] r54 [001] [001] ; para M N X Y [012] r27 [010] [011] ; cong X Y X Y [013] a00 [012] [012] ; eqangle A B C D E F [014] r62 [000] [001] [001] ; </proof>",
            },
        }
        res2 = {
            "problem_id": "S2",
            "proof": {
                "analysis": "<analysis> coll A B C [000] ; coll D E F [001] ; </analysis>",
                "numerical_check": "",
                "proof": "<proof> midp M A B [010] r54 [000] [000] ; midp N D E [011] r54 [001] [001] ; para M N X Y [012] r27 [010] [011] ; cong X Y X Y [013] a00 [012] [012] ; eqangle A B C D E F [014] r62 [000] [001] [001] ; </proof>",
            },
        }
        g.from_single_result(res1)
        g.from_single_result(res2)
        return g

    def test_stepwise_limit(self):
        pg = self.build_pg_small()
        miner = GSpanMiner(pg, min_support=2, min_rule_nodes=1, min_edges=2, sample_embeddings=2)
        # 开很小的扩展上限，确保函数能正常返回而不挂起
        patterns = miner.run_branched(min_rule_indeg2_count=0, max_edges=8, debug_limit_expansions=1000, debug_log_every=200)
        self.assertIsInstance(patterns, list)
        # 即使受限，也应当能返回一些基础模式
        self.assertTrue(len(patterns) >= 1)


if __name__ == "__main__":
    unittest.main()
