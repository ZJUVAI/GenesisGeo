import os
import sys
import logging
import unittest

# 将 src 加入 sys.path，便于在本地直接运行测试
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from newclid.data_discovery.proof_graph import ProofGraph  # noqa: E402

class ProofGraphTests(unittest.TestCase):
    def test_parse_facts_from_text_basic(self):
        g = ProofGraph(verbose=False, log_level=logging.DEBUG)
        text = "<analysis> cong a d a e [000] ; coll a d e [002] ; r54 ; </analysis>"
        g.parse_facts_from_text("0001", text)

        # 两个 fact + 1 个无法解析的片段（r54）
        self.assertEqual(len(g.nodes), 2)
        self.assertGreaterEqual(g.stats["unparsed_fact_segments"], 1)

        n0 = g.nodes[g.fact_id_map["0001"]["000"]]
        self.assertEqual(n0["type"], "fact")
        self.assertEqual(n0["label"], "cong")
        self.assertEqual(n0["args"], ["a", "d", "a", "e"])
        self.assertEqual(n0["layer"], 1)
        n1 = g.nodes[g.fact_id_map["0001"]["002"]]
        self.assertEqual(n1["type"], "fact")
        self.assertEqual(n1["label"], "coll")
        self.assertEqual(n1["args"], ["a", "d", "e"])
        self.assertEqual(n1["layer"], 1)

    def test_parse_proof_step_basic(self):
        g = ProofGraph(verbose=False, log_level=logging.DEBUG)
        line = "eqratio a d a e d r f r [004] a00 [000] [001]"
        parsed = g.parse_proof_step(line)
        self.assertIsNotNone(parsed)
        pred, args, cid, rule, pres = parsed  # type: ignore
        self.assertEqual(pred, "eqratio")
        self.assertEqual(args, ["a", "d", "a", "e", "d", "r", "f", "r"])
        self.assertEqual(cid, "004")
        self.assertEqual(rule, "a00")
        self.assertEqual(pres, ["000", "001"])

    def test_add_rule_step_with_placeholders(self):
        g = ProofGraph(verbose=False, log_level=logging.DEBUG)
        # 前提缺失，需创建占位
        rid, fid = g.add_rule_step(
            problem_id="p1",
            step_index=1,
            concl_pred="midp",
            concl_args=["a", "d", "e"],
            concl_id="005",
            rule_code="r54",
            premise_ids=["002", "000"],
        )
        self.assertIn(rid, g.nodes)
        self.assertIn(fid, g.nodes)
        # 两个占位 + 规则 + 结论 = 4 个节点
        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(g.stats["placeholders"], 2)

        rule = g.nodes[rid]
        concl = g.nodes[fid]
        # 占位层=1 => rule.layer=2 => concl.layer=3
        self.assertEqual(rule["layer"], 2)
        self.assertEqual(concl["layer"], 3)
        # 边：2 条前提->规则 + 规则->结论 = 3 条
        self.assertEqual(len(g.edges), 3)

    def test_from_single_result_flow(self):
        g = ProofGraph(verbose=False, log_level=logging.DEBUG)
        result = {
            "problem_id": "0001",
            "proof": {
                "analysis": "<analysis> cong a d a e [000] ; cong d r f r [001] ; </analysis>",
                "numerical_check": "",
                "proof": "<proof> eqratio a d a e d r f r [004] a00 [000] [001] ; </proof>",
            },
        }
        g.from_single_result(result)

        # 4 个节点：2 个初始 fact + 1 规则 + 1 结论 fact
        self.assertEqual(len(g.nodes), 4)
        # 3 条边：2 条前提->规则 + 规则->结论
        self.assertEqual(len(g.edges), 3)

        # 层级检查：初始=1，规则=2，结论=3
        f0 = g.nodes[g.fact_id_map["0001"]["000"]]
        f1 = g.nodes[g.fact_id_map["0001"]["001"]]
        rule_id = g.rule_step_map["0001"][1]
        r = g.nodes[rule_id]
        concl = g.nodes[g.fact_id_map["0001"]["004"]]

        self.assertEqual(f0["layer"], 1)
        self.assertEqual(f1["layer"], 1)
        self.assertEqual(r["layer"], 2)
        self.assertEqual(concl["layer"], 3)

    def test_from_results_obj_multi(self):
        g = ProofGraph(verbose=False, log_level=logging.DEBUG)
        obj = {
            "results": [
                {
                    "problem_id": "p1",
                    "proof": {
                        "analysis": "<analysis> coll a d e [002] ; cong a d a e [000] ; </analysis>",
                        "numerical_check": "",
                        "proof": "<proof> midp a d e [005] r54 [002] [000] ; </proof>",
                    },
                },
                {
                    "problem_id": "p2",
                    "proof": {
                        "analysis": "<analysis> coll d f r [003] ; cong d r f r [001] ; </analysis>",
                        "numerical_check": "",
                        "proof": "<proof> midp r d f [007] r54 [003] [001] ; </proof>",
                    },
                },
            ]
        }

        g.from_results_obj(obj)
        # 两题，各 4 节点（2 初始 + 1 规则 + 1 结论）
        self.assertEqual(len(g.nodes), 8)
        self.assertEqual(len(g.edges), 6)


if __name__ == "__main__":
    unittest.main()
