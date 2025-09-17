from __future__ import annotations
import os, json, time, threading, queue, re as _re
from datetime import datetime


class SchemaMiner:
    """
    子图挖掘后的轻量收集：仅生成 schema 与渲染信息，写出结果，不包含任何筛选/审计逻辑。
    约定：aconst/rconst 的最后一个参数是数值常量，不参与依赖/闭包，渲染时保留字面量。
    """
    _ATOM_RE = _re.compile(r"\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*")

    def __init__(self, pg, miner, args, out_dir: str):
        self.pg = pg
        self.miner = miner
        self.args = args
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # 仅挖掘：不创建任何审计文件
        self.only_mine = True
        # 收集（不做去重/筛选）
        self.coll = {
            "patterns_all": [],
        }
        # 简化：不启用流式写入
        self.do_stream = False
        self.tmp_ndjson = None
        self.wf = None

    # ---------- 基础解析/工具 ----------
    @classmethod
    def parse_schema(cls, schema: str):
        if not schema or "=>" not in schema:
            return [], None
        left, right = schema.split("=>", 1)
        left = left.strip(); right = right.strip()
        prem_atoms = []
        if left:
            for part in left.split("∧"):
                m = cls._ATOM_RE.match(part.strip())
                if not m: continue
                pred = m.group(1)
                args = [a.strip() for a in m.group(2).split(",") if a.strip()]
                prem_atoms.append((pred, args))
        m = cls._ATOM_RE.match(right)
        concl = None
        if m:
            pred = m.group(1)
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            concl = (pred, args)
        return prem_atoms, concl

    @staticmethod
    def has_unknown(schema: str) -> bool:
        s = schema.strip()
        return "unknown(" in s or s.startswith("unknown()")

    @classmethod
    def concl_args_subset_of_prem(cls, schema: str) -> bool:
        prem, concl = cls.parse_schema(schema)
        if not concl:
            return False
        prem_args = set()
        for pred, args in prem:
            if pred in {"aconst", "rconst"} and len(args) >= 1:
                prem_args.update(args[:-1])
            else:
                prem_args.update(args)
        concl_pred, concl_args = concl
        if concl_pred in {"aconst", "rconst"} and len(concl_args) >= 1:
            concl_args = concl_args[:-1]
        return set(concl_args).issubset(prem_args)

    # ---------- 前提清洗：移除指定谓词（如 sameclock/sameside） ----------
    @classmethod
    def _drop_premises_by_predicates(cls, schema: str, preds: set[str]) -> str:
        """从 schema 的前提部分移除谓词名在 preds 内的原子；结论保持不变。
        - 谓词名大小写不敏感（lower 比较）。
        - 若前提清空，则返回 "=> <conclusion>"。
        - 解析失败或无结论时，原样返回。
        """
        try:
            prem, concl = cls.parse_schema(schema)
            if not concl:
                return schema
            keep = []
            drop_set = {p.lower() for p in preds}
            for pred, args in prem:
                if pred.lower() in drop_set:
                    continue
                keep.append((pred, args))
            concl_pred, concl_args = concl
            left = " ∧ ".join(
                f"{p}(" + ",".join(a) + ")" for (p, a) in keep
            )
            right = f"{concl_pred}(" + ",".join(concl_args) + ")"
            if left:
                return left + " => " + right
            else:
                return "=> " + right
        except Exception:
            return schema

    # ---------- 简化：仅生成条目，不进行筛选/去重 ----------
    def process_one_pattern(self, p: dict):
        # 生成 schema 与渲染信息
        expr, var_map = (
            self.miner.pattern_to_schema_branched(p)
            if self.args.mode == "fact_rule"
            else self.miner.pattern_to_schema_rules_only(p)
        )
        schema_before_dep = expr
        try:
            rendered = self._render_graph_with_schema_vars(self.args.mode, p, dict(var_map or {}))
        except Exception:
            rendered = None
        
        # 组装条目（不做任何筛选/去重）
        item = {
            "support": p.get("support"),
            "pids_count": len(p.get("pids", [])),
            "nodes": len(p.get("nodes", [])),
            "edges": len(p.get("edges", [])),
            "labels": p.get("labels", []),
            "schema": expr,
            "schema_before_dependency": schema_before_dep,
            "rendered": rendered,
        }
        # 附加代表坐标（取首个嵌入的 pid）
        try:
            embs = p.get("embeddings", [])
            if embs:
                pid = str(embs[0].get("pid"))
                coords = getattr(self.pg, "point_coords", {}).get(pid)
                if isinstance(coords, dict):
                    pl = coords.get("point_lines")
                    pts = coords.get("points")
                    if isinstance(pl, list) and isinstance(pts, list):
                        item["point_lines"] = list(pl)
                        item["points"] = list(pts)
                # 透传题目级点依赖（原始点名域）
                rely_map = getattr(self.pg, "point_rely_on", {}).get(pid)
                if isinstance(rely_map, dict):
                    # 转为 JSON 友好的 list 形式
                    item["point_rely_on"] = {str(k): sorted(list(v)) for k, v in rely_map.items()}
        except Exception:
            pass
        self.coll["patterns_all"].append(item)

    # ---------- 收尾：合并流式写入 ----------
    def finalize_stream(self):
        return self.coll["patterns_all"]

    # ---------- 主入口 ----------
    def run(self, problems: int, fact_cnt: int, rule_cnt: int, merged_nodes: int, merged_edges: int, t0: float, t1: float, patterns: list[dict]):
        args = self.args
        # 处理传入的 patterns（单进程）
        self.process_patterns(patterns)
        t2 = time.time()
        return self.finalize_results(problems, fact_cnt, rule_cnt, merged_nodes, merged_edges, t0, t1, t2)

    # ---------- 新增：批量处理与收尾分离 ----------
    def process_patterns(self, patterns_iterable):
        for p in patterns_iterable:
            self.process_one_pattern(p)

    def finalize_results(self, problems: int, fact_cnt: int, rule_cnt: int, merged_nodes: int, merged_edges: int, t0: float, t1: float, t2: float):
        args = self.args
        patterns_all = self.finalize_stream()

        topn = max(0, int(args.top_k))
        patterns_summary = patterns_all[:topn]

        result_obj = {
            "created_at": datetime.now().isoformat(),
            "problems": problems,
            "proof_graph": {
                "nodes_total": len(self.pg.nodes),
                "edges_total": len(self.pg.edges),
                "fact_nodes": fact_cnt,
                "rule_nodes": rule_cnt,
                "stats": self.pg.stats,
            },
            "merged_graph": {
                "nodes_total": merged_nodes,
                "edges_total": merged_edges,
            },
            "params": {
                "mode": args.mode,
                "min_support": args.min_support,
                "min_rule_nodes": args.min_rule_nodes,
                "min_edges": args.min_edges,
                "min_rule_indeg2_count": args.min_rule_indeg2_count,
                "max_nodes": args.max_nodes,
                "rule_only_max_nodes_ratio": getattr(args, "rule_only_max_nodes_ratio", None),
                "sample_embeddings": args.sample_embeddings,
                "debug_limit_expansions": args.debug_limit_expansions,
                "debug_log_every": args.debug_log_every,
                "time_budget_seconds": args.time_budget_seconds,
                "prune_low_support_labels": bool(args.prune_low_support_labels),
                "prune_by_rule": bool(args.prune_by_rule),
                "attach_producer": bool(args.attach_producer),
                "max_producer_depth": int(args.max_producer_depth),
                "skip_unknown": bool(args.skip_unknown),
                "enable_var_closure_check": bool(getattr(args, "enable_var_closure_check", False)),
                "stream_write": int(getattr(args, "stream_write", 0)),
                "drop_unknown_in_schema": bool(getattr(args, "drop_unknown_in_schema", False)),
                "enable_dependency_filter": int(getattr(args, "enable_dependency_filter", 0)),
                "strip_clock_side_in_schema": bool(getattr(args, "strip_clock_side_in_schema", 1)),
            },
            "timing_sec": {
                "parse_build": t1 - t0,
                "mining": t2 - t1,
                "total": t2 - t0,
            },
            "pattern_count": len(patterns_all),
            "patterns_summary_topN": patterns_summary,
        }
        result_obj["patterns"] = patterns_all

        out_name = ("branched_mining.json" if args.mode=="fact_rule" else "rules_only_mining.json")
        out_path = os.path.join(self.out_dir, out_name)
        # 自定义 pretty writer：labels / rules / fr_edges 的数组保持单行，其余使用缩进多行
        def _dump_pretty_mixed(o, fp, level=0, indent=2, inline_keys={"labels", "rules", "fr_edges"}, parent_key: str | None=None):
            pad = " " * (indent * level)
            if isinstance(o, dict):
                items = list(o.items())
                fp.write("{\n")
                for idx, (k, v) in enumerate(items):
                    fp.write(" " * (indent * (level + 1)))
                    fp.write(json.dumps(k, ensure_ascii=False))
                    fp.write(": ")
                    if isinstance(v, list) and k in inline_keys:
                        fp.write(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
                    else:
                        _dump_pretty_mixed(v, fp, level + 1, indent, inline_keys, k)
                    if idx != len(items) - 1:
                        fp.write(",\n")
                    else:
                        fp.write("\n")
                fp.write(pad + "}")
            elif isinstance(o, list):
                if not o:
                    fp.write("[]")
                    return
                fp.write("[\n")
                for i, it in enumerate(o):
                    fp.write(" " * (indent * (level + 1)))
                    _dump_pretty_mixed(it, fp, level + 1, indent, inline_keys, parent_key)
                    if i != len(o) - 1:
                        fp.write(",\n")
                    else:
                        fp.write("\n")
                fp.write(pad + "]")
            else:
                fp.write(json.dumps(o, ensure_ascii=False))

        with open(out_path, "w", encoding="utf-8") as f:
            _dump_pretty_mixed(result_obj, f, level=0, indent=2)
        print(f"results saved to: {out_path}")
        # 无审计文件，无需关闭
        return result_obj

    # ---------- 子图渲染（基于 schema 变量映射，仅首个嵌入） ----------
    def _render_graph_with_schema_vars(self, mode: str, patt: dict, var_map: dict | None):
        if not patt or not patt.get("embeddings"):
            return None
        embs = patt.get("embeddings") or []
        emb = embs[0]
        mapping = emb.get("mapping", {}) or {}
        pid = emb.get("pid")
        labels = patt.get("labels", []) or []
        edges = patt.get("edges", []) or []
        vmap = var_map or {}

        def vget(x: str) -> str:
            if x not in vmap:
                vmap[x] = f"X{len(vmap)+1}"
            return vmap[x]

        def render_fact_gid(fid: int) -> str:
            try:
                orig = self.pg.nodes[self.miner.G.nodes[fid]["orig_node_id"]]
                pred = str(orig.get("label", ""))
                args = list(orig.get("args", []))
                if pred in {"aconst", "rconst"} and len(args) >= 1:
                    head = ",".join(vget(a) for a in args[:-1])
                    tail = str(args[-1])
                    return f"{pred}(" + (head + "," if head else "") + tail + ")"
                return f"{pred}(" + ",".join(vget(a) for a in args) + ")"
            except Exception:
                return ""

        # 依据题目级依赖，构造变量域下的 rely_on
        def build_rely_on_vars() -> dict | None:
            try:
                rely_src = getattr(self.pg, "point_rely_on", {}).get(str(pid))
                if not isinstance(rely_src, dict):
                    return None
                out = {}
                for k, deps in rely_src.items():
                    vk = vget(str(k))
                    if isinstance(deps, (set, list, tuple)):
                        out[vk] = sorted({vget(str(d)) for d in deps})
                    else:
                        out[vk] = []
                return out
            except Exception:
                return None

        rely_on_vars = build_rely_on_vars()

        if mode == "fact_rule":
            nodes = []
            for i, lb in enumerate(labels):
                gid = mapping.get(i)
                if gid is None:
                    continue
                nd = self.miner.G.nodes.get(gid, {})
                ntype = nd.get("type")
                if ntype == "fact":
                    label_str = render_fact_gid(gid)
                else:
                    label_str = lb  # e.g., R:code
                nodes.append({"idx": i, "type": ntype, "label": label_str, "gid": gid})
            out = {"nodes": nodes, "edges": list(edges), "schema_vars": dict(vmap)}
            if rely_on_vars:
                out["rely_on"] = rely_on_vars
            return out

        # rule_only：回映射事实并标注角色
        # 规则节点集合
        rule_gids = [mapping.get(i) for i in range(len(labels))]
        if any(g is None for g in rule_gids):
            return None
        Rset = set(rule_gids)

        produced, consumed, premises_ext = set(), set(), set()
        # 规则输出的事实（produced）与被内部规则再次消耗的事实（consumed）
        for rg in Rset:
            # in-edges: premises to rule
            for fu in self.miner.G.in_edges.get(rg, []):
                nd_f = self.miner.G.nodes[fu]
                if nd_f.get("type") != "fact" or nd_f.get("problem_id") != pid:
                    continue
                premises_ext.add(fu)
            # out-edges: rule to fact
            for fv in self.miner.G.out_edges.get(rg, []):
                nd_fv = self.miner.G.nodes[fv]
                if nd_fv.get("type") != "fact" or nd_fv.get("problem_id") != pid:
                    continue
                produced.add(fv)
                # fact to rule inside Rset => consumed
                for r2 in self.miner.G.out_edges.get(fv, []):
                    if r2 in Rset and self.miner.G.nodes[r2]["problem_id"] == pid:
                        consumed.add(fv)

        premises_ext -= produced
        conclusion = produced - consumed
        concl_gid = next(iter(conclusion)) if len(conclusion) == 1 else None

        facts: list[dict] = []
        # 外部前提
        for fid in sorted(list(premises_ext)):
            facts.append({"gid": fid, "label": render_fact_gid(fid), "role": "premise"})
        # 内部产生
        for fid in sorted(list(produced)):
            role = "conclusion" if concl_gid is not None and fid == concl_gid else "internal"
            facts.append({"gid": fid, "label": render_fact_gid(fid), "role": role})

        # 收集 rule<->fact 边（限制在同一题目 & Rset 内）
        fr_edges = []  # fact -> rule
        rf_edges = []  # rule -> fact
        for rg in Rset:
            for fu in self.miner.G.in_edges.get(rg, []):
                if fu in premises_ext and self.miner.G.nodes[fu]["problem_id"] == pid:
                    fr_edges.append((fu, rg))
            for fv in self.miner.G.out_edges.get(rg, []):
                if fv in produced and self.miner.G.nodes[fv]["problem_id"] == pid:
                    rf_edges.append((rg, fv))

        out = {
            "rules": list(labels),
            "facts": facts,
            "fr_edges": fr_edges,
            "rf_edges": rf_edges,
            "schema_vars": dict(vmap),
        }
        if rely_on_vars:
            out["rely_on"] = rely_on_vars
        return out
