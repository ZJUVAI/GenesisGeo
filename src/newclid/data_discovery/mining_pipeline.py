from __future__ import annotations
import os, json, time, threading, queue, re as _re
from datetime import datetime


class MiningPipeline:
    """
    封装挖掘后处理：schema 生成 → 过滤（unknown/依赖/变量闭包）→ 两层去重 → 最终“同结论按前提集合最小化” → 写结果与五步审计。
    约定：aconst/rconst 的最后一个参数是数值常量，不参与依赖/闭包，渲染时保留字面量。
    """
    _ATOM_RE = _re.compile(r"\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*")

    def __init__(self, pg, miner, args, out_dir: str):
        self.pg = pg
        self.miner = miner
        self.args = args
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # 审计文件
        self.audit_step1 = os.path.join(out_dir, f"audit_{args.mode}_step1_after_unknown.ndjson")
        self.audit_step2_kept = os.path.join(out_dir, f"audit_{args.mode}_step2_dep_kept.ndjson")
        self.audit_step2_dropped = os.path.join(out_dir, f"audit_{args.mode}_step2_dep_dropped.ndjson")
        self.audit_step3 = os.path.join(out_dir, f"audit_{args.mode}_step3_after_varclosure.ndjson")
        self.audit_step4 = os.path.join(out_dir, f"audit_{args.mode}_step4_final_dedup.ndjson")
        self.audit_step5_dropped = os.path.join(out_dir, f"audit_{args.mode}_step5_subset_min_dropped.ndjson")

        self.af1 = open(self.audit_step1, "w", encoding="utf-8")
        self.af2k = open(self.audit_step2_kept, "w", encoding="utf-8")
        self.af2d = open(self.audit_step2_dropped, "w", encoding="utf-8")
        self.af3 = open(self.audit_step3, "w", encoding="utf-8")
        self.af4 = open(self.audit_step4, "w", encoding="utf-8")
        self.af5d = open(self.audit_step5_dropped, "w", encoding="utf-8")

        # 收集/去重
        self.coll = {
            "sig_seen": set(),
            "schema_seen": set(),
            "patterns_all": [],
            "dropped_unknown": 0,
            "dropped_dep": 0,
            "dropped_var_subset": 0,
            "dedup_sig": 0,
            "dedup_schema": 0,
            "written": 0,
        }
        # 流式写入
        self.do_stream = bool(args.stream_write)
        self.tmp_ndjson = os.path.join(out_dir, f"tmp_{args.mode}_patterns.ndjson")
        self.wf = open(self.tmp_ndjson, "w", encoding="utf-8") if self.do_stream else None

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

    def _args_points_from_fact_gid(self, fid: int) -> set:
        try:
            node = self.miner.G.nodes[fid]
            orig = self.pg.nodes[node["orig_node_id"]]
            pred = str(orig.get("label", ""))
            args = list(orig.get("args", []))
            if pred in {"aconst", "rconst"} and len(args) >= 1:
                args = args[:-1]
            return set(str(a).strip() for a in args if str(a).strip())
        except Exception:
            return set()

    # ---------- 依赖过滤（首嵌入，三分支） ----------
    def _dep_apply_on_first_embedding(self, mode: str, patt: dict, schema_before: str):
        embs = patt.get("embeddings", [])
        if not embs:
            return "drop", {"reason": "no_embedding"}
        emb = embs[0]
        pid = emb.get("pid")
        pid_key = str(pid)
        mapping: dict = emb.get("mapping", {})
        rely_map = self.pg.point_rely_on.get(pid_key, {})

        labels = patt.get("labels", [])
        edges = patt.get("edges", [])
        premise_fact_gids: list[int] = []
        conclusion_gid: int | None = None
        if mode == "fact_rule":
            n = len(labels)
            indeg = [0]*n; outdeg=[0]*n
            for a,b in edges:
                outdeg[a]+=1; indeg[b]+=1
            prem_idxs = [i for i,lb in enumerate(labels) if lb.startswith("F:") and indeg[i]==0]
            sink = [i for i,lb in enumerate(labels) if lb.startswith("F:") and outdeg[i]==0]
            if len(sink) != 1:
                return "drop", {"reason": "no_unique_conclusion"}
            for i in prem_idxs:
                gid = mapping.get(i)
                if gid is not None:
                    premise_fact_gids.append(gid)
            cg = mapping.get(sink[0])
            if cg is None:
                return "drop", {"reason": "conclusion_unmapped"}
            conclusion_gid = cg
        else:
            rule_gids = [mapping.get(i) for i in range(len(labels))]
            if any(gid is None for gid in rule_gids):
                return "drop", {"reason": "rule_unmapped"}
            Rset = set(rule_gids)
            produced, consumed, premises_ext = set(), set(), set()
            for rg in Rset:
                for fu in self.miner.G.in_edges.get(rg, []):
                    if self.miner.G.nodes[fu]["type"] != "fact" or self.miner.G.nodes[fu]["problem_id"] != pid:
                        continue
                    premises_ext.add(fu)
                for fv in self.miner.G.out_edges.get(rg, []):
                    if self.miner.G.nodes[fv]["type"] != "fact" or self.miner.G.nodes[fv]["problem_id"] != pid:
                        continue
                    produced.add(fv)
                    for r2 in self.miner.G.out_edges.get(fv, []):
                        if r2 in Rset and self.miner.G.nodes[r2]["problem_id"] == pid:
                            consumed.add(fv)
            premises_ext -= produced
            conclusion = produced - consumed
            if len(conclusion) != 1:
                return "drop", {"reason": "no_unique_conclusion"}
            conclusion_gid = next(iter(conclusion))
            premise_fact_gids = sorted(list(premises_ext))

        if not premise_fact_gids:
            return "drop", {"reason": "no_premises"}

        concl_points = self._args_points_from_fact_gid(conclusion_gid)
        union_rely: set = set(concl_points)
        for pt in concl_points:
            union_rely |= set(rely_map.get(str(pt), set()))

        premises_detail = []
        kept_gids, removed_gids = [], []
        all_subset, all_not_subset = True, True
        for gid in premise_fact_gids:
            pts = self._args_points_from_fact_gid(gid)
            subset_ok = pts.issubset(union_rely)
            if subset_ok:
                kept_gids.append(gid); all_not_subset = False
            else:
                removed_gids.append(gid); all_subset = False
            premises_detail.append({
                "fact_gid": gid,
                "points": sorted(list(pts)),
                "subset_ok": bool(subset_ok),
            })

        if all_subset:
            return "drop", {
                "reason": "all_subset",
                "union_rely": sorted(list(union_rely)),
                "conclusion_points": sorted(list(concl_points)),
                "premises": premises_detail,
                "schema_before": schema_before,
            }
        if all_not_subset:
            return "drop", {
                "reason": "all_not_subset",
                "union_rely": sorted(list(union_rely)),
                "conclusion_points": sorted(list(concl_points)),
                "premises": premises_detail,
                "schema_before": schema_before,
            }

        var_map: dict[str, str] = {}
        def vget(x: str) -> str:
            if x not in var_map:
                var_map[x] = f"X{len(var_map)+1}"
            return var_map[x]
        def fact_term_from_gid(fid: int) -> str:
            orig = self.pg.nodes[self.miner.G.nodes[fid]["orig_node_id"]]
            pred = orig["label"]
            args = orig.get("args", [])
            if pred in {"aconst","rconst"} and len(args)>=1:
                head = ",".join(vget(a) for a in args[:-1])
                tail = str(args[-1])
                return f"{pred}(" + (head + "," if head else "") + tail + ")"
            return f"{pred}(" + ",".join(vget(a) for a in args) + ")"

        premise_terms_kept = [fact_term_from_gid(g) for g in kept_gids]
        concl_term = fact_term_from_gid(conclusion_gid)
        schema_after = (" ∧ ".join(premise_terms_kept) + (" => " if premise_terms_kept else "=> ") + concl_term)

        return "keep", {
            "union_rely": sorted(list(union_rely)),
            "conclusion_points": sorted(list(concl_points)),
            "premises": premises_detail,
            "kept_count": len(kept_gids),
            "removed_count": len(removed_gids),
            "schema_before": schema_before,
            "schema_after": schema_after,
        }

    # ---------- 规范化键（最终子集最小化） ----------
    @classmethod
    def _norm_for_subset(cls, schema: str):
        prem, concl = cls.parse_schema(schema)
        if not concl:
            return None
        var_map: dict[str, str] = {}
        def vget(x: str) -> str:
            if x not in var_map:
                var_map[x] = f"X{len(var_map)+1}"
            return var_map[x]
        def norm_args(pred: str, args: list[str]) -> list[str]:
            n = len(args)
            cut = n-1 if (pred in {"aconst","rconst"} and n >= 1) else n
            out = []
            for i,a in enumerate(args):
                out.append(vget(a) if i < cut else str(a))
            return out
        concl_pred, concl_args = concl
        concl_norm = f"{concl_pred}(" + ",".join(norm_args(concl_pred, concl_args)) + ")"
        prem_keys = []
        for pred, args in sorted(prem, key=lambda x: (x[0], len(x[1]), tuple(x[1]))):
            na = norm_args(pred, args)
            prem_keys.append(f"{pred}(" + ",".join(na) + ")")
        return concl_norm, frozenset(prem_keys), len(prem_keys)

    # ---------- 单条模式处理 ----------
    def process_one_pattern(self, p: dict):
        sig = (tuple(p.get("labels", [])), tuple(sorted(p.get("edges", []))))
        if sig in self.coll["sig_seen"]:
            self.coll["dedup_sig"] += 1
            return
        self.coll["sig_seen"].add(sig)

        expr, var_map = (self.miner.pattern_to_schema_branched(p) if self.args.mode == "fact_rule" else self.miner.pattern_to_schema_rules_only(p))
        schema_before_dep = expr
        # 渲染：用 schema 变量映射生成子图可视化（仅首个嵌入）
        try:
            rendered = self._render_graph_with_schema_vars(self.args.mode, p, dict(var_map or {}))
        except Exception:
            rendered = None

        if self.args.drop_unknown_in_schema and self.has_unknown(expr):
            self.coll["dropped_unknown"] += 1
            return
        try:
            self.af1.write(json.dumps({"schema": expr, "support": p.get("support"), "pids": list(p.get("pids", []))}, ensure_ascii=False) + "\n")
        except Exception:
            pass

        if int(self.args.enable_dependency_filter) == 1:
            decision, info = self._dep_apply_on_first_embedding(self.args.mode, p, expr)
            if decision == "drop":
                self.coll["dropped_dep"] += 1
                try:
                    self.af2d.write(json.dumps({
                        "schema_before": info.get("schema_before", expr),
                        "support": p.get("support"),
                        "pids": list(p.get("pids", [])),
                        "reason": info.get("reason"),
                        "union_rely": info.get("union_rely"),
                        "conclusion_points": info.get("conclusion_points"),
                        "premises": info.get("premises"),
                    }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                return
            elif decision == "keep":
                expr = info.get("schema_after", expr)
                schema_before_dep = info.get("schema_before", schema_before_dep)
                try:
                    rec = {
                        "schema_before": info.get("schema_before", expr),
                        "schema_after": expr,
                        "support": p.get("support"),
                        "pids": list(p.get("pids", [])),
                        "union_rely": info.get("union_rely"),
                        "conclusion_points": info.get("conclusion_points"),
                        "premises": info.get("premises"),
                        "kept_count": info.get("kept_count"),
                        "removed_count": info.get("removed_count"),
                    }
                    self.af2k.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        if bool(self.args.enable_var_closure_check) and not self.concl_args_subset_of_prem(expr):
            self.coll["dropped_var_subset"] += 1
            return
        try:
            self.af3.write(json.dumps({"schema": expr, "support": p.get("support"), "pids": list(p.get("pids", []))}, ensure_ascii=False) + "\n")
        except Exception:
            pass

        key = self.canonical_key(expr) if expr else None
        if key and key in self.coll["schema_seen"]:
            self.coll["dedup_schema"] += 1
            return
        if key:
            self.coll["schema_seen"].add(key)
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
        try:
            self.af4.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass
        if self.do_stream and self.wf is not None:
            self.wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            self.coll["written"] += 1
        else:
            self.coll["patterns_all"].append(item)

    @classmethod
    def canonical_key(cls, schema: str) -> str:
        prem, concl = cls.parse_schema(schema)
        order = []
        def vnorm(arg: str) -> str:
            if arg not in order:
                order.append(arg)
            return f"X{order.index(arg)+1}"
        prem_norm = []
        for pred, args in prem:
            prem_norm.append((pred, tuple(vnorm(a) for a in args), len(args)))
        prem_norm.sort(key=lambda x: (x[0], x[2], x[1]))
        concl_norm = None
        if concl:
            concl_norm = (concl[0], tuple(vnorm(a) for a in concl[1]), len(concl[1]))
        return json.dumps({
            "prem": [(p, list(a), k) for (p,a,k) in prem_norm],
            "concl": None if concl_norm is None else [concl_norm[0], list(concl_norm[1]), concl_norm[2]],
        }, ensure_ascii=False, sort_keys=True)

    # ---------- 收尾：合并流式写入 ----------
    def finalize_stream(self):
        patterns_all = self.coll["patterns_all"]
        if self.do_stream and self.wf is not None:
            self.wf.close()
            with open(self.tmp_ndjson, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    patterns_all.append(obj)
            try: os.remove(self.tmp_ndjson)
            except Exception: pass
        return patterns_all

    # ---------- 最终“最小包含”去重 + 审计 ----------
    def minimize_by_premise_subset(self, patterns_all: list[dict]):
        groups = {}
        for it in patterns_all:
            schema = it.get("schema") or ""
            norm = self._norm_for_subset(schema)
            if not norm: continue
            concl_key, prem_set, _ = norm
            groups.setdefault(concl_key, []).append((prem_set, it))

        minimized, dropped = [], []
        for concl_key, lst in groups.items():
            keep = [True]*len(lst)
            for i, (Si, iti) in enumerate(lst):
                if not keep[i]:
                    continue
                for j, (Sj, itj) in enumerate(lst):
                    if i == j: continue
                    if Sj.issubset(Si) and len(Sj) < len(Si):
                        keep[i] = False
                        dropped.append({
                            "conclusion": concl_key,
                            "schema": iti.get("schema"),
                            "schema_before_dependency": iti.get("schema_before_dependency"),
                            "premises": sorted(list(Si)),
                            "premise_count": len(Si),
                            "dropped_reason": "premise_superset_of_another",
                            "winner_premises": sorted(list(Sj)),
                            "winner_premise_count": len(Sj),
                            "winner_schema": itj.get("schema"),
                        })
                        break
            for k, flag in enumerate(keep):
                if flag:
                    minimized.append(lst[k][1])

        try:
            for rec in dropped:
                self.af5d.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return minimized

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
        patterns_all = self.minimize_by_premise_subset(patterns_all)

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
                "enable_var_closure_check": bool(args.enable_var_closure_check),
                "stream_write": int(args.stream_write),
                "drop_unknown_in_schema": bool(args.drop_unknown_in_schema),
                "enable_dependency_filter": int(args.enable_dependency_filter),
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

        try:
            self.af1.close(); self.af2k.close(); self.af2d.close(); self.af3.close(); self.af4.close(); self.af5d.close()
        except Exception:
            pass
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
            return {"nodes": nodes, "edges": list(edges), "schema_vars": dict(vmap)}

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

        return {
            "rules": list(labels),
            "facts": facts,
            "fr_edges": fr_edges,
            "rf_edges": rf_edges,
            "schema_vars": dict(vmap),
        }
