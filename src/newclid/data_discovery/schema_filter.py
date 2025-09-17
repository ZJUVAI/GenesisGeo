from __future__ import annotations

"""
Schema 筛选工具（从 MiningPipeline 中抽离）。

功能总览：
- SchemaFilter：可复用的筛选器类，按顺序执行以下可控步骤：
    1) 丢弃 unknown() 谓词相关的 schema（可关闭/自动禁用）
    2) 依赖过滤（可选，需要 ctx 中提供 pg/miner）
    3) 变量闭包检查（结论变量应包含于前提变量集合）
    4) 前提清洗：移除 sameclock/sameside 时钟/边侧约束（strip_clock_side_in_schema）
    5) 规范化去重（schema 级）
    6) 同结论下按前提集合执行 subset-min（可选）

主要方法说明：
- parse_schema(schema: str) -> (premises, conclusion)：解析 "A(x) ∧ B(y) => C(z)" 形式的 DSL 为原子列表与结论元组。
- has_unknown(schema: str) -> bool：检测 schema 中是否出现 unknown()。
- concl_args_subset_of_prem(schema: str) -> bool：检测结论变量是否为前提变量的子集（变量闭包）。
- _drop_premises_by_predicates(schema: str, preds: set[str]) -> str：移除指定谓词的前提（如 sameclock/sameside）。
- _args_points_from_fact_gid(fid: int) -> set：从 fact 节点 gid 提取参与推理的点集合（不含常量尾参）。
- _dep_apply_on_first_embedding(mode, patt, schema_before) -> (decision, info)：
    基于第一条 embedding 做依赖过滤；decision ∈ {"keep", "drop", "skip"}。
- _norm_for_subset(schema: str)：为 subset-min 生成标准化键（结论键、前提集合键）。
- canonical_key(schema: str) -> str：生成 schema 的规范化键（用于去重）。
- process_patterns(patterns_iterable)：批量处理入口。
- process_one_pattern(p)：处理单个 pattern，按开关依次应用各步骤并产出记录。
- finalize_stream()：若启用流式临时写入，将其收拢为内存列表。
- minimize_by_premise_subset(patterns_all)：同结论下按前提集合做子集最小化。
- finalize_results(meta)：生成统一最终 JSON（可选摘要），并安全关闭审计句柄。

输出：
- 默认仅输出一个统一的最终 JSON；如开启审计，将写出与旧 MiningPipeline 兼容的 NDJSON 文件。

依赖：
- 仅标准库。
"""

import os, json, re as _re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


class SchemaFilter:
    _ATOM_RE = _re.compile(r"\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*")

    def __init__(self, args, out_dir: str, ctx: Optional[dict] = None):
        """
        参数：
        - args：包含多个可控开关/参数的命名对象，字段包括但不限于：
          - mode: str，"fact_rule" | "rule_only"，影响结论定位与输出文件名
          - drop_unknown_in_schema: bool，是否丢弃出现 unknown() 的 schema
          - enable_dependency_filter: int (0/1)，是否启用依赖过滤（需要 ctx）
          - enable_var_closure_check: bool，是否启用变量闭包检查
          - strip_clock_side_in_schema: bool，是否移除 sameclock/sameside 前提
          - enable_schema_dedup: bool，是否启用规范化 schema 去重
          - enable_subset_min: bool，是否启用同结论下的前提集合最小化
          - enable_audit_ndjson: bool，是否输出审计 NDJSON
          - top_k: int，最终 JSON 中 patterns_summary_topN 的大小
          - stream_write: int (0/1)，是否流式写入临时 NDJSON
          - auto_disabled_unknown_filter: bool，是否因预扫而自动禁用 unknown 过滤
        - out_dir：输出目录（最终 JSON 与可选审计文件会写入此处）
        - ctx：依赖过滤所需的上下文（包含 'pg' 与 'miner'）。
        """
        self.args = args
        self.ctx = ctx or {}
        self.pg = self.ctx.get("pg")
        self.miner = self.ctx.get("miner")
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # filter flags (with safe defaults)
        self.enable_audit_ndjson = bool(getattr(args, "enable_audit_ndjson", False))
        self.enable_schema_dedup = bool(getattr(args, "enable_schema_dedup", True))
        self.enable_subset_min = bool(getattr(args, "enable_subset_min", False))
        # unknown filter effective flag + auto-disabled marker (to be set by caller if pre-scanned)
        self.drop_unknown_effective = bool(getattr(args, "drop_unknown_in_schema", False))
        self.auto_disabled_unknown_filter = bool(getattr(args, "auto_disabled_unknown_filter", False))

        # 审计文件（可选，默认不生成）
        if self.enable_audit_ndjson:
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
        else:
            self.audit_step1 = self.audit_step2_kept = self.audit_step2_dropped = None
            self.audit_step3 = self.audit_step4 = self.audit_step5_dropped = None
            self.af1 = self.af2k = self.af2d = self.af3 = self.af4 = self.af5d = None

        # 运行期收集器（计数器与缓存集合）
        self.coll = {
            "schema_seen": set(),
            "patterns_all": [],
            "dropped_unknown": 0,
            "dropped_dep": 0,
            "dropped_var_subset": 0,
            "dedup_schema": 0,
            "written": 0,
        }
        # 是否启用临时 NDJSON 的流式写入
        self.do_stream = bool(getattr(self.args, "stream_write", 0))
        self.tmp_ndjson = os.path.join(self.out_dir, f"tmp_{self.args.mode}_patterns.ndjson")
        self.wf = open(self.tmp_ndjson, "w", encoding="utf-8") if self.do_stream else None

    # ---------- parsing utils ----------
    @classmethod
    def parse_schema(cls, schema: str):
    # 解析 DSL 形式的 schema 为（前提原子列表, 结论原子）二元组。
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
    # 检测 schema 中是否包含 unknown() 谓词。
        s = schema.strip()
        return "unknown(" in s or s.startswith("unknown()")

    @classmethod
    def concl_args_subset_of_prem(cls, schema: str) -> bool:
    # 检查结论变量是否为前提变量集合的子集（变量闭包）。
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

    @classmethod
    def _drop_premises_by_predicates(cls, schema: str, preds: set[str]) -> str:
    # 从前提中移除谓词位于 preds 的原子，返回清洗后的 schema（失败则原样返回）。
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

    # ---------- dependency filter helpers ----------
    def _args_points_from_fact_gid(self, fid: int) -> set:
    # 从 fact 节点 gid 中提取参与推理的点（忽略 aconst/rconst 的常量尾参）。
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

    def _dep_apply_on_first_embedding(self, mode: str, patt: dict, schema_before: str):
    # 依赖过滤：基于第一条 embedding 的上下文判定保留/丢弃/跳过。
    # 返回：
    # - ("keep", info)：保留，并在 info.schema_after 中给出可能精简后的 schema
    # - ("drop", info)：丢弃，info.reason 提供原因
    # - ("skip", info)：无法评估（无上下文/无 embedding），不上过滤
        if not (self.pg and self.miner):
            return "skip", {"reason": "no_ctx"}
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

    # ---------- normalization for subset-min ----------
    @classmethod
    def _norm_for_subset(cls, schema: str):
    # 为 subset-min 生成标准化键（结论键、前提集合键、前提个数）。
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

    @classmethod
    def canonical_key(cls, schema: str) -> str:
    # 生成 schema 的规范化 JSON 键，用于去重（变量次序与前提顺序归一化）。
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

    # ---------- processing ----------
    def process_patterns(self, patterns_iterable: Iterable[dict]):
    # 批量处理入口：逐个调用 process_one_pattern。
        for p in patterns_iterable:
            self.process_one_pattern(p)

    def process_one_pattern(self, p: dict):
    # 单条 pattern 的处理流程：
    # 1) 结构签名去重（可控）
    # 2) 取得 schema（及依赖前的 schema）与渲染信息
    # 3) unknown 过滤（可控）
    # 4) 依赖过滤（可控）
    # 5) 变量闭包检查（可控）
    # 6) 清洗 sameclock/sameside（可控）
    # 7) 规范化 schema 去重（可控）
    # 8) 组装输出项（坐标透传/补全）
    # 9) 写入审计/汇总
    # 结构签名去重已移除：统一依赖规范化 schema 去重

        # get schema before dependency phase (if provided) else raw schema
        expr = p.get("schema") or ""
        schema_before_dep = p.get("schema_before_dependency") or expr

        # render info passthrough (if provided), also used to fetch schema_vars
        rendered = p.get("rendered")

        # drop unknown if needed
        if bool(self.drop_unknown_effective) and expr and self.has_unknown(expr):
            self.coll["dropped_unknown"] += 1
            return
        try:
            if expr and self.af1 is not None:
                self.af1.write(json.dumps({"schema": expr, "support": p.get("support"), "pids": list(p.get("pids", []))}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # 依赖过滤（需开启且具备上下文；基于第一条 embedding 决策）
        if int(getattr(self.args, "enable_dependency_filter", 0)) == 1:
            decision, info = self._dep_apply_on_first_embedding(getattr(self.args, "mode", "fact_rule"), p, schema_before_dep or expr)
            if decision == "drop":
                self.coll["dropped_dep"] += 1
                try:
                    if self.af2d is not None:
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
                    if self.af2k is not None:
                        self.af2k.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            else:  # skip
                pass

    # 变量闭包检查（结论变量需包含于前提变量集合）
        if bool(getattr(self.args, "enable_var_closure_check", True)) and expr and not self.concl_args_subset_of_prem(expr):
            self.coll["dropped_var_subset"] += 1
            return
        try:
            if expr and self.af3 is not None:
                self.af3.write(json.dumps({"schema": expr, "support": p.get("support"), "pids": list(p.get("pids", []))}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # 清洗前提中的 sameclock/sameside（不影响语义推断的展示性约束）
        try:
            if bool(getattr(self.args, "strip_clock_side_in_schema", 1)) and expr:
                cleaned = self._drop_premises_by_predicates(expr, {"sameclock", "sameside"})
                expr = cleaned if cleaned else expr
        except Exception:
            pass

        # canonical schema dedup
        key = self.canonical_key(expr) if expr else None
        if self.enable_schema_dedup:
            if key and key in self.coll["schema_seen"]:
                self.coll["dedup_schema"] += 1
                return
            if key:
                self.coll["schema_seen"].add(key)

        # 统一计数字段（可能为 int 或 list）
        def _norm_count(v):
            if isinstance(v, int):
                return v
            if isinstance(v, list):
                return len(v)
            return None

        item = {
            "support": p.get("support"),
            "pids_count": _norm_count(p.get("pids")),
            "nodes": _norm_count(p.get("nodes")),
            "edges": _norm_count(p.get("edges")),
            "labels": p.get("labels"),
            "schema": expr,
            "schema_before_dependency": schema_before_dep,
            "rendered": rendered,
        }
        # 透传已有坐标信息（若存在）
        if isinstance(p.get("point_lines"), list):
            item["point_lines"] = list(p["point_lines"])
        if isinstance(p.get("points"), list):
            item["points"] = list(p["points"])
        # 若存在问题级坐标缓存，则补充代表性坐标
        try:
            embs = p.get("embeddings", [])
            if embs and self.pg is not None:
                pid = str(embs[0].get("pid"))
                coords = getattr(self.pg, "point_coords", {}).get(pid)
                if isinstance(coords, dict):
                    pl = coords.get("point_lines")
                    pts = coords.get("points")
                    if isinstance(pl, list) and isinstance(pts, list):
                        item["point_lines"] = list(pl)
                        item["points"] = list(pts)
        except Exception:
            pass
        try:
            if self.af4 is not None:
                self.af4.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass
        if self.do_stream and self.wf is not None:
            self.wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            self.coll["written"] += 1
        else:
            self.coll["patterns_all"].append(item)

    def finalize_stream(self):
        # 若启用流式临时写入，将 NDJSON 收拢到内存列表并删除临时文件。
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

    def minimize_by_premise_subset(self, patterns_all: List[dict]):
    # 同结论下按前提集合键做子集最小化，丢弃前提为他者真超集的项。
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

    def finalize_results(self, meta: Optional[dict] = None):
        # 生成统一输出 JSON，并在 params.filter_flags 与 params.filter_stats 中记录开关与统计。
        args = self.args
        patterns_all = self.finalize_stream()
        if self.enable_subset_min:
            patterns_all = self.minimize_by_premise_subset(patterns_all)

        topn = max(0, int(getattr(args, "top_k", 0)))
        patterns_summary = patterns_all[:topn]
        # flags snapshot
        filter_flags = {
            "drop_unknown_in_schema": bool(self.drop_unknown_effective),
            "auto_disabled_unknown_filter": bool(self.auto_disabled_unknown_filter),
            "enable_dependency_filter": int(getattr(args, "enable_dependency_filter", 0)),
            "enable_var_closure_check": bool(getattr(args, "enable_var_closure_check", True)),
            "strip_clock_side_in_schema": bool(getattr(args, "strip_clock_side_in_schema", 1)),
            "enable_schema_dedup": bool(self.enable_schema_dedup),
            "enable_subset_min": bool(self.enable_subset_min),
            "enable_audit_ndjson": bool(self.enable_audit_ndjson),
        }
        filter_stats = {
            "dropped_unknown": int(self.coll.get("dropped_unknown", 0)),
            "dropped_dep": int(self.coll.get("dropped_dep", 0)),
            "dropped_var_subset": int(self.coll.get("dropped_var_subset", 0)),
            "dedup_schema": int(self.coll.get("dedup_schema", 0)),
        }

        result_obj = {
            "created_at": datetime.now().isoformat(),
            "params": {
                "mode": getattr(args, "mode", None),
                "enable_var_closure_check": bool(getattr(args, "enable_var_closure_check", True)),
                "stream_write": int(getattr(args, "stream_write", 0)),
                "drop_unknown_in_schema": bool(self.drop_unknown_effective),
                "enable_dependency_filter": int(getattr(args, "enable_dependency_filter", 0)),
                "strip_clock_side_in_schema": bool(getattr(args, "strip_clock_side_in_schema", 1)),
                "filter_flags": filter_flags,
                "filter_stats": filter_stats,
            },
            "pattern_count": len(patterns_all),
            "patterns_summary_topN": patterns_summary,
            "patterns": patterns_all,
        }
        if isinstance(meta, dict):
            result_obj.update(meta)

        out_name = ("branched_mining.json" if getattr(args, "mode", "fact_rule")=="fact_rule" else "rules_only_mining.json")
        out_path = os.path.join(self.out_dir, out_name)

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
            for h in (self.af1, self.af2k, self.af2d, self.af3, self.af4, self.af5d):
                if h is not None:
                    h.close()
        except Exception:
            pass
        return result_obj

    # ---------- post-filter: partition by point coverage ----------
    def partition_by_point_coverage(self, items: Iterable[dict] | Iterable[str], with_reason: bool = True,
                                    strip_clock_side: bool = True):
        """
        基于点覆盖关系将 schema 分为三类：
        - error_schemas：前提点并集是结论点集合的真子集（覆盖不足）或 schema 解析异常/缺失结论/缺失前提。
        - discarded_schemas：前提点并集等于结论点集合（无辅助点）。
        - candidate_schemas：其余（通常为前提点并集严格大于结论点集合，存在辅助点）。

        统计时，默认先移除 sameclock/sameside 前提后再计算点集合（strip_clock_side=True）。

        输入：
        - items：可为 schema 字符串迭代器，或包含 "schema" 字段的字典迭代器（与本模块输出 item 兼容）。
        - with_reason：是否返回详细信息（包含 prem_points / concl_points / relation / rendered）。
        - strip_clock_side：是否在统计前移除 sameclock/sameside 前提。

        输出：
        - dict: {"error_schemas": [...], "discarded_schemas": [...], "candidate_schemas": [...]}。
          当 with_reason=True 时，元素为详细记录；否则为原输入元素。
        """

        def _get_schema_and_rendered(x) -> tuple[str, Any, Any]:
            if isinstance(x, str):
                return x, x, None
            if isinstance(x, dict):
                return str(x.get("schema") or ""), x, x.get("rendered")
            return "", x, None

        def _points_from_args(pred: str, args: list[str]) -> list[str]:
            # 忽略 aconst/rconst 尾常量；其他参数全部视作点。
            if pred in {"aconst", "rconst"} and len(args) >= 1:
                return [a for a in args[:-1] if a]
            return [a for a in args if a]

        out = {
            "error_schemas": [],
            "discarded_schemas": [],
            "candidate_schemas": [],
        }

        for it in items:
            schema, original_item, rendered = _get_schema_and_rendered(it)
            if not schema or "=>" not in schema:
                rec = original_item if not with_reason else {
                    "item": original_item,
                    "schema": schema,
                    "schema_cleaned": schema,
                    "prem_points": [],
                    "concl_points": [],
                    "relation": "unknown",
                    "reason": "malformed_or_no_arrow",
                }
                if with_reason and rendered is not None:
                    rec["rendered"] = rendered
                out["error_schemas"].append(rec)
                continue

            try:
                schema_for_calc = self._drop_premises_by_predicates(schema, {"sameclock", "sameside"}) if strip_clock_side else schema
                prem, concl = self.parse_schema(schema_for_calc)
                if not prem:
                    rec = original_item if not with_reason else {
                        "item": original_item,
                        "schema": schema,
                        "schema_cleaned": schema_for_calc,
                        "prem_points": [],
                        "concl_points": [],
                        "relation": "unknown",
                        "reason": "no_premises",
                    }
                    if with_reason and rendered is not None:
                        rec["rendered"] = rendered
                    out["error_schemas"].append(rec)
                    continue
                if not concl:
                    rec = original_item if not with_reason else {
                        "item": original_item,
                        "schema": schema,
                        "schema_cleaned": schema_for_calc,
                        "prem_points": [],
                        "concl_points": [],
                        "relation": "unknown",
                        "reason": "no_conclusion",
                    }
                    if with_reason and rendered is not None:
                        rec["rendered"] = rendered
                    out["error_schemas"].append(rec)
                    continue

                # collect union of premise points and conclusion points
                prem_points_set = set()
                for pred, args in prem:
                    prem_points_set.update(_points_from_args(pred, args))
                concl_pred, concl_args = concl
                concl_points_set = set(_points_from_args(concl_pred, concl_args))

                # decide relation and bucket
                relation = "unknown"
                bucket = "candidate_schemas"
                if concl_points_set and prem_points_set:
                    if prem_points_set < concl_points_set:
                        relation = "subset"
                        bucket = "error_schemas"
                    elif prem_points_set == concl_points_set:
                        relation = "equal"
                        bucket = "discarded_schemas"
                    else:
                        relation = "superset"
                        bucket = "candidate_schemas"
                else:
                    # 缺失任一集合，视作错误分类
                    relation = "unknown"
                    bucket = "error_schemas"

                if with_reason:
                    rec = {
                        "item": original_item,
                        "schema": schema,
                        "schema_cleaned": schema_for_calc,
                        "prem_points": sorted(list(prem_points_set)),
                        "concl_points": sorted(list(concl_points_set)),
                        "relation": relation,
                    }
                    if rendered is not None:
                        rec["rendered"] = rendered
                    out[bucket].append(rec)
                else:
                    out[bucket].append(original_item)

            except Exception as e:
                rec = original_item if not with_reason else {
                    "item": original_item,
                    "schema": schema,
                    "schema_cleaned": schema,
                    "prem_points": [],
                    "concl_points": [],
                    "relation": "unknown",
                    "reason": f"exception: {type(e).__name__}",
                }
                if with_reason and rendered is not None:
                    rec["rendered"] = rendered
                out["error_schemas"].append(rec)

        return out

    # ---------- post-filter: partition by rely_on on conclusion ----------
    def partition_by_rely_on(self, candidate_items: Iterable[dict], with_reason: bool = True,
                             one_hop_only: bool = True):
        """
        基于结论点的 rely_on 信息，对 candidate_schemas 做二次筛选。

        输入（与 scripts/filt_schemas.py 产物兼容）：
        - candidate_items: `partition_by_point_coverage.json` 中的 candidate_schemas 数组元素，每个元素应包含：
          - prem_points: List[str]  （变量名域 X1...，已按约定剔除 sameclock/sameside）
          - concl_points: List[str] （变量名域 X1...）
          - item: 原始条目字典，建议包含 rendered/schema 等
            - rendered.schema_vars: Dict[orig_point_name -> var_name]，用于名称域对齐
            - rendered.rely_on 或 item.rely_on 或 item.point_rely_on: Dict[point_name -> Iterable[point_name]]
              依赖可在原点名域或变量名域；将基于 schema_vars 做映射到变量名域

        规则：
        - pre_union = set(prem_points)
        - union_rely = concl_points ∪ (⋃ rely_on(concl_point))
        分类：
          1) pre_union == union_rely               -> discarded_schemas
          2) pre_union ⊂ union_rely（真子集）      -> discarded_schemas
          3) union_rely ⊂ pre_union（真子集）      -> candidate_schemas
          4) 互不包含/部分交叠无包含               -> candidate_schemas_type2
        解析失败/缺字段 -> error_schemas

        返回：dict，同上述四分类；当 with_reason=True 时，记录诊断字段（pre_union/union_rely/relation/reason/rendered 等）。
        """

        def _get_schema_vars(d: dict | None) -> dict:
            if not isinstance(d, dict):
                return {}
            sv = d.get("schema_vars")
            return sv if isinstance(sv, dict) else {}

        def _get_rely_map_any(rec: dict) -> dict:
            # 优先 rendered.rely_on，其次 item.rely_on，再次 item.point_rely_on
            item = rec.get("item") if isinstance(rec, dict) else None
            rendered = rec.get("rendered") if isinstance(rec, dict) else None
            if isinstance(rendered, dict):
                rm = rendered.get("rely_on")
                if isinstance(rm, dict):
                    return rm
            if isinstance(item, dict):
                rm = item.get("rely_on")
                if isinstance(rm, dict):
                    return rm
                rm = item.get("point_rely_on")
                if isinstance(rm, dict):
                    return rm
            return {}

        def _map_name_to_var(name: str, schema_vars: dict) -> str:
            # schema_vars: orig_name -> var_name（如 'a' -> 'X1'）
            if not name:
                return name
            if name.startswith("X"):
                return name
            v = schema_vars.get(name)
            return v if isinstance(v, str) and v else name

        out = {
            "discarded_schemas": [],
            "candidate_schemas": [],
            "candidate_schemas_type2": [],
            "error_schemas": [],
        }

        for rec in candidate_items:
            try:
                prem_points = list(rec.get("prem_points") or [])
                concl_points = list(rec.get("concl_points") or [])
                if not prem_points or not concl_points:
                    if with_reason:
                        out["error_schemas"].append({
                            "item": rec.get("item"),
                            "schema": rec.get("schema"),
                            "prem_points": prem_points,
                            "concl_points": concl_points,
                            "reason": "missing_prem_or_concl_points",
                        })
                    else:
                        out["error_schemas"].append(rec)
                    continue

                # 名称域对齐：把 rely_on map 到变量名域
                item = rec.get("item") if isinstance(rec, dict) else None
                rendered = rec.get("rendered") or (item.get("rendered") if isinstance(item, dict) else None)
                schema_vars = _get_schema_vars(rendered)
                rely_map_any = _get_rely_map_any(rec)
                if not rely_map_any:
                    if with_reason:
                        e = {
                            "item": item,
                            "schema": rec.get("schema"),
                            "prem_points": prem_points,
                            "concl_points": concl_points,
                            "reason": "missing_rely_on",
                        }
                        if isinstance(rendered, dict):
                            e["rendered"] = rendered
                        out["error_schemas"].append(e)
                    else:
                        out["error_schemas"].append(rec)
                    continue

                # 将 rely_on 的 key 和 value 都映射为变量名域
                rely_map_var: dict[str, set[str]] = {}
                for k, vs in rely_map_any.items():
                    if not isinstance(vs, (list, tuple, set)):
                        continue
                    k_var = _map_name_to_var(str(k), schema_vars)
                    vals = set()
                    for u in vs:
                        u_var = _map_name_to_var(str(u), schema_vars)
                        vals.add(u_var)
                    rely_map_var.setdefault(k_var, set()).update(vals)

                pre_union = set(prem_points)
                # union_rely = concl_points ∪ rely_on(concl_point)
                union_rely = set(concl_points)
                for cp in concl_points:
                    deps = set(rely_map_var.get(cp, set()))
                    union_rely |= deps
                    if not one_hop_only:
                        # 简单广度式传递闭包（可选）；默认不开启
                        frontier = set(deps)
                        seen = set(deps)
                        while frontier:
                            nxt = set()
                            for q in list(frontier):
                                for z in rely_map_var.get(q, set()):
                                    if z not in seen:
                                        seen.add(z); nxt.add(z)
                            frontier = nxt
                        union_rely |= seen

                # 四类分类
                if pre_union == union_rely:
                    bucket = "discarded_schemas"; relation = "equal"
                elif pre_union < union_rely:
                    bucket = "discarded_schemas"; relation = "pre_subset_of_union_rely"
                elif union_rely < pre_union:
                    bucket = "candidate_schemas"; relation = "union_rely_subset_of_pre"
                else:
                    bucket = "candidate_schemas_type2"; relation = "incomparable"

                if with_reason:
                    rec_out = {
                        "item": item,
                        "schema": rec.get("schema"),
                        "pre_union": sorted(list(pre_union)),
                        "union_rely": sorted(list(union_rely)),
                        "relation": relation,
                    }
                    if isinstance(rendered, dict):
                        rec_out["rendered"] = rendered
                    out[bucket].append(rec_out)
                else:
                    out[bucket].append(rec)

            except Exception as e:
                if with_reason:
                    out["error_schemas"].append({
                        "item": rec.get("item") if isinstance(rec, dict) else rec,
                        "schema": rec.get("schema") if isinstance(rec, dict) else None,
                        "reason": f"exception: {type(e).__name__}",
                    })
                else:
                    out["error_schemas"].append(rec)

        return out
