#!/usr/bin/env python3
"""
使用 SchemaFilter 按步骤对挖掘得到的模式进行筛选，并写出统一的最终结果 JSON（可选产出审计 NDJSON）。

用法：python scripts/filt_schemas.py

所有参数均在文件顶部常量中配置（不提供命令行参数）。
"""
from __future__ import annotations

import json
from pathlib import Path
import os, sys
from types import SimpleNamespace
from typing import Iterable, List, Dict


# ---------------- 配置区（按需修改） ----------------
# 仓库根目录：本脚本位于 repo_root/scripts/ 下
REPO_ROOT = Path(__file__).resolve().parent.parent

# 输入文件：可直接使用挖掘产物（包含 {"patterns": [...]} 的 JSON）
INPUT_PATH = REPO_ROOT / "src/newclid/data_discovery/data/branched_mining.json"
# 模式：影响输出文件名与部分渲染逻辑，可选 "fact_rule" 或 "rule_only"
MODE = "fact_rule"

# 输出目录：放置（可选）审计文件与最终结果 JSON
OUTPUT_DIR = REPO_ROOT / "src/newclid/data_discovery/data/filt_out"

# 步骤开关（默认：最小化过滤，不生成审计）
# 是否丢弃包含 unknown() 的 schema（通常数据已无 unknown，可保持 False）
DROP_UNKNOWN_IN_SCHEMA = False
# 若为 True，则先预扫描 unknown；若不存在则自动关闭 unknown 过滤并在结果中标记
AUTO_DISABLE_UNKNOWN_IF_ABSENT = True
# 依赖过滤（需要 pg/miner 上下文；此脚本默认为 0 不启用）
ENABLE_DEPENDENCY_FILTER = 0
# 变量闭包检查（结论变量应为前提变量子集），默认关闭避免过度丢弃
ENABLE_VAR_CLOSURE_CHECK = True
# 清洗前提中的 sameclock/sameside（通常安全，建议保持 True）
STRIP_CLOCK_SIDE_IN_SCHEMA = True
# 规范化 schema 去重（变量重命名 + 前提无序），通常应开启
ENABLE_SCHEMA_DEDUP = True
# 同结论“前提集合最小化”（可能过于激进，默认关闭，验证后再开）
ENABLE_SUBSET_MIN = False
# 是否输出阶段审计 NDJSON（默认 False，不产审计，仅保留最终 JSON）
ENABLE_AUDIT_NDJSON = True
# 顶层汇总 patterns_summary_topN 的条数（0 表示不生成概要仅输出完整列表）
TOP_K = 0
# 是否启用流式写入临时 NDJSON（0：关闭；一般无需开启）
STREAM_WRITE = 0

# 额外审计：按点覆盖关系分类（默认关闭，不影响主流程）
ENABLE_POINT_COVERAGE_AUDIT = True
# 点覆盖审计时是否先移除 sameclock/sameside 再统计（与需求一致）
PARTITION_STRIP_CLOCK_SIDE = True
# 点覆盖审计输出文件名
PARTITION_AUDIT_BASENAME = "partition_by_point_coverage.json"

# 二次筛选：基于 rely_on 的分类（仅在启用点覆盖审计后可使用其输出作为输入）
ENABLE_RELY_ON_PARTITION = True
RELY_ON_ONE_HOP = True  # 仅一跳依赖，关闭则做简单传递闭包
RELY_ON_INPUT_BASENAME = PARTITION_AUDIT_BASENAME  # 复用点覆盖审计输出
RELY_ON_OUTPUT_BASENAME = "partition_by_rely_on.json"

# 二次筛选后的可视化：默认开启；默认数量 0 表示不限制
ENABLE_POST_VISUALIZE = True
POST_VIZ_BUCKETS = ["candidate_schemas", "candidate_schemas_type2"]
POST_VIZ_BASE_OUTDIR_NAME = "schema_fig"  # 基础输出子目录名，位于 OUTPUT_DIR 下
POST_VIZ_MAX_PER_BUCKET = 0
POST_VIZ_OVERWRITE = True
POST_VIZ_LABEL_MODE = "legend"       # "full" | "short" | "legend"; 在二次筛选可视化时默认用 legend 解决溢出
POST_VIZ_HIGHLIGHT_KIND = True


def load_patterns_from_final_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    arr = data.get("patterns")
    if not isinstance(arr, list):
        raise SystemExit("Invalid input JSON: missing 'patterns' array")
    return arr


def _prescan_unknown(patterns: List[Dict]) -> int:
    """轻量预扫 unknown() 的出现次数，用于自动关闭 unknown 过滤。"""
    cnt = 0
    for p in patterns:
        sch = p.get("schema") or ""
        if "unknown(" in sch or sch.strip().startswith("unknown()"):
            cnt += 1
    return cnt


def main() -> None:
    # 确保可导入项目源码（静态检查可能报未解析，运行时可用）
    SRC_ROOT = REPO_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    # 构造与 SchemaFilter 兼容的参数对象
    args = SimpleNamespace(
        mode=MODE,
        drop_unknown_in_schema=DROP_UNKNOWN_IN_SCHEMA,
        auto_disabled_unknown_filter=False,  # will be set after prescan
        enable_dependency_filter=ENABLE_DEPENDENCY_FILTER,
        enable_var_closure_check=ENABLE_VAR_CLOSURE_CHECK,
        strip_clock_side_in_schema=STRIP_CLOCK_SIDE_IN_SCHEMA,
        enable_schema_dedup=ENABLE_SCHEMA_DEDUP,
        enable_subset_min=ENABLE_SUBSET_MIN,
        enable_audit_ndjson=ENABLE_AUDIT_NDJSON,
        top_k=TOP_K,
        stream_write=STREAM_WRITE,
    )

    out_dir = str(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 延迟导入（在 sys.path 注入之后）
    from newclid.data_discovery.schema_filter import SchemaFilter

    # 读取挖掘产物的 patterns
    patterns = load_patterns_from_final_json(INPUT_PATH)

    # 预扫 unknown：若不存在则自动关闭 unknown 过滤
    if AUTO_DISABLE_UNKNOWN_IF_ABSENT and not DROP_UNKNOWN_IN_SCHEMA:
        unk_count = _prescan_unknown(patterns)
        if unk_count == 0:
            args.drop_unknown_in_schema = False
            args.auto_disabled_unknown_filter = True
        else:
            # 保持用户设置（此处默认 False）
            pass

    # 运行筛选（本脚本不提供依赖过滤上下文 ctx）
    filt = SchemaFilter(args=args, out_dir=out_dir, ctx=None)
    filt.process_patterns(patterns)
    meta = {"source": str(INPUT_PATH)}
    result_obj = filt.finalize_results(meta=meta)

    # 可选：对（输入）patterns 进行点覆盖关系分类审计
    if ENABLE_POINT_COVERAGE_AUDIT:
        try:
            final_patterns = result_obj.get("patterns", []) if isinstance(result_obj, dict) else []
            part = filt.partition_by_point_coverage(final_patterns, with_reason=True, strip_clock_side=PARTITION_STRIP_CLOCK_SIDE)
            summary = {k: len(v) for k, v in part.items()}
            print("Point coverage summary:", json.dumps(summary, ensure_ascii=False))
            outp = OUTPUT_DIR / PARTITION_AUDIT_BASENAME
            outp.write_text(json.dumps(part, ensure_ascii=False, indent=2))
            print(f"Point coverage audit written: {outp}")
        except Exception as e:
            print(f"[WARN] point coverage audit failed: {type(e).__name__}: {e}")

    # 依赖二次筛选：基于结论点的 rely_on 信息对 candidate_schemas 进行四类划分
    if ENABLE_RELY_ON_PARTITION:
        try:
            in_path = OUTPUT_DIR / RELY_ON_INPUT_BASENAME
            if not in_path.exists():
                print(f"[WARN] rely_on partition skipped: input not found: {in_path}")
            else:
                data = json.loads(in_path.read_text(encoding="utf-8"))
                cand = data.get("candidate_schemas") if isinstance(data, dict) else None
                if not isinstance(cand, list):
                    print("[WARN] rely_on partition skipped: candidate_schemas missing or not a list")
                else:
                    part2 = filt.partition_by_rely_on(cand, with_reason=True, one_hop_only=RELY_ON_ONE_HOP)
                    out2 = OUTPUT_DIR / RELY_ON_OUTPUT_BASENAME
                    out2.write_text(json.dumps(part2, ensure_ascii=False, indent=2))
                    summary2 = {k: len(v) for k, v in part2.items()}
                    print("Rely-on partition summary:", json.dumps(summary2, ensure_ascii=False))
                    print(f"Rely-on partition written: {out2}")

                    # 可选：基于二次筛选结果进行可视化，仅绘制 candidate_* 两类
                    if ENABLE_POST_VISUALIZE:
                        try:
                            # 延迟导入可视化工具类
                            from newclid.data_discovery.schema_visualizer import SchemaVisualizer
                            base_outdir = OUTPUT_DIR / POST_VIZ_BASE_OUTDIR_NAME
                            viz = SchemaVisualizer()
                            stats = viz.render_from_rely_on_file(
                                out2,
                                buckets=POST_VIZ_BUCKETS,
                                base_out_dir=base_outdir,
                                max_items_per_bucket=POST_VIZ_MAX_PER_BUCKET,
                                overwrite=POST_VIZ_OVERWRITE,
                                label_mode=POST_VIZ_LABEL_MODE,
                                highlight_kind=POST_VIZ_HIGHLIGHT_KIND,
                                enable_union_rely_styling=True,
                                style_opts={"font_size": 8, "figsize": (16, 9)},
                            )
                            print("Post visualize (rely_on) stats:", json.dumps(stats, ensure_ascii=False))
                            print(f"Post visualize output base dir: {base_outdir}")
                        except Exception as e:
                            print(f"[WARN] post visualize failed: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[WARN] rely_on partition failed: {type(e).__name__}: {e}")

    print(f"筛选完成，输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
