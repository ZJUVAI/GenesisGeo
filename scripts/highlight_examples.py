#!/usr/bin/env python3
"""几何图形高亮绘制功能 - 使用示例

演示如何使用 draw_bw 的高亮功能：
- 不高亮（全黑色）
- 按construction clause索引高亮
- 按predicate字符串高亮
- 混合使用

运行：
    conda run -n Genesis python scripts/highlight_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newclid.api import GeometricSolverBuilder
from newclid.numerical.draw_bw import draw_bw


SEED = 998244353
OUTPUT_DIR = Path("highlight_examples_output")


def example_problem(problem_text: str):
    """构建并返回solver和builder"""
    builder = GeometricSolverBuilder(seed=SEED)
    builder.load_problem_from_txt(problem_text)
    solver = builder.build(max_attempts=100)
    return solver, builder


# ==============================================================================
# 示例1: 基本使用 - 全黑色
# ==============================================================================

def example1_basic():
    """示例1: 基本绘制，全部黑色"""
    print("示例1: 基本绘制（全黑色）")

    problem = "a b c = triangle a b c; o = circle o a b c"
    solver, builder = example_problem(problem)

    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex1_basic_black.svg",
    )
    print("  ✓ ex1_basic_black.svg")


# ==============================================================================
# 示例2: 高亮construction clauses
# ==============================================================================

def example2_highlight_clauses():
    """示例2: 按clause索引高亮"""
    print("\n示例2: 按clause索引高亮")

    problem = "a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c"
    # Clause 0: a b c = triangle a b c
    # Clause 1: d = midpoint d a b
    # Clause 2: e = midpoint e b c
    solver, builder = example_problem(problem)

    # 2a: 高亮单个clause
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex2a_clause_1.svg",
        highlight_clauses=1,  # 可以传int
    )
    print("  ✓ ex2a_clause_1.svg - 高亮clause 1（中点d）")

    # 2b: 高亮多个clauses
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex2b_clauses_0_2.svg",
        highlight_clauses=[0, 2],  # 或传list[int]
    )
    print("  ✓ ex2b_clauses_0_2.svg - 高亮clause 0和2（三角形+中点e）")


# ==============================================================================
# 示例3: 高亮predicates
# ==============================================================================

def example3_highlight_predicates():
    """示例3: 按predicate字符串高亮"""
    print("\n示例3: 按predicate字符串高亮")

    problem = "a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c"
    solver, builder = example_problem(problem)

    # 3a: 高亮单个predicate
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex3a_midp.svg",
        highlight_predicates=["midp d a b"],  # midpoint谓词
    )
    print("  ✓ ex3a_midp.svg - 高亮midp d a b（线段ab）")

    # 3b: 高亮多个predicates
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex3b_two_midps.svg",
        highlight_predicates=[
            "midp d a b",
            "midp e b c",
        ],
    )
    print("  ✓ ex3b_two_midps.svg - 高亮两个中点")


# ==============================================================================
# 示例4: 常见predicate类型
# ==============================================================================

def example4_predicate_types():
    """示例4: 常见predicate类型演示"""
    print("\n示例4: 常见predicate类型")

    problem = "a b c = triangle a b c; o = circle o a b c"
    solver, builder = example_problem(problem)

    # 4a: cong - 线段相等
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex4a_cong.svg",
        highlight_predicates=["cong a b b c"],  # 画线段ab和bc
    )
    print("  ✓ ex4a_cong.svg - cong（线段ab和bc）")

    # 4b: cyclic - 共圆
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex4b_cyclic.svg",
        highlight_predicates=["cyclic a b c o"],  # 画外接圆
    )
    print("  ✓ ex4b_cyclic.svg - cyclic（外接圆）")

    # 4c: coll - 共线
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex4c_coll.svg",
        highlight_predicates=["coll a o c"],  # 画线段a-o-c
    )
    print("  ✓ ex4c_coll.svg - coll（链式线段）")


# ==============================================================================
# 示例5: 更多predicate类型
# ==============================================================================

def example5_more_predicates():
    """示例5: 更多predicate类型"""
    print("\n示例5: 更多predicate类型")

    problem = "a b c d = rectangle a b c d"
    solver, builder = example_problem(problem)

    # 5a: para - 平行
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex5a_para.svg",
        highlight_predicates=["para a b c d"],  # 画线段ab和cd
    )
    print("  ✓ ex5a_para.svg - para（平行线段）")

    # 5b: perp - 垂直
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex5b_perp.svg",
        highlight_predicates=["perp a b b c"],  # 画线段ab和bc
    )
    print("  ✓ ex5b_perp.svg - perp（垂直线段）")

    # 5c: 多个predicates
    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex5c_multi.svg",
        highlight_predicates=[
            "para a b c d",
            "perp a b b c",
        ],
    )
    print("  ✓ ex5c_multi.svg - 多个predicate（平行+垂直）")


# ==============================================================================
# 示例6: 混合使用
# ==============================================================================

def example6_mixed():
    """示例6: 同时使用clause和predicate高亮"""
    print("\n示例6: 混合使用")

    problem = "a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c"
    solver, builder = example_problem(problem)

    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex6_mixed.svg",
        highlight_clauses=[0],  # 高亮三角形
        highlight_predicates=["midp d a b"],  # 同时高亮中点
    )
    print("  ✓ ex6_mixed.svg - 同时高亮clause和predicate")


# ==============================================================================
# 示例7: 点不存在时的处理
# ==============================================================================

def example7_missing_points():
    """示例7: predicate中的点不存在时会静默跳过"""
    print("\n示例7: 点不存在的处理")

    problem = "a b c = triangle a b c"
    solver, builder = example_problem(problem)

    draw_bw(
        solver.proof,
        builder.problemJGEX,
        save_to=OUTPUT_DIR / "ex7_missing.svg",
        highlight_predicates=[
            "cong a b c d",  # 点d不存在，跳过
            "coll a b c",  # 点都存在，会画
        ],
    )
    print("  ✓ ex7_missing.svg - 点不存在时静默跳过（只画了coll）")


# ==============================================================================
# 示例8: 所有支持的predicate类型参考
# ==============================================================================

def example8_reference():
    """示例8: 所有支持的predicate类型参考"""
    print("\n示例8: Predicate类型参考")
    print("  支持的predicate类型：")
    print("    • cong a b c d       - 线段相等（画ab、cd）")
    print("    • para a b c d       - 平行（画ab、cd）")
    print("    • perp a b c d       - 垂直（画ab、cd）")
    print("    • coll a b c         - 共线（画a-b-c链）")
    print("    • midp m a b         - 中点（画线段ab）")
    print("    • cyclic a b c d     - 共圆（画外接圆）")
    print("    • eqangle a b c d e f g h  - 角相等（画ab,cd,ef,gh）")
    print("    • eqratio a b c d e f g h  - 比例相等（画ab,cd,ef,gh）")
    print("    • simtri a b c d e f       - 相似三角形（画△abc和△def）")
    print("    • contri a b c d e f       - 全等三角形（画△abc和△def）")
    print("    • circle o a b c           - 外接圆（画圆）")


# ==============================================================================
# Main
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("几何图形高亮功能 - 使用示例")
    print("=" * 70)

    example1_basic()
    example2_highlight_clauses()
    example3_highlight_predicates()
    example4_predicate_types()
    example5_more_predicates()
    example6_mixed()
    example7_missing_points()
    example8_reference()

    print("\n" + "=" * 70)
    print(f"✅ 所有示例已生成，输出目录: {OUTPUT_DIR}/")
    print("=" * 70)

    print("\n快速参考：")
    print("  1. 不高亮:    draw_bw(proof, problem, save_to='out.svg')")
    print("  2. 高亮clause: draw_bw(..., highlight_clauses=[0,1])")
    print("  3. 高亮pred:  draw_bw(..., highlight_predicates=['cong a b c d'])")
    print("  4. 混合使用:   两个参数都传")
    print("\nClause索引从0开始，用分号';'分隔计数")
    print("Predicate字符串格式: '<类型> <点名1> <点名2> ...'")


if __name__ == "__main__":
    main()
