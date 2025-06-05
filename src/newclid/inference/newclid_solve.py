import logging
from pathlib import Path
from typing import Any, Optional
import copy
import time

from newclid.api import GeometricSolverBuilder
from newclid.formulations.clause import Clause

# 导入beam_search和translate
from beam_search import newclid_beam_search
from translate import translate_constrained_to_constructive

class AlphaGeometry:
    def __init__(
        self,
        model,
        seed: int = 998244353,
        input_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        time_limit: int = 600,
    ):
        """
        model: 语言模型推理接口，需实现 inference(problem_des)
        seed: 随机种子
        input_file: 题目文件路径
        output_dir: 输出目录
        time_limit: 单题最大求解时间（秒）
        """
        self.model = model
        self.seed = seed
        self.input_file = input_file
        self.output_dir = output_dir
        self.time_limit = time_limit

    def run_ddar(self, solver, out_file: Optional[Path] = None) -> bool:
        """
        运行DDAR（符号推理引擎），成功则写出证明并返回True。
        """
        success = solver.run(time_limit=self.time_limit)
        if success and out_file:
            solver.write_proof_steps(out_file)
        return success

    def try_translate_lm_output(self, lm_output: str, solver) -> str:
        """
        将语言模型输出翻译为可插入的构造语句。
        支持格式如: "C perp A B C D" 或 "A coll A B C"
        """
        try:
            tokens = lm_output.strip().split()
            if len(tokens) < 2:
                return "ERROR: Invalid LLM output"
            point = tokens[0]
            predicate = tokens[1]
            args = tokens[2:]
            pred, new_args = translate_constrained_to_constructive(point, predicate, args)
            return f"{point} {pred} {' '.join(new_args)}"
        except Exception as e:
            return f"ERROR: {e}"

    def insert_aux_to_premise(self, problem_str: str, aux_str: str) -> str:
        """
        将辅助点构造插入到问题描述中。
        简单实现：在结尾?前插入
        """
        problem_str = problem_str.strip()
        if problem_str.endswith("?"):
            return problem_str[:-1].rstrip() + "; " + aux_str + " ?"
        else:
            return problem_str + "; " + aux_str

    def solve(
        self,
        problem_txt: str,
        search_depth: int,
        beam_size: int,
        out_file: str,
    ) -> bool:
        """
        使用LLM+DDAR混合推理求解几何问题。
        """
        # 1. 初始化问题
        solver_builder = GeometricSolverBuilder(seed=self.seed)
        solver_builder.load_problem_from_txt(problem_txt)
        logging.info("start init solver")
        solver = solver_builder.build()
        problem_des = problem_txt
        logging.info("finish init solver")

        # 2. 先尝试符号推理
        if self.run_ddar(solver, Path(out_file)):
            return True

        logging.debug("Initial DDAR failed, starting beam search...")

        # 3. 不用beam_search，直接while循环尝试
        max_try = search_depth * beam_size
        tried_aux = set()
        for _ in range(max_try):
            # 用模型预测一个辅助构造
            lm_out = self.model.inference(problem_des)
            if not lm_out:
                break
            aux_pred = self.try_translate_lm_output(lm_out, solver)
            if aux_pred.startswith("ERROR") or aux_pred in tried_aux:
                continue
            tried_aux.add(aux_pred)
            try:
                clauses = Clause.parse_line(aux_pred)
            except Exception as e:
                logging.warning(f"[Iterative] Clause parse failed: {aux_pred}, error: {e}")
                continue

            # 深拷贝solver，避免污染
            new_solver = copy.deepcopy(solver)
            try:
                for clause in clauses:
                    new_solver.proof.add_construction(clause)
            except Exception as e:
                logging.warning(f"[Iterative] Add construction failed: {aux_pred}, error: {e}")
                continue

            # 构造新的问题描述
            problem_des = self.insert_aux_to_premise(problem_des, aux_pred)

            # 再次尝试符号推理
            if self.run_ddar(new_solver, Path(out_file)):
                logging.info("[Iterative] Solved with auxiliary construction.")
                return True
        return False  # 未能成功求解

        # # 3. 调用统一的beam search流程（如需启用）
        # solved = newclid_beam_search(
        #     model=self.model,
        #     solver=solver,
        #     problem_des=problem_des,
        #     search_depth=search_depth,
        #     beam_size=beam_size,
        #     out_file=out_file,
        #     run_ddar_fn=self.run_ddar,
        #     try_translate_lm_output_fn=self.try_translate_lm_output,
        #     insert_aux_to_premise_fn=self.insert_aux_to_premise,
        # )
        # return solved

    def solve_from_file(
        self,
        search_depth: int = 3,
        beam_size: int = 3,
    ):
        """
        从文件批量读取题目，逐个求解并输出到指定目录。
        """
        assert self.input_file is not None, "input_file must be set"
        assert self.output_dir is not None, "output_dir must be set"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.input_file, "r", encoding="utf-8") as fin:
            problems = fin.read().split("\n\n")  # 假设每题用空行分隔

        for idx, problem_txt in enumerate(problems):
            problem_txt = problem_txt.strip()
            if not problem_txt:
                continue
            out_folder = self.output_dir / f"problem_{idx+1:03d}"
            out_folder.mkdir(parents=True, exist_ok=True)
            out_file = out_folder / "proof_steps.txt"
            logging.info(f"Solving problem {idx+1} ...")
            start_time = time.time()
            solved = False
            try:
                solved = self.solve(
                    problem_txt=problem_txt,
                    search_depth=search_depth,
                    beam_size=beam_size,
                    out_file=str(out_file),
                )
            except Exception as e:
                logging.error(f"Error solving problem {idx+1}: {e}")
            elapsed = time.time() - start_time
            with open(out_folder / "result.txt", "w", encoding="utf-8") as fout:
                fout.write(f"Solved: {solved}\nTime: {elapsed:.2f}s\n")
            logging.info(f"Problem {idx+1} solved: {solved}, time: {elapsed:.2f}s")

    def test_solve(self):
        """
        单元测试：对单个题目进行求解，输出结果到控制台。
        """
        problem_txt = (
            "a = free a; "
            "b c d e = quadrangle b c d e; "
            "f = on_aline f a e c d b; "
            "g = on_dia g a c, on_line g f c; "
            "h = eq_triangle h e c; "
            "i = on_pline i h e g; "
            "j = shift j g c i "
            "? eqangle c f c g c i g j"
        )
        print("测试题目：")
        print(problem_txt)
        out_file = "test_proof_steps.txt"
        search_depth = 3
        beam_size = 3
        solved = self.solve(
            problem_txt=problem_txt,
            search_depth=search_depth,
            beam_size=beam_size,
            out_file=str(out_file),
        )
        print(f"是否求解成功: {solved}")
        if solved:
            with open(out_file, "r", encoding="utf-8") as fin:
                print("证明步骤：")
                print(fin.read())

        print("假设直接辅助构造失败，尝试添加辅助构造")
        solved = False
        print(f"是否求解成功: {solved}")

        # 1. 初始化问题
        solver_builder = GeometricSolverBuilder(seed=self.seed)
        solver_builder.load_problem_from_txt(problem_txt)
        logging.info("start init solver")
        solver = solver_builder.build()
        problem_des = problem_txt
        logging.info("finish init solver")

        if solved:
            with open(out_file, "r", encoding="utf-8") as fin:
                print("证明步骤：")
                print(fin.read())
        else:
            # 这里模拟LLM输出和辅助构造流程
            tried_aux = []
            lm_out = "k = coll c f k"
            aux_pred = self.try_translate_lm_output(lm_out, solver)
            logging.info(f'翻译后的辅助构造：{aux_pred}')
            aux_pred = "k = on_line k f c "
            tried_aux.append(aux_pred)
            try:
                clauses = Clause.parse_line(aux_pred)
            except Exception as e:
                logging.warning(f"[Iterative] Clause parse failed: {aux_pred}, error: {e}")

            # 深拷贝solver，避免污染
            new_solver = copy.deepcopy(solver)
            try:
                for clause in clauses:
                    new_solver.proof.add_construction(clause)
            except Exception as e:
                logging.warning(f"[Iterative] Add construction failed: {aux_pred}, error: {e}")

            # 构造新的问题描述
            problem_des = self.insert_aux_to_premise(problem_des, aux_pred)

            solved = self.run_ddar(new_solver, Path(out_file))

            if solved:
                with open(out_file, "r", encoding="utf-8") as fin:
                    print("证明步骤：")
                    print(fin.read())

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=False, help="题目文件路径")
    parser.add_argument("--output_dir", type=Path, required=False, help="输出目录")
    parser.add_argument("--search_depth", type=int, default=3, help="beam search深度")
    parser.add_argument("--beam_size", type=int, default=3, help="beam search宽度")
    parser.add_argument("--time_limit", type=int, default=600, help="单题最大求解时间（秒）")
    parser.add_argument("--seed", type=int, default=998244353, help="随机种子")
    parser.add_argument("--log_level", required=False, default="info", choices=["debug", "info", "warning", "error"])

    # 这里假设model实例化方式由用户自行实现
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # 用户需自行实现model加载
    model = ...  # 请替换为实际模型加载代码

    solver = AlphaGeometry(
        model=model,
        seed=args.seed,
        input_file=args.input_file,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
    )
    solver.test_solve()
    # solver.solve_from_file(
    #     search_depth=args.search_depth,
    #     beam_size=args.beam_size,
    # )

if __name__ == "__main__":
    main()

# a = free a; b c d e = quadrangle b c d e; f = on_aline f a e c d b; g = on_dia g a c, on_line g f c; h = eq_triangle h e c; i = on_pline i h e g; j = shift j g c i; k = on_dia k c d, on_line k f c ? eqangle c f c g c i g j",,5,"
# <problem>
# a:; b:; c:; d:; e:; f: eqangle a e a f b d c d [000]; g: perp a g c g [001] coll c f g [002]; h: cong c e e h [003] cong c e c h [004]; i: para e g h i [005]; j: cong c i g j [006] cong c j g i [007] ? eqangle c f c g c i g j
# </problem>
# <aux>
# k: perp c k d k [008] coll c f k [009]
# </aux>
# <analysis>
# coll c f g [002]
# cong c i g j [006]
# cong c j g i [007]
# </analysis>
# <numerical_check>
# sameclock c i j g j i [010]
# </numerical_check>
# <proof>
# cong c i g j [006] (Ratio Chasing)=> eqratio c i i j g j i j [011]
# cong c j g i [007] (Ratio Chasing)=> eqratio c j i j g i i j [012]
# eqratio c i i j g j i j [011] eqratio c j i j g i i j [012] sameclock c i j g j i [010] (r60 SSS Similarity of triangles (Direct))=> simtri c i j g j i [013]
# simtri c i j g j i [013] (r52 Properties of similar triangles (Direct))=> eqangle c i i j g j i j [014]
# coll c f g [002] coll c f k [009] eqangle c i i j g j i j [014] (Angle Chasing)=> eqangle c f c g c i g j [015]
# </proof>