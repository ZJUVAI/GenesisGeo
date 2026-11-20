from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional, List, Tuple
from typing_extensions import Self
from fractions import Fraction
from functools import cache


from newclid.agent.ddarn import DDARN
from newclid.formulations.definition import DefinitionJGEX
from newclid.ddar_build_input import build_ddar_input
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.load_geogebra import load_geogebra
from newclid.numerical.draw_figure import draw_figure
from newclid.algebraic_reasoning.algebraic_manipulator import (
    AlgebraicManipulator,
)
from newclid.formulations.rule import Rule
from newclid.proof import ProofState
from newclid.configs import default_defs_path, default_rules_path
from newclid.agent.agents_interface import DeductiveAgent
from newclid.formulations.problem import ProblemJGEX
from newclid.proof_writing import write_proof_steps
import numpy as np

from newclid.statement import Statement
from newclid.tools import atomize
from newclid.webapp import pull_to_server
from newclid.numerical.geometries import PointNum
from newclid.dependencies.dependency import Dependency
from newclid.DDAR.build import DDAR

import time


# Worker function for subprocess isolation (must be at module level for pickling)
def _run_ddar_in_subprocess(
    problem_name, points, premises, goals, max_level, result_queue
):
    """Worker function to run DDAR in a subprocess to isolate memory leaks.

    This function runs in a separate process and puts the result in a queue.
    """
    try:
        solved, dep_graph = DDAR.run_ddar(
            problem_name, points, premises, goals, max_level
        )
        result_queue.put({"success": True, "solved": solved, "dep_graph": dep_graph})
    except Exception as e:
        import traceback

        result_queue.put(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


class GeometricSolver:
    def __init__(
        self, proof: "ProofState", rules: list[Rule], deductive_agent: DeductiveAgent
    ) -> None:
        self.proof = proof
        self.rules = rules
        self.goals = proof.goals
        self.rng = proof.rng
        self.deductive_agent = deductive_agent
        self.run_infos: dict[str, Any] = {}

    def run(self, timeout: int = 3600) -> bool:
        infos = self.deductive_agent.run(
            proof=self.proof, rules=self.rules, timeout=timeout
        )
        self.run_infos = infos
        return infos["success"]

        # infos = run_loop(self.deductive_agent, proof=self.proof, rules=self.rules, timeout=timeout)
        # self.run_infos = infos
        # return infos["success"]

    def write_proof_steps(self, out_file: Optional[Path]):
        write_proof_steps(self.proof, out_file)

    def draw_figure(self, *, out_file: Optional[Path]):
        draw_figure(self.proof, save_to=out_file, rng=self.rng)

    def write_run_infos(self, out_file: Optional[Path]):
        if out_file is None:
            print(self.run_infos)
        else:
            with open(out_file, "w", encoding="utf-8") as f:
                print(self.run_infos, file=f)

    def write_all_outputs(self, out_folder_path: Optional[Path] = None):
        out_folder_path = out_folder_path or self.proof.problem_path
        assert out_folder_path
        out_folder_path.mkdir(exist_ok=True, parents=True)
        self.write_run_infos(out_folder_path / "run_infos.txt")
        self.write_proof_steps(out_folder_path / "proof_steps.txt")
        self.draw_figure(out_file=out_folder_path / "proof_figure.svg")
        pull_to_server(self.proof, server_path=out_folder_path / "html")
        logging.info("Written all outputs at %s", out_folder_path)


class GeometricSolverBuilder:
    def __init__(self, seed: int = None) -> None:
        self.problemJGEX: Optional[ProblemJGEX] = None
        self._defs: Optional[dict[str, DefinitionJGEX]] = None
        self._rules: Optional[list[Rule]] = None
        self.goals: list[Statement] = []
        self.dep_graph = DependencyGraph(AlgebraicManipulator())
        self.deductive_agent: Optional[DeductiveAgent] = None
        self.seed = seed or 998244353
        self.problem_path: Optional[Path] = None

    @property
    def defs(self) -> dict[str, DefinitionJGEX]:
        if self._defs is None:
            self._defs = DefinitionJGEX.to_dict(
                DefinitionJGEX.parse_txt_file(default_defs_path())
            )
        return self._defs

    @property
    def rules(self) -> list[Rule]:
        if self._rules is None:
            self._rules = Rule.parse_txt_file(default_rules_path())
        return self._rules

    def build(self, max_attempts: int = 10000) -> "GeometricSolver":
        if self.problemJGEX:
            logging.debug(
                f"Use problemJGEX {self.problemJGEX} to build the proof state"
            )
            proof_state = ProofState.build_problemJGEX(
                problemJGEX=self.problemJGEX,
                defsJGEX=self.defs,
                problem_path=self.problem_path,
                rng=np.random.default_rng(self.seed),
                max_attempts=max_attempts,
            )
        else:
            logging.info("Use dep_graph to build the proof state")
            proof_state = ProofState(
                rng=np.random.default_rng(self.seed),
                dep_graph=self.dep_graph,
                problem_path=self.problem_path,
                goals=self.goals,
                defs=self.defs,
            )
        if self.deductive_agent is None:
            self.deductive_agent = DDARN()

        if hasattr(self.deductive_agent, "problemJGEX"):
            self.deductive_agent.problemJGEX = self.problemJGEX

        # proof_state.dep_graph.obtain_numerical_checked_eqangle_and_eqratio()
        return GeometricSolver(proof_state, self.rules, self.deductive_agent)

    def load_problem_from_file(
        self, problems_path: Path, problem_name: str, rename: bool = False
    ) -> Self:
        """
        `translate = True` for better LLM training
        """
        self.problemJGEX = ProblemJGEX.from_file(problems_path, problem_name)
        if rename:
            self.problemJGEX = self.problemJGEX.renamed()
        return self

    def load_problem(self, problem: ProblemJGEX) -> Self:
        self.problemJGEX = problem
        return self

    def del_goals(self) -> Self:
        if self.problemJGEX:
            self.problemJGEX = ProblemJGEX(
                self.problemJGEX.name, self.problemJGEX.constructions, ()
            )
        self.goals = []
        return self

    def load_problem_from_txt(self, problem_txt: str) -> Self:
        self.problemJGEX = ProblemJGEX.from_text(problem_txt)
        return self

    def load_rules_from_txt(self, rule_txt: str) -> Self:
        self._rules = Rule.parse_text(rule_txt)
        return self

    def load_rules_from_file(self, rules_path: Optional[Path] = None) -> Self:
        if rules_path is None:
            rules_path = default_rules_path()
        self._rules = Rule.parse_txt_file(rules_path)
        return self

    def load_defs_from_file(self, defs_path: Optional[Path] = None) -> Self:
        if defs_path is None:
            defs_path = default_defs_path()
        self._defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(defs_path))
        return self

    def load_defs_from_txt(self, defs_txt: str) -> Self:
        self._defs = DefinitionJGEX.to_dict(DefinitionJGEX.parse_text(defs_txt))
        return self

    def with_deductive_agent(self, deductive_agent: DeductiveAgent) -> Self:
        self.deductive_agent = deductive_agent
        return self

    def load_geogebra(self, path: Path) -> Self:
        load_geogebra(path, self.dep_graph)
        return self

    def load_goal(self, goal: str) -> Self:
        goal_statement = Statement.from_tokens(atomize(goal), self.dep_graph)
        assert goal_statement, "goal must parse"
        self.goals.append(goal_statement)
        return self

    def load_goals_file(self, path: Path) -> Self:
        for goal in atomize(path.read_text(), "\n"):
            if goal:
                self.load_goal(goal)
        return self

    def with_problem_path(self, path: Path) -> Self:
        self.problem_path = path
        return self


class CSolver:
    def __init__(
        self,
        problem: str = "",
        problem_name: str = "anonymity",
        seed: int = 123,
        solver: GeometricSolver = None,
        config_path: Path = None,
        points: List[Tuple[str, Any, Any]] = None,
        premises: List[Tuple[str, List[str]]] = None,
        goals: List[Tuple[str, List[str]]] = None,
        **overrides,
    ):
        """
        Initialize CSolver with configuration.

        Args:
            problem: Problem string
            problem_name: Name of the problem
            seed: Random seed
            solver: Optional pre-built GeometricSolver
            config_path: Path to config file (default: None, uses default config)
            points: Pre-built points list (lowest priority, used only when no other init path applies)
            premises: Pre-built premises list (used together with points/goals)
            goals: Pre-built goals list (used together with points/premises)
            **overrides: Config overrides (e.g., using_log=True, using_exp=False, light=True)

        Examples:
            # Use default config
            CSolver(fl_statement, solver=solver)

            # Use custom config file
            CSolver(fl_statement, solver=solver, config_path=Path("my_config.json"))

            # Override config
            CSolver(fl_statement, solver=solver, using_log=True)

            # Custom config + override
            CSolver(fl_statement, solver=solver, config_path=Path("my_config.json"), using_log=True)

            # Directly supply pre-built data (lowest priority)
            CSolver(points=pts, premises=prems, goals=gls)
        """
        from newclid.configs import load_solver_config

        # Load config with overrides
        final_config = load_solver_config(config_path, **overrides)

        self.problem = problem
        self.problem_name = problem_name
        self.seed = seed
        self.config = final_config

        self.points: List[Tuple[str, Any, Any]] = []
        self.premises: List[Tuple[str, List[str]]] = []
        self.goals: List[Tuple[str, List[str]]] = []
        self.useful_points: List[str] = []

        if final_config.get("light", False):
            # 轻量化路径：跳过 dep_graph、Matcher、rely 等符号推理开销
            self.solver = None
            problemJGEX = ProblemJGEX.from_text(self.problem)
            _defs = DefinitionJGEX.to_dict(
                DefinitionJGEX.parse_txt_file(default_defs_path())
            )
            rng = np.random.default_rng(self.seed)
            self.points, self.premises, self.goals = build_ddar_input(
                problemJGEX, _defs, rng
            )
            for _, args in self.premises:
                for a in args:
                    if a and a not in self.useful_points:
                        self.useful_points.append(a)
        elif solver is None and problem:
            self.solver = (
                GeometricSolverBuilder(self.seed)
                .load_problem_from_txt(self.problem)
                .build()
            )
            self._extract_premises()
            self._extract_goals()
            self._extract_points()
        elif solver is not None:
            self.solver = solver
            self._extract_premises()
            self._extract_goals()
            self._extract_points()
        elif points is not None and premises is not None and goals is not None:
            # 直接传入预构建数据，跳过所有解析和推理初始化
            self.solver = None
            self.points = list(points) if points is not None else []
            self.premises = list(premises) if premises is not None else []
            self.goals = list(goals) if goals is not None else []
            for _, args in self.premises:
                for a in args:
                    if a and a not in self.useful_points:
                        self.useful_points.append(a)

    # -------------------- 内部方法 -------------------- #
    def _extract_points(self):
        """提取几何点"""
        for name, point in self.solver.proof.symbols_graph.name2node.items():
            if isinstance(point.num, PointNum) and name in self.useful_points:
                self.points.append((name, point.num.x, point.num.y))

    def _extract_premises(self):
        """提取前提"""
        for stmt in self.solver.proof.dep_graph.hyper_graph:
            predicate = stmt.predicate.NAME
            args = []
            for pt in stmt.args:
                if isinstance(pt, Fraction):
                    args.append(str(pt))
                else:
                    args.append(pt.name)
                    if pt.name not in self.useful_points:
                        self.useful_points.append(pt.name)
            self.premises.append((predicate, args))

    def _extract_goals(self):
        """提取目标"""
        for stmt in self.solver.proof.goals:
            predicate = stmt.predicate.NAME
            args = []
            for pt in stmt.args:
                if isinstance(pt, Fraction):
                    args.append(str(pt))
                else:
                    args.append(pt.name)
                    if pt.name not in self.useful_points:
                        self.useful_points.append(pt.name)
            self.goals.append((predicate, args))

    # -------------------- 核心方法 -------------------- #
    def run(
        self,
        max_level: int = 500,
        save_path: str | Path | None = None,
        custom_rules: List[str] = None,
        *,
        goal_checks_only: bool = False,
        goal_proofs_only: bool = False,
    ) -> bool:
        """
        运行 DDAR 并执行求解。
        :param max_level: 最大推理层数
        :param save_path: 可选，保存证明步骤的路径。
        :param goal_checks_only: 仅用于之后检查目标真假、不读取证明的调用。
        :param goal_proofs_only: 导入已证目标的证明，以及判定其余目标所需的候选证明。
        :return: bool 表示是否成功求解。
        """
        t0 = time.time()

        # These checks read the target fact and the state built from premises.
        # DDAR graph import does not update that numerical or algebraic state.
        goal_predicates = {name for name, _ in self.goals}
        supported_goals = {
            "coll",
            "cong",
            "cyclic",
            "eqangle",
            "eqratio",
            "midp",
            "para",
            "perp",
        }
        if goal_checks_only:
            supported_goals.update(
                ("aconst", "rconst", "simtri", "simtrir", "contri", "contrir")
            )
        can_limit_graph = (
            save_path is None
            and bool(goal_predicates)
            and goal_predicates <= supported_goals
        )

        config = dict(
            self.config,
            immutable_dependency_tokens=True,
            export_goal_predicates_only=can_limit_graph and goal_checks_only,
            export_goal_proofs_only=(
                can_limit_graph and goal_proofs_only and not goal_checks_only
            ),
        )

        if custom_rules:
            solved, dep_graph = DDAR.run_ddar_with_custom_theorems(
                self.problem_name,
                self.points,
                self.premises,
                self.goals,
                custom_rules,
                max_level,
                config,
            )
        else:
            solved, dep_graph = DDAR.run_ddar(
                self.problem_name,
                self.points,
                self.premises,
                self.goals,
                max_level,
                config,
            )

        if self.solver is not None:
            proof_graph = self.solver.proof.dep_graph

            @cache
            def parse_statement(tokens):
                return Statement.from_tokens(tokens, proof_graph)

            if can_limit_graph and goal_checks_only:
                for stmt, deps, reason in dep_graph:
                    conclusion = parse_statement(tuple(stmt))
                    for dep in deps:
                        if parse_statement(tuple(dep)) == conclusion:
                            break
                    else:
                        proof_graph.hyper_graph[conclusion] = Dependency(
                            conclusion, reason, ()
                        )
            else:
                for stmt, deps, reason in dep_graph:
                    conclusion = parse_statement(tuple(stmt))
                    why = []
                    for dep in deps:
                        premise = parse_statement(tuple(dep))
                        if premise == conclusion:
                            break
                        why.append(premise)
                    else:
                        proof_graph.hyper_graph[conclusion] = Dependency.mk(
                            conclusion, reason, tuple(why)
                        )

            self.solver.run_infos["success"] = solved
            self.solver.run_infos["runtime"] = time.time() - t0

        # print(self.solver.proof.check_goals())

        if solved:
            # print(
            # f"[CSolver] Problem {self.problem_name} solved successfully ✅")
            if save_path:
                out_path = Path(save_path)
                self.solver.write_proof_steps(out_path)
                # print(f"[CSolver] Proof steps written to {out_path}")

        return solved

    def possible_goals(self) -> List[str]:
        return DDAR.get_possible_goals(
            self.problem_name, self.points, self.premises, self.config
        )

    # -------------------- 辅助输出 -------------------- #
    def print_info(self):
        """打印提取的几何点、前提与目标"""
        print("\n[Points]")
        for p in self.points:
            print(p)
        print("\n[Premises]")
        for pr in self.premises:
            print(pr)
        print("\n[Goals]")
        for g in self.goals:
            print(g)
