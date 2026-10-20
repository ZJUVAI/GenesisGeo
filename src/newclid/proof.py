"""Implements the proof state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from newclid.construction_validation import (
    build_construction_mapping,
    validate_construction_new_points,
)
from newclid.formulations.clause import Clause, translate_sentence
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.dependencies.symbols import Point
from newclid.numerical.draw_figure import init_figure
from newclid.numerical.geometries import (
    InvalidIntersectError,
    InvalidReduceError,
    ObjNum,
    PointNum,
    reduce,
)
from newclid.algebraic_reasoning.algebraic_manipulator import (
    AlgebraicManipulator,
)
from newclid.statement import Statement
from newclid.formulations.definition import DefinitionJGEX
from newclid.match_theorems import Matcher

from newclid.numerical.distances import (
    PointTooCloseError,
    PointTooFarError,
    check_too_far_numerical,
    check_too_close_numerical,
)
from newclid.numerical.sketch import sketch

from newclid.formulations.problem import ProblemJGEX
from newclid.dependencies.dependency import IN_PREMISES, Dependency
from newclid.formulations.rule import Rule
from newclid.tools import atomize, notNone, runtime_cache_path

if TYPE_CHECKING:
    from numpy.random import Generator


class ConstructionError(Exception):
    pass


class ProofState:
    """Object representing the proof state."""

    def __init__(
        self,
        *,
        rng: "Generator",
        dep_graph: Optional[DependencyGraph] = None,
        problem_path: Optional[Path] = None,
        goals: Optional[list[Statement]] = None,
        defs: dict[str, DefinitionJGEX],
    ):
        self.dep_graph = dep_graph or DependencyGraph(AlgebraicManipulator())
        self.symbols_graph = self.dep_graph.symbols_graph
        self.goals: list[Statement] = goals or []
        self.rng = rng

        self.problem_path = problem_path
        self.runtime_cache = runtime_cache_path(self.problem_path)
        self.matcher = Matcher(self.dep_graph, self.runtime_cache, self.rng)

        self.fig = init_figure()
        self.defs = defs

    def add_construction(self, construction: Clause):
        """Add a new clause of construction, e.g. a new excenter."""
        adds: list[Dependency] = []
        numerics: list[tuple[str, ...]] = []
        construction_points: set[Point] = set()
        existing_points = list(self.symbols_graph.nodes_of_type(Point))
        existing_point_names = list(p.name for p in existing_points)
        point_names = validate_construction_new_points(
            construction,
            self.defs,
            existing_point_names,
        )

        for constr_sentence in construction.sentences:
            cdef = self.defs[constr_sentence[0]]
            mapping = build_construction_mapping(construction, constr_sentence, cdef)

            for premise in cdef.require.sentences:
                if len(premise) == 0:
                    continue
                statement = notNone(
                    Statement.from_tokens(
                        translate_sentence(mapping, premise), self.dep_graph
                    )
                )
                if not statement.check_numerical():
                    raise ConstructionError(
                        "Requirement check_numerical failed. " + str(construction)
                    )

            for arg in cdef.args:
                if mapping[arg] in existing_point_names:
                    construction_points.add(self.symbols_graph.name2node[mapping[arg]])

            for bs in cdef.basics:
                for t in bs.sentences:
                    statement = notNone(
                        Statement.from_tokens(
                            translate_sentence(mapping, t), self.dep_graph
                        )
                    )
                    adds.append(Dependency.mk(statement, IN_PREMISES, ()))

            for n in cdef.numerics:
                numerics.append(tuple(mapping[a] if a in mapping else a for a in n))

        fix_point_postions: list[Optional[PointNum]] = []
        for s in construction.points:
            if "@" in s:
                _, pos = atomize(s, "@")
                x, y = atomize(pos, "_")
                fix_point_postions.append(PointNum(x, y))
            else:
                fix_point_postions.append(None)
        new_points = self.symbols_graph.names2points(point_names)
        for p in new_points:
            p.clause = construction

        # draw
        def draw_fn() -> tuple[PointNum, ...]:
            to_be_intersected: list[ObjNum] = []
            for n in numerics:
                args: list[Union[PointNum, str]] = []
                for t in n[1:]:
                    if str.isalpha(t[0]):  # a1 => a
                        args.append(self.symbols_graph.names2points([t])[0].num)
                    else:
                        args.append(t)
                to_be_intersected += sketch(n[0], tuple(args), self.rng)

            return reduce(
                to_be_intersected,
                [p.num for p in existing_points],
                [p.num for p in construction_points],
                rng=self.rng,
            )

        if None in fix_point_postions:
            new_numerical_point = draw_fn()
            for p, num, num0 in zip(
                new_points, new_numerical_point, fix_point_postions
            ):
                p.num = num0 or num
        else:
            for p, num in zip(new_points, fix_point_postions):
                p.num = num
            new_numerical_point = fix_point_postions

        # check two things
        existing_numerical_points = [p.num for p in existing_points]
        if check_too_close_numerical(new_numerical_point, existing_numerical_points):
            raise PointTooCloseError()
        if check_too_far_numerical(new_numerical_point, existing_numerical_points):
            raise PointTooFarError()

        # draw some specific figures (to be refactored, if there are multiple branches)
        # if construction.sentences[0][0] == "triangle":
        #     (ax,) = self.fig.axes
        #     draw_segment(ax, new_points[0], new_points[1])
        #     draw_segment(ax, new_points[1], new_points[2])
        #     draw_segment(ax, new_points[2], new_points[0])
        for add in adds:
            if not add.statement.check_numerical():
                raise ConstructionError(
                    "This is probably because the construction itself is wrong"
                )
            add.add()

        # add rely for new points
        rely_dict: dict[str, list[str]] = {}
        for constr_sentence in construction.sentences:
            cdef = self.defs[constr_sentence[0]]
            if len(constr_sentence) == len(cdef.declare):
                mapping = dict(zip(cdef.declare[1:], constr_sentence[1:]))
            else:
                if len(constr_sentence) + len(construction.points) != len(cdef.declare):
                    raise ValueError("Construction definition length mismatch.")
                mapping = dict(
                    zip(cdef.declare[1:], construction.points + constr_sentence[1:])
                )
            for point, relys in cdef.rely.items():
                point = mapping[point]
                relys = [mapping[p] for p in relys]
                if point not in rely_dict:
                    rely_dict[point] = []
                rely_dict[point].extend(relys)

        for point in new_points:
            if point.name in rely_dict:
                relys = set(rely_dict[point.name])
                point.rely_on = set(self.symbols_graph.names2points(relys))
            else:
                point.rely_on = set()

        self.matcher.update()

    @classmethod
    def build_problemJGEX(
        cls,
        problemJGEX: ProblemJGEX,
        defsJGEX: dict[str, DefinitionJGEX],
        problem_path: Optional[Path],
        max_attempts: int,
        *,
        rng: "Generator",
    ) -> ProofState:
        """Build a problem into a Proof state object."""

        err = ConstructionError(f"Construction failed {max_attempts} times")
        for _ in range(max_attempts):
            # Search for coordinates that checks premises conditions numerically.
            try:
                proof = ProofState(rng=rng, defs=defsJGEX)
                for construction in problemJGEX.constructions:
                    proof.add_construction(construction)
                if problem_path:
                    proof.problem_path = problem_path
                    proof.matcher.update(runtime_cache_path(problem_path))

            except (
                InvalidIntersectError,
                InvalidReduceError,
                ConstructionError,
                PointTooCloseError,
                PointTooFarError,
                ValueError,
            ) as e:
                err = e
                continue

            if not problemJGEX.goals:
                break

            all_check = True
            proof.goals = [
                notNone(Statement.from_tokens(goal, proof.dep_graph))
                for goal in problemJGEX.goals
            ]
            for goal in proof.goals:
                if not goal.check_numerical():
                    all_check = False
                    break
            if all_check:
                break

        else:
            raise Exception(f"Build failed too many times, last error: {repr(err)}")

        return proof

    @classmethod
    def build_predicates(
        cls,
        predicates: list[Statement],
        defsJGEX: dict[str, DefinitionJGEX],
        goals_str: list[str] | None = None,
        *,
        rng: "Generator",
    ) -> ProofState:
        """Build a proof state from given predicates and points."""
        proof = ProofState(rng=rng, defs=defsJGEX)
        adds: list[Dependency] = []

        if not predicates:
            raise ValueError(
                f"Empty predicates list! Cannot build proof state.\n"
                f"goals_str: {goals_str}"
            )

        old_points = predicates[0].dep_graph.symbols_graph.nodes_of_type(Point)
        for old_point in old_points:
            proof.symbols_graph.new_node(Point, old_point.name)

        for statement in predicates:
            new_statement = notNone(
                Statement.from_tokens(
                    tuple(statement.to_str().split()), proof.dep_graph
                )
            )
            adds.append(Dependency.mk(new_statement, IN_PREMISES, ()))

        new_points = proof.symbols_graph.names2points([p.name for p in old_points])
        for p_old, p_new in zip(old_points, new_points):
            p_new.num = p_old.num
            p_new.rely_on = set(
                proof.symbols_graph.names2points([p.name for p in p_old.rely_on])
            )
            p_new.clause = p_old.clause

        for add in adds:
            if not add.statement.check_numerical():
                raise ValueError(
                    "Numerical check failed when building from predicates."
                )
            add.add()

        proof.matcher.update()

        if goals_str:
            proof.goals = [
                notNone(Statement.from_tokens(goal_str.split(), proof.dep_graph))
                for goal_str in goals_str
            ]

        return proof

    def match_theorem(self, theorem: Rule) -> list[Dependency]:
        return list(self.matcher.match_theorem(theorem))

    def apply_dep(self, dep: Dependency) -> bool:
        """Add the dependency to the proof dependency graph.

        Returns:
            True if the statment is a new one, false otherwise.
        """
        if dep.statement in dep.statement.dep_graph.hyper_graph:
            return False
        dep.add()
        return True

    def check_goals(self) -> bool:
        if not self.goals:
            return False
        for goal in self.goals:
            if not goal.check():
                return False
        return True
