from __future__ import annotations

from typing import Optional

from newclid.algebraic_reasoning.algebraic_manipulator import (
    AlgebraicManipulator,
)
from newclid.construction_validation import (
    build_construction_mapping,
    validate_construction_new_points,
)
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.dependencies.symbols import Point
from newclid.formulations.clause import translate_sentence
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.problem import ProblemJGEX
from newclid.numerical.distances import (
    PointTooCloseError,
    PointTooFarError,
    check_too_close_numerical,
    check_too_far_numerical,
)
from newclid.numerical.geometries import (
    InvalidIntersectError,
    InvalidReduceError,
    PointNum,
    reduce as _geo_reduce,
)
from newclid.numerical.sketch import sketch
from newclid.proof import ConstructionError
from newclid.statement import Statement
from newclid.tools import atomize, notNone


def build_ddar_input(
    problemJGEX: ProblemJGEX,
    defs: dict[str, DefinitionJGEX],
    rng,
    max_attempts: int = 10000,
    *,
    only_useful_points: bool = True,
) -> tuple[list, list, list]:
    """Build the numeric DDAR input without constructing a full ProofState."""
    err: Exception = Exception("Build failed")

    for _ in range(max_attempts):
        point_nums: dict[str, PointNum] = {}
        raw_premises: list[tuple[str, list[str]]] = []
        temp_dep_graph = DependencyGraph(AlgebraicManipulator())

        try:
            for construction in problemJGEX.constructions:
                validate_construction_new_points(
                    construction,
                    defs,
                    point_nums.keys(),
                )
                existing_nums = list(point_nums.values())
                construction_arg_nums: list[PointNum] = []
                numerics: list[tuple[str, ...]] = []

                for constr_sentence in construction.sentences:
                    cdef = defs[constr_sentence[0]]
                    mapping = build_construction_mapping(
                        construction, constr_sentence, cdef
                    )

                    for premise in cdef.require.sentences:
                        if len(premise) == 0:
                            continue
                        statement = notNone(
                            Statement.from_tokens(
                                translate_sentence(mapping, premise),
                                temp_dep_graph,
                            )
                        )
                        if not statement.check_numerical():
                            raise ConstructionError(
                                "Requirement check_numerical failed. "
                                + str(construction)
                            )

                    for arg in cdef.args:
                        name = mapping[arg]
                        if name in point_nums:
                            num = point_nums[name]
                            if num not in construction_arg_nums:
                                construction_arg_nums.append(num)

                    for bs in cdef.basics:
                        for t in bs.sentences:
                            translated = translate_sentence(mapping, t)
                            if translated:
                                raw_premises.append(
                                    (translated[0], list(translated[1:]))
                                )

                    for n in cdef.numerics:
                        numerics.append(tuple(mapping.get(a, a) for a in n))

                point_names: list[str] = []
                fix_positions: list[Optional[PointNum]] = []
                for s in construction.points:
                    if "@" in s:
                        name, pos = atomize(s, "@")
                        x, y = atomize(pos, "_")
                        point_names.append(name)
                        fix_positions.append(PointNum(x, y))
                    else:
                        point_names.append(s)
                        fix_positions.append(None)

                if None in fix_positions:
                    to_be_intersected = []
                    for n in numerics:
                        args: list = []
                        for t in n[1:]:
                            if t and t[0].isalpha():
                                args.append(point_nums[t])
                            else:
                                args.append(t)
                        to_be_intersected += sketch(n[0], tuple(args), rng)
                    new_nums = _geo_reduce(
                        to_be_intersected, existing_nums, construction_arg_nums, rng=rng
                    )
                    for name, num, fixed in zip(point_names, new_nums, fix_positions):
                        point_nums[name] = fixed if fixed is not None else num
                else:
                    new_nums = list(fix_positions)
                    for name, num in zip(point_names, fix_positions):
                        point_nums[name] = num

                if check_too_close_numerical(new_nums, existing_nums):
                    raise PointTooCloseError()
                if check_too_far_numerical(new_nums, existing_nums):
                    raise PointTooFarError()

                for name in point_names:
                    pt = temp_dep_graph.symbols_graph.new_node(Point, name, None)
                    pt.num = point_nums[name]

            raw_goals: list[tuple[str, list[str]]] = []
            useful_points: set[str] = set()
            for goal_tokens in problemJGEX.goals:
                raw_goals.append((goal_tokens[0], list(goal_tokens[1:])))
                for t in goal_tokens[1:]:
                    if t and t[0].isalpha():
                        useful_points.add(t)
            for _, args in raw_premises:
                for a in args:
                    if a and a[0].isalpha():
                        useful_points.add(a)

            for goal_tokens in problemJGEX.goals:
                goal_stmt = Statement.from_tokens(goal_tokens, temp_dep_graph)
                if goal_stmt is None:
                    raise ValueError(f"Failed to parse goal: {goal_tokens}")
                if not goal_stmt.check_numerical():
                    raise ValueError(f"Goal {goal_stmt.pretty()} fails numerical check")

            point_names = useful_points if only_useful_points else point_nums.keys()
            points = [
                (name, num.x, num.y)
                for name, num in point_nums.items()
                if name in point_names
            ]
            return points, raw_premises, raw_goals

        except (
            InvalidIntersectError,
            InvalidReduceError,
            PointTooCloseError,
            PointTooFarError,
            ValueError,
            KeyError,
            AssertionError,
            ConstructionError,
        ) as e:
            err = e
            continue

    raise Exception(f"Build failed too many times, last error: {repr(err)}")
