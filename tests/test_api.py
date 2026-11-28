"""Tests for CSolver light mode and GeometricSolverBuilder API."""

from newclid.api import CSolver, GeometricSolverBuilder


class TestCSolverLightMode:
    def test_custom_rule_can_require_unequal_angles(self):
        problem = (
            "a b c = triangle a b c; o = circle o a b c; d = on_line d a b; "
            "e = on_line e a c, on_circle e a d; "
            "f = on_bline f b d, on_circle f o a; "
            "g = on_bline g e c, on_circle g o a ? para d e f g"
        )
        rule = (
            "equal_projection_reflected_chord|coll a b d,coll a c e,cong a d a e,"
            "cong f b f d,cong g c g e,cong o a o b,cong o a o c,cong o f o g"
            "|para d e f g|ncoll a b c,ncoll a f g,neqangle a d o f a e o g"
        )
        assert CSolver(problem, seed=123, light=True).run(custom_rules=[rule])

    def test_light_mode_extracts_points(self):
        solver = CSolver(
            problem="a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c",
            problem_name="test",
            seed=123,
            light=True,
        )
        assert len(solver.points) > 0

    def test_light_mode_extracts_premises(self):
        solver = CSolver(
            problem="a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c",
            problem_name="test",
            seed=123,
            light=True,
        )
        assert len(solver.premises) > 0

    def test_light_mode_extracts_goals(self):
        solver = CSolver(
            problem="a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c",
            problem_name="test",
            seed=123,
            light=True,
        )
        assert len(solver.goals) > 0
        assert solver.goals[0][0] == "perp"

    def test_possible_goals(self):
        solver = CSolver(
            problem="a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c",
            problem_name="test",
            seed=123,
            light=True,
        )
        goals = solver.possible_goals()
        assert isinstance(goals, list)

    def test_custom_rule_can_conclude_a_constant_ratio(self):
        problem = (
            "a b c = triangle a b c; o = circumcenter o a b c; "
            "x y z g = centroid x y z g a b c; h = orthocenter h a b c "
            "? rconst o g o h 1/3"
        )
        rule = (
            "euler_line_ratio|cong o a o b,cong o b o c,"
            "perp h a b c,perp h b c a,midp x b c,midp y a c,"
            "coll a x g,coll b y g|rconst o g o h 1/3|ncoll a b c"
        )
        assert CSolver(problem, seed=123, light=True).run(custom_rules=[rule])

    def test_custom_rule_checks_angle_after_binding_its_points(self):
        problem = (
            "a b d = triangle a b d; e = midpoint e d a; "
            "c = angle_bisector c a b d, on_dia c a b ? para c e b d"
        )
        rule = (
            "bisector_projection|midp e a d,eqangle a b b c b c b d,"
            "perp a c b c|para c e b d"
        )
        assert CSolver(problem, seed=123, light=True).run(custom_rules=[rule])

    def test_custom_rule_checks_ratio_after_binding_its_points(self):
        problem = (
            "a b c = triangle a b c; "
            "d = angle_bisector d c b a, on_line d a c; "
            "e = angle_bisector e a c b, on_line e a b; "
            "f = on_line f d e, on_line f b c ? eqangle a b a f a f a c"
        )
        rule = (
            "intersecting_ratio_cevians|coll a c d,coll a b e,coll d e f,"
            "coll b c f,eqratio a d d c a b b c,eqratio a e e b a c b c"
            "|eqangle a b a f a f a c|ncoll a b c"
        )
        assert CSolver(problem, seed=123, light=True).run(custom_rules=[rule])


class TestGeometricSolverBuilder:
    def test_chained_api(self):
        solver = (
            GeometricSolverBuilder(seed=123)
            .load_problem_from_txt(
                "a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c"
            )
            .build()
        )
        assert solver.run()

    def test_builder_returns_self(self):
        builder = GeometricSolverBuilder(seed=123)
        result = builder.load_problem_from_txt("a b c = triangle a b c")
        assert result is builder

    def test_builder_with_rules(self):
        solver = (
            GeometricSolverBuilder(seed=123)
            .load_problem_from_txt(
                "a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c"
            )
            .load_rules_from_txt("")
            .build()
        )
        assert solver is not None

    def test_multiple_builds_different_seeds(self):
        for seed in [1, 42, 999]:
            solver = (
                GeometricSolverBuilder(seed=seed)
                .load_problem_from_txt("a b c = triangle a b c")
                .build()
            )
            assert solver is not None
