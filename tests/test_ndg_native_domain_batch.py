from newclid.api import DDAR


def test_native_domain_batch_matches_angle_based_parallel_semantics():
    points = [("a", 0.0, 0.0), ("b", 0.0, 1.0), ("c", 1.0, 0.0), ("d", 1.00001, 1.0)]
    statements = [
        ("para", list("abcd")),
        ("npara", list("abcd")),
        ("nperp", list("abcd")),
        ("diff", list("aa")),
        ("diff", ["a", "missing"]),
    ]
    assert DDAR.check_numerical_statements(points, statements) == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_native_domain_batch_fails_closed_for_invalid_geometry():
    points = [("a", 0.0, 0.0), ("b", 1.0, 0.0), ("c", 0.0, 1.0)]
    statements = [
        ("ncoll", list("abc")),
        ("diff", ["a"]),
        ("angle_defined", list("abababab")),
        ("ncoll", ["a", "b", "missing"]),
    ]
    assert DDAR.check_numerical_statements(points, statements) == [
        True,
        False,
        False,
        False,
    ]
    assert DDAR.check_numerical_statements(
        [("a", float("nan"), 0.0), ("b", 1.0, 0.0)], [("diff", list("ab"))]
    ) == [False]
