import pytest
from newclid.api import DDAR


def config():
    settings = {f"r{i:02d}": False for i in range(1000)}
    settings.update(
        using_ar=False,
        targeted_ar=False,
        using_log=False,
        using_exp=False,
        verbose=False,
        allow_duplicate_points=False,
    )
    return settings


@pytest.mark.parametrize(
    "guard,expected",
    [
        ("diff a c", True),
        ("diff a a", False),
        ("diff a missing", False),
        ("diff a", False),
        ("diff a b c", False),
        ("angle_defined a b c d a b c d", True),
        ("angle_defined a b a b a b a b", False),
        ("angle_defined a b c d", False),
        ("neqangle a b a c a b b c", True),
        ("neqangle a b a c a b a c", False),
    ],
)
def test_native_guard_dispatch_is_fail_closed(guard, expected):
    points = [("a", 0.0, 0.0), ("b", 1.0, 0.0), ("c", 0.0, 1.0), ("d", 0.0, 2.0)]
    premises = [("perp", list("abcd"))]
    goals = [("cong", list("abcd"))]
    baseline, _ = DDAR.run_ddar("base", points, premises, goals, 3, config())
    assert not baseline
    rule = "guard_test|perp a b c d|cong a b c d|" + guard
    solved, _ = DDAR.run_ddar_with_custom_theorems(
        "guarded", points, premises, goals, [rule], 3, config()
    )
    assert solved == expected
