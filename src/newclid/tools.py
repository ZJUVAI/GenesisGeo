from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional, Sequence, TypeVar

from newclid.numerical import close_enough

if TYPE_CHECKING:
    from pyvis.network import Network  # type: ignore
    from newclid.statement import Statement


class InfQuotientError(Exception):
    pass


# maximum denominator for a fraction.
MAX_DENOMINATOR = 1000000

T = TypeVar("T")


def notNone(x: Optional[T]) -> T:
    assert x is not None
    return x


def get_quotient(v: Any) -> Fraction:
    v = float(v)
    if abs(v) < 1e-6:
        v = float(0)
    n = v
    d = 1
    while not close_enough(n, round(n)):
        d += 1
        n += v
        if d > MAX_DENOMINATOR:
            e = InfQuotientError(v)
            raise e

    n = int(round(n))
    return Fraction(n, d)


def atomize(s: str, split_by: Optional[str] = None) -> tuple[str, ...]:
    words = s.split(split_by)
    return tuple(word.strip() for word in words)


def str_to_fraction(s: str) -> Fraction:
    if "pi/" in s:
        ns, ds = s.split("pi/")
        n, d = int(ns), int(ds)
        if d < 0:
            n, d = -n, -d
        return Fraction(n % d, d)
    elif "o" in s:
        n = int(s[:-1])
        d = 180
        return Fraction(n % d, d)
    elif "/" in s:
        ns, ds = s.split("/")
        n, d = int(ns), int(ds)
        return Fraction(n, d)
    else:
        n = int(s)
        d = 1
        return Fraction(n, d)


def fraction_to_len(f: Fraction):
    return f"{f.numerator}/{f.denominator}"


def fraction_to_ratio(f: Fraction):
    return f"{f.numerator}/{f.denominator}"


def fraction_to_angle(f: Fraction):
    n, d = f.numerator, f.denominator
    return f"{n % d}pi/{d}"


def reshape(to_reshape: Sequence[T], n: int) -> Generator[tuple[T, ...], None, None]:
    for i in range(0, len(to_reshape), n):
        yield tuple(to_reshape[i + k] for k in range(n))


def add_edge(net: "Network", u: Any, v: Any):
    net.add_node(u)  # type: ignore
    net.add_node(v)  # type: ignore
    net.add_edge(u, v)  # type: ignore


def runtime_cache_path(problem_path: Optional[Path]):
    return problem_path / "runtime_cache.json" if problem_path else None


def run_static_server(directory_to_serve: Path):
    print(f"command to run the server: python -m http.server -d {directory_to_serve}")


def boring_statement(statement: "Statement"):
    s = statement.pretty()
    if "=" in s:
        splited = atomize(s, "=")
        return all(t == splited[0] for t in splited)
    if "≅" in s:
        a, b = atomize(s, "≅")
        return a == b
    if "are sameclock to" in s:
        a, b = atomize(s, "are sameclock to")
        return a == b
    return False
