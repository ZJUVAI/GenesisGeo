from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Optional
from numpy.random import Generator
from fractions import Fraction
from newclid.tools import fraction_to_angle, fraction_to_ratio

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from newclid.dependencies.dependency import Dependency
    from newclid.dependencies.dependency_graph import DependencyGraph
    from newclid.statement import Statement



class Predicate(ABC):
    """
    When the args are passed in functions other than parse and to_tokens,
    the orders are guaranteed to be canonique.
    """

    NAME: str

    @classmethod
    def preparse(cls, args: tuple[str, ...]) -> Optional[tuple[str, ...]]:
        raise NotImplementedError(f"{cls.NAME} preparse not implemented")

    @classmethod
    def parse(
        cls, args: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[tuple[Any, ...]]:
        raise NotImplementedError(f"{cls.NAME} parse not implemented")

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        raise NotImplementedError(f"{cls.NAME} check_numerical not implemented")

    @classmethod
    def check(cls, statement: Statement) -> bool:
        """
        Hypothesis : the numercial test is passed
        """
        return False

    @classmethod
    def add(cls, dep: Dependency) -> None:
        return

    @classmethod
    def why(cls, statement: Statement) -> Optional[Dependency]:
        """
        Hypothesis : the numercial test is passed
        This function should only be giving one same dependency, which is the implicit dependency used in the first check success.
        """
        return None

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        """Write the predicate in a natural language."""
        return cls.to_repr(statement)

    @classmethod
    def to_repr(cls, statement: Statement) -> str:
        res = cls.NAME + "["
        for a in statement.args:
            res += repr(a) + ","
        res += "]"
        return res
    @classmethod
    def to_str(cls, statement: Statement) -> str:
        res = cls.NAME
        for a in statement.args:
            if isinstance(a, Fraction):
                # res += " " + str(float(a))
                if cls.NAME == "aconst":
                    res += " " + fraction_to_angle(a)
                if cls.NAME == "rconst":
                    res += " " + fraction_to_ratio(a)
            else:
                res += " " + repr(a)
        return res

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        raise NotImplementedError(f"{cls.NAME} to_tokens not implemented")

    @classmethod
    def to_constructive(cls, point: str, args: tuple[str, ...]) -> str:
        raise NotImplementedError

    @classmethod
    def draw(
        cls, ax: Axes, args: tuple[Any, ...], dep_graph: DependencyGraph, rng: Generator
    ):
        ...
