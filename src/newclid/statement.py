from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

from newclid.predicates import NAME_TO_PREDICATE
from newclid.dependencies.dependency import Dependency
from numpy.random import Generator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from newclid.predicates.predicate import Predicate
    from newclid.dependencies.dependency_graph import DependencyGraph


class Statement:
    """One predicate applied to a set of points and values. Comes with a proof that args are well ordered"""

    def __init__(
        self,
        predicate: type[Predicate],
        args: tuple[Any, ...],
        dep_graph: DependencyGraph,
    ) -> None:
        self.predicate = predicate
        self.args: tuple[Any, ...] = args
        self.dep_graph = dep_graph
        self._hash = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_hash' not in self.__dict__:
            self._hash = None

    def check(self) -> bool:
        """Symbolically check if the statement is currently considered True."""
        if self in self.dep_graph.hyper_graph:
            return True
        if not self.check_numerical():
            return False
        if self.predicate.check(self):
            self.why()
            return True
        return False

    def check_numerical(self) -> bool:
        """Check if the statement is numerically sound."""
        if self in self.dep_graph.check_numerical:
            return self.dep_graph.check_numerical[self]
        res = self.predicate.check_numerical(self)
        self.dep_graph.check_numerical[self] = res
        return res

    def why(self) -> Optional[Dependency]:
        res = self.dep_graph.hyper_graph.get(self)
        if res is not None:
            return res
        res = self.predicate.why(self)
        if res is not None:
            self.dep_graph.hyper_graph[self] = res
        return res
    
    def to_str(self) -> str:
        return self.predicate.to_str(self)

    def __repr__(self) -> str:
        return self.predicate.to_repr(self)

    def __hash__(self) -> int:
        if not hasattr(self, '_hash') or self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, obj: object) -> bool:
        return isinstance(obj, Statement) and hash(self) == hash(obj)

    @classmethod
    def from_tokens(
        cls, tokens: tuple[str, ...], dep_graph: DependencyGraph
    ) -> Optional[Statement]:
        # preparse, reorder args
        pred = NAME_TO_PREDICATE[tokens[0]]
        preparsed = pred.preparse(tokens[1:])
        if not preparsed:
            return None
        tokens = (tokens[0],) + preparsed

        if tokens in dep_graph.token_statement:
            return dep_graph.token_statement[tokens]

        # pred = NAME_TO_PREDICATE[tokens[0]]
        parsed = pred.parse(tokens[1:], dep_graph)
        if not parsed:
            dep_graph.token_statement[tokens] = None
            return None
        s = Statement(pred, parsed, dep_graph)
        dep_graph.token_statement[tokens] = s
        return s

    def pretty(self) -> str:
        return self.predicate.pretty(self)

    def with_new(
        self,
        new_predicate: Optional[type[Predicate]],
        new_args: Optional[tuple[Any, ...]],
    ):
        predicate = new_predicate or self.predicate
        args = new_args or self.args
        newst = self.from_tokens(
            (predicate.NAME,) + predicate.to_tokens(args), self.dep_graph
        )
        assert newst
        return newst

    def draw(self, ax: "Axes", rng: Generator):
        self.predicate.draw(ax, self.args, self.dep_graph, rng)
