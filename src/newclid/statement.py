from __future__ import annotations
import logging
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

    _fields = ("predicate", "args", "dep_graph", "_hash", "_representation", "_text")
    __slots__ = (*_fields, "__dict__", "__weakref__")

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
        self._representation: Optional[str] = None
        self._text: Optional[str] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update((field, getattr(self, field)) for field in self._fields)
        return state

    def __setstate__(self, state):
        self._hash = self._representation = self._text = None
        for field, value in state.items():
            setattr(self, field, value)

    def check(self, *, explain: bool = True) -> bool:
        """Symbolically check if the statement is currently considered True."""
        if self in self.dep_graph.hyper_graph:
            return True
        if not self.check_numerical():
            return False
        if self.predicate.check(self):
            if explain:
                self.why()
            return True
        return False

    def check_numerical(self) -> bool:
        """Check if the statement is numerically sound."""
        checked = self.dep_graph.check_numerical
        res = checked.get(self)
        if res is None:
            res = self.predicate.check_numerical(self)
            checked[self] = res
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
        text = self._text
        if text is None:
            text = self.predicate.to_str(self)
            self._text = text
        return text

    def __repr__(self) -> str:
        representation = self._representation
        if representation is None:
            representation = self.predicate.to_repr(self)
            self._representation = representation
        return representation

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, obj: object) -> bool:
        return self is obj or (isinstance(obj, Statement) and hash(self) == hash(obj))

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

        parsed = pred.parse_preparsed(preparsed, dep_graph)
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

    def draw(self, ax: "Axes", rng: Generator, draw_annotations: bool = True):
        try:
            self.predicate.draw(
                ax=ax,
                args=self.args,
                dep_graph=self.dep_graph,
                rng=rng,
                draw_annotations=draw_annotations,
            )
        except Exception as e:
            logging.warning(f"Failed to draw statement {self}: {e}")
