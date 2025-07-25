from __future__ import annotations
from typing import NamedTuple

from newclid.tools import atomize


class Clause(NamedTuple):
    points: tuple[str, ...]
    sentences: tuple[tuple[str, ...], ...]

    @classmethod
    def parse_line(cls, s: str) -> tuple[Clause, ...]:
        levels = atomize(s, ";")

        basics: list[Clause] = []
        for level in levels:
            points: tuple[str, ...] = tuple()
            points_str, bs_str = (
                level.split(":")
                if ":" in level
                else level.split("=")
                if "=" in level
                else ("", level)
            )
            points = tuple(points_str.strip().split())
            bs_str = bs_str.strip()
            bs = tuple(atomize(b) for b in bs_str.split(",") if b.strip() != "")
            basics.append(Clause(points, bs))

        return tuple(basics)

    def renamed(self, mp: dict[str, str]) -> Clause:
        return Clause(
            # tuple(mp[p] if p in mp else p for p in self.points),
            tuple(mp[p.split('@')[0]] + '@' + p.split('@')[1] if '@' in p and p.split('@')[0] in mp else mp[p] if p in mp else p for p in self.points),
            tuple(translate_sentence(mp, s) for s in self.sentences),
        )

    def __str__(self) -> str:
        return f"{' '.join(p for p in self.points)} = {', '.join(' '.join(s) for s in self.sentences)}"


def translate_sentence(
    mapping: dict[str, str], sentence: tuple[str, ...]
) -> tuple[str, ...]:
    return (sentence[0],) + tuple(
        mapping[a] if a in mapping else a for a in sentence[1:]
    )
