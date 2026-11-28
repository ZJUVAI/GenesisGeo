"""DDAR rule representation used by compression passes."""

from typing import NamedTuple


class Rule(NamedTuple):
    name: str
    premises: tuple[str, ...]
    conclusions: tuple[str, ...]
    guards: tuple[str, ...]

    def pipe(self) -> str:
        fields = [self.name, ", ".join(self.premises), ", ".join(self.conclusions)]
        if self.guards:
            fields.append(", ".join(self.guards))
        return "|".join(fields)


def split(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())
