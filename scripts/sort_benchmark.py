"""Order a two-line JGEX problem list by construction point count."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

from newclid.formulations.problem import ProblemJGEX


def sort_problems(source: Path, output: Path) -> int:
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) % 2:
        raise ValueError("Expected a problem name and statement for each problem")
    records = list(zip(lines[::2], lines[1::2], strict=True))
    if len({name for name, _ in records}) != len(records):
        raise ValueError("Problem names must be unique")

    def point_count(record: tuple[str, str]) -> int:
        problem = ProblemJGEX.from_text(record[1])
        return sum(len(clause.points) for clause in problem.constructions)

    records.sort(key=point_count, reverse=True)
    stream = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.writelines(f"{name}\n{statement}\n" for name, statement in records)
        if output.exists():
            temporary.chmod(output.stat().st_mode & 0o777)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input benchmark text file")
    parser.add_argument("output", type=Path, help="Sorted benchmark text file")
    args = parser.parse_args()
    try:
        count = sort_problems(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Sorted {count} problems into {args.output}")


if __name__ == "__main__":
    main()
