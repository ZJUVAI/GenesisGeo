"""Build a source-based predicate probe independently of optional pybind APIs."""

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


SOURCE = r"""
// Test-only access to the native predicate contract, without Python bindings.
#include "problem.hpp"
#include <iostream>
#include <string>
#include <vector>

int main()
{
    size_t cases;
    std::cin >> cases;
    for (size_t c = 0; c < cases; ++c)
    {
        Problem problem;
        size_t points, statements;
        std::cin >> points >> statements;
        for (size_t i = 0; i < points; ++i)
        {
            std::string name;
            double x, y;
            std::cin >> name >> x >> y;
            problem.add_point(name, x, y);
        }
        for (size_t i = 0; i < statements; ++i)
        {
            std::string kind;
            size_t count;
            std::cin >> kind >> count;
            std::vector<std::string> args(count);
            for (auto &arg : args) std::cin >> arg;
            bool valid = false;
            try
            {
                if (kind == "diff")
                    valid = count == 2 && !problem.find_point(args[0]).is_close(problem.find_point(args[1]));
                else
                {
                    const auto statement = problem.create_statement(kind, args);
                    valid = statement && statement->check_numerically();
                }
            }
            catch (const std::exception &) {}
            std::cout << valid << '\n';
        }
    }
}
"""


@lru_cache(maxsize=1)
def executable():
    root = Path(__file__).resolve().parents[1]
    native = Path(
        os.environ.get("GENESIS_NATIVE_SOURCE_DIR", str(root / "src/newclid/DDAR"))
    )
    sources = sorted(
        p
        for p in native.rglob("*.cpp")
        if p.name != "bindings.cpp" and "build" not in p.relative_to(native).parts
    )
    digest = hashlib.sha256()
    for path in sources + sorted(native.rglob("*.hpp")):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(SOURCE.encode())
    folder = root / ".pytest_cache/native-predicate-probe" / digest.hexdigest()[:20]
    binary = folder / "build/probe"
    if not binary.exists():
        if shutil.which("cmake") is None:
            raise RuntimeError(
                "Native predicate regression tests require CMake and a C++ compiler"
            )
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / "probe.cpp"
        probe.write_text(SOURCE)

        def quote(path):
            return '"' + str(path).replace("\\", "/") + '"'

        (folder / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.14)\nproject(PredicateProbe LANGUAGES CXX)\n"
            "set(CMAKE_CXX_STANDARD 20)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
            "add_executable(probe "
            + " ".join(quote(p) for p in sources + [probe])
            + ")\n"
            "target_include_directories(probe PRIVATE " + quote(native) + ")\n"
        )
        for command in (
            ["cmake", "-S", str(folder), "-B", str(folder / "build")],
            ["cmake", "--build", str(folder / "build"), "--parallel", "2"],
        ):
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=240
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
    return binary


def query(cases):
    lines = [str(len(cases))]
    for case in cases:
        lines.append(f"{len(case['points'])} {len(case['statements'])}")
        lines.extend(f"{name} {x:.17g} {y:.17g}" for name, x, y in case["points"])
        lines.extend(
            " ".join([kind, str(len(args)), *args]) for kind, args in case["statements"]
        )
    result = subprocess.run(
        [str(executable())],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    values = iter(result.stdout.split())
    output = [[next(values) == "1" for _ in case["statements"]] for case in cases]
    assert next(values, None) is None
    return output
