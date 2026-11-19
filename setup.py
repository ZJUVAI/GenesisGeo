import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent
DDAR = "newclid.DDAR.build.DDAR"
TRAINING_INPUTS = ROOT / "benchmarks" / "ddar-pgo-inputs.jsonl.gz"


class DDARBuildCommand(build_ext):
    def build_extension(self, ext):
        if ext.name != DDAR:
            return super().build_extension(ext)

        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        pybind11_cmake = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
        ).strip()
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="ddar-", dir=build_temp) as tmp:
            root = Path(tmp)
            if os.environ.get("PGO") != "0":
                try:
                    artifact = self._build_pgo(
                        root / "pgo", pybind11_cmake, output.name
                    )
                except (
                    EOFError,
                    OSError,
                    RuntimeError,
                    subprocess.CalledProcessError,
                    ValueError,
                ) as error:
                    print(
                        f"DDAR PGO failed, building ordinary Release: {error}",
                        flush=True,
                    )
                else:
                    self._install_ddar(artifact, output)
                    return
            artifact = self._build_ddar(root / "release", pybind11_cmake, output.name)
            self._install_ddar(artifact, output)

    @staticmethod
    def _install_ddar(artifact, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, output)

    def _build_ddar(self, build, pybind11_cmake, filename, mode="OFF"):
        library = build / "module"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(ROOT / "src/newclid/DDAR"),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-Dpybind11_DIR={pybind11_cmake}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={library}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={library}",
                f"-DDDAR_PGO={mode}",
            ],
            check=True,
        )
        command = ["cmake", "--build", str(build), "--config", "Release"]
        if not os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"):
            command += ["--parallel", str(min(8, os.cpu_count() or 1))]
        subprocess.run(command, check=True)
        artifact = library / filename
        if not artifact.is_file():
            raise RuntimeError(f"CMake did not produce {artifact}")
        return artifact

    def _build_pgo(self, build, pybind11_cmake, filename):
        with gzip.open(TRAINING_INPUTS, "rt", encoding="utf-8") as stream:
            inputs = [json.loads(line) for line in stream if line.strip()]
        if not inputs:
            raise RuntimeError("DDAR profiling workload is empty")
        artifact = self._build_ddar(build, pybind11_cmake, filename, "GENERATE")
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("DDAR", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for args in json.load(sys.stdin):
    module.run_ddar(*args)
""",
                str(artifact),
            ],
            input=json.dumps(inputs),
            text=True,
            check=True,
        )
        if not any((build / "profile").rglob("*.gcda")):
            raise RuntimeError("DDAR profiling produced no counters")
        artifact.unlink()
        return self._build_ddar(build, pybind11_cmake, filename, "USE")


ext_modules = [
    Extension(DDAR, sources=[]),
    Pybind11Extension(
        "newclid.dependencies.geometry",
        ["src/newclid/dependencies/geometry.cpp"],
        extra_compile_args=["-O3", "-std=c++14"],
    ),
    Pybind11Extension(
        "newclid.matchinC",
        ["src/newclid/matchinC.cpp"],
        extra_compile_args=["-O3", "-std=c++14"],
    ),
    Pybind11Extension(
        "newclid.generation.auxiliary",
        [
            "src/newclid/generation/auxiliary/geometry.cpp",
            "src/newclid/generation/auxiliary/line.cpp",
            "src/newclid/generation/auxiliary/circle.cpp",
            "src/newclid/generation/auxiliary/utils.cpp",
            "src/newclid/generation/auxiliary/pybind.cpp",
        ],
        extra_compile_args=["-O3", "-std=c++17", "-march=native"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": DDARBuildCommand},
)
