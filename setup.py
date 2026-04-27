from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent
CORE = ROOT / "core"

ext = Extension(
    name="bindings.human",
    sources=[
        "bindings/human.pyx",
        "core/human.c",
    ],
    include_dirs=[str(CORE)],
    libraries=["m"],
    extra_compile_args=["-O2", "-Wall", "-Wextra", "-std=c99"],
)

setup(ext_modules=cythonize([ext], language_level=3))
