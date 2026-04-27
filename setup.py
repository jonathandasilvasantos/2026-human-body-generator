from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent
CORE = ROOT / "core"

core_sources = [
    "core/hmath.c",
    "core/skeleton.c",
    "core/morph.c",
    "core/skin.c",
    "core/proto.c",
    "core/io.c",
    "core/human.c",
]

ext = Extension(
    name="bindings.human",
    sources=["bindings/human.pyx", *core_sources],
    include_dirs=[str(CORE), np.get_include()],
    libraries=["m"],
    extra_compile_args=["-O2", "-Wall", "-std=c99"],
)

setup(ext_modules=cythonize([ext], language_level=3))
