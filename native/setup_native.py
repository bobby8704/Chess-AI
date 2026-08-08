"""Build the native kernel in place.

    .venv/Scripts/python.exe native/setup_native.py build_ext --inplace

Produces chesskernel.*.pyd (Windows) / .so (Linux) next to this file. The
extension is OPTIONAL at runtime — consumers import it inside a try/except and
fall back to pure Python, so nothing else in the repo requires a compiler.
"""

import os
import sys

from setuptools import Extension, setup

import pybind11

_HERE = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    cflags = ["/O2", "/std:c++17", "/EHsc"]
else:
    cflags = ["-O2", "-std=c++17"]

setup(
    name="chesskernel",
    ext_modules=[
        Extension(
            "chesskernel",
            sources=[os.path.join(_HERE, "chesskernel.cpp")],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=cflags,
            language="c++",
        )
    ],
    script_args=sys.argv[1:] or ["build_ext", "--inplace"],
)
