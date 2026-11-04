# -*- coding: utf-8 -*-
"""
Build script for the C++ helper library.  Package metadata lives in
pyproject.toml; this file only describes the ctypes extension, which is a
plain shared library (no Python C API) loaded by numpy.ctypeslib.
"""
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

EXPORTED = ["calc_sdwgw", "calc_sdlwgw", "calc_lwgw", "calc_ace", "calc_ace_scores", "calc_rsr", "calc_race", "delete_doubleptr"]

import sys
import sysconfig


def _target_is_emscripten():
    """True when cross compiling for Pyodide (the host python is still linux)."""
    platform = os.environ.get('_PYTHON_HOST_PLATFORM', '') or sysconfig.get_platform()
    return 'emscripten' in platform or 'wasm' in platform or sys.platform == 'emscripten'


libs = []
compile_args = []
link_args = []
if os.name == 'posix':
    libs.append('m')
    compile_args = ['-O3', '-std=c++11']
    if not sys.platform.startswith('darwin') and not _target_is_emscripten():
        # std::thread needs -pthread on Linux; Pyodide (emscripten) has no
        # threads and must not be linked with it
        compile_args.append('-pthread')
        link_args.append('-pthread')


class win_build_ext(build_ext):
    """On Windows build a plain .dll instead of a .pyd and export the C API."""

    def get_export_symbols(self, ext):
        return ext.export_symbols + EXPORTED

    def get_ext_filename(self, ext_name):
        p = ext_name.split(".")
        if len(p) == 1:
            return ext_name + '.dll'
        else:
            return os.path.join(*p[:-1], p[-1] + '.dll')


def get_build_ext():
    if os.name == 'posix':
        return build_ext
    else:
        return win_build_ext


setup(
    ext_modules=[
        Extension(
            'colorcorrect._cutil',
            ['cutil/cutil.cpp'],
            include_dirs=['cutil'],
            libraries=libs,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ],
    cmdclass={"build_ext": get_build_ext()},
)
