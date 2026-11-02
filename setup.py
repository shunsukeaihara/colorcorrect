# -*- coding: utf-8 -*-
"""
Build script for the C++ helper library.  Package metadata lives in
pyproject.toml; this file only describes the ctypes extension, which is a
plain shared library (no Python C API) loaded by numpy.ctypeslib.
"""
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

EXPORTED = ["calc_sdwgw", "calc_sdlwgw", "calc_lwgw", "calc_ace", "delete_doubleptr"]

libs = []
compile_args = []
if os.name == 'posix':
    libs.append('m')
    compile_args = ['-O3', '-std=c++11']


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
        ),
    ],
    cmdclass={"build_ext": get_build_ext()},
)
