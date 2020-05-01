# -*- coding: utf-8 -*-
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
sys.path.append('./src')
sys.path.append('./test')
version = open('VERSION').read().strip()

libs = []
if os.name == 'posix':
    libs.append('m')

class win_build_ext(build_ext):
    def get_export_symbols(self, ext):
        return ext.export_symbols + ["calc_sdwgw", "calc_sdlwgw", "calc_lwgw", "calc_ace", "delete_doubleptr"]

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

setup(name='colorcorrect',
      version=version,
      description="imprement some of color correction algorithms",
      long_description=open('README').read(),
      classifiers=[],
      keywords=('image-processing computer-vision'),
      author='Shunsuke Aihara',
      author_email='s.aihara@gmail.com',
      url='http://www.bitbucket.org/aihara/colorcorrect',
      license='MIT License',
      package_dir={'': 'src'},
      packages=['colorcorrect'],
      ext_modules=[
          Extension(
              'colorcorrect._cutil', [
                  'cutil/cutil.cpp',
              ],
              include_dirs=['cutil'],
              libraries=libs,
              extra_compile_args=[],
          ),
      ],
      install_requires=["numpy", "Pillow", "six"],
      tests_require=["nose"],
      test_suite='nose.collector',
      cmdclass = {"build_ext": get_build_ext()}
      )
