# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import sys
sys.path.append('./src')
sys.path.append('./test')
version = open('VERSION').read().strip()

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
              libraries=['m'],
              extra_compile_args=[],
          ),
      ],
      install_requires=["numpy", "Pillow", "six"],
      test_requires=["nose"]
      )
