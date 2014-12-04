from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

cython_modules = ['cypico/pico.pyx']

requirements = ['numpy==1.9.0', 'Cython==0.21']

setup(name='cypico',
      version='0.2',
      description='A Cython wrapper around the Pico face detection library.',
      author='Patrick Snape',
      author_email='p.snape@imperial.ac.uk',
      url='https://github.com/menpo/cypico',
      include_dirs=[np.get_include()],
      ext_modules=cythonize(cython_modules, quiet=True),
      package_data={'cypico': ['pico/runtime/cascades/*',
                               'pico/runtime/picort.*',
                               'pico.pyx',
                               'pico.pxd',
                               'pico_wrapper.h',
                               'pico_wrapper.c']},
      install_requires=requirements,
      packages=find_packages()
)
