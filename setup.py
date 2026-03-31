from setuptools import setup, Extension
import numpy

setup(ext_modules=[
    Extension("amriso", ["amriso.c"], include_dirs=[numpy.get_include()])
])
