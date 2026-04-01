import os
import sys
import platform
import tempfile
import shutil
from setuptools import setup, Extension
import numpy


def has_openmp():
    from distutils.ccompiler import new_compiler
    from distutils.sysconfig import customize_compiler

    compiler = new_compiler()
    customize_compiler(compiler)

    if platform.system() == "Darwin":
        flags = ["-Xpreprocessor", "-fopenmp"]
        libs = ["omp"]
        prefix = os.environ.get("LIBOMP_PREFIX", "")
        if not prefix:
            for d in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
                if os.path.isdir(d):
                    prefix = d
                    break
        include = [os.path.join(prefix, "include")] if prefix else []
        libdir = [os.path.join(prefix, "lib")] if prefix else []
    else:
        flags = ["-fopenmp"]
        libs = ["gomp"]
        include = []
        libdir = []

    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "test.c")
        with open(src, "w") as f:
            f.write("#include <omp.h>\nint main(void)"
                    "{return omp_get_num_threads();}\n")
        obj = compiler.compile([src],
                               output_dir=tmp,
                               extra_preargs=flags,
                               include_dirs=include)
        compiler.link_executable(obj,
                                 os.path.join(tmp, "test"),
                                 library_dirs=libdir,
                                 libraries=libs)
        return flags, libs, include, libdir
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp)


omp = has_openmp()
if omp:
    omp_compile, omp_libs, omp_include, omp_libdir = omp
    print("amriso: building with OpenMP support")
else:
    omp_compile, omp_libs, omp_include, omp_libdir = [], [], [], []
    print("amriso: building without OpenMP (install libomp for parallel support)")

setup(ext_modules=[
    Extension(
        "amriso", ["amriso.c"],
        include_dirs=[numpy.get_include()] + omp_include,
        library_dirs=omp_libdir,
        libraries=omp_libs,
        extra_compile_args=omp_compile,
    )
])
