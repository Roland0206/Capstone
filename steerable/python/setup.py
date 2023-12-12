from setuptools import find_packages
from distutils.core import setup, Extension
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import numpy as np

# _README           = os.path.join(os.path.dirname(__file__), 'README.md')
# _LONG_DESCRIPTION = open(_README).read()

# Setup information
setup(
    name = 'steerable',
    version = '1.0.0',
    packages = find_packages(),
    description = 'Steerable filters for edge and ridge detection',
    author = 'Francois Aguet (Broad Institute)',
    author_email = 'francois@broadinstitute.org',
    # long_description = _LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    ext_modules = cythonize(
        [Extension(
            "steerable",
            sources=["steerablefilter.pyx", "../src/steerableDetector.cpp"],
            include_dirs=[np.get_include(), "/usr/local/include/", '../src/'],
            language="C++",
            library_dirs=['/usr/local/lib'],  # libgsl.a
            extra_link_args=["-lgsl", "-lgslcblas"],
        )],
    ),
    install_requires = [
        'Cython',
        'numpy',
        'matplotlib',
        'Pillow',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
