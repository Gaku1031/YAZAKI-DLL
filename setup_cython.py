#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cython setup for Blood Pressure Estimation DLL
Compiles Python code to C++ for C# integration with code obfuscation
"""

import os
import sys
import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

# Cython compiler options for obfuscation
Options.annotate = False

# Platform-specific settings
if platform.system() == "Windows":
    extra_compile_args = ["/O2", "/DNDEBUG", "/DWIN32", "/D_WINDOWS"]
    extra_link_args = []
    define_macros = [("WIN32", None), ("_WINDOWS", None)]
else:
    extra_compile_args = ["-O3", "-DNDEBUG", "-fPIC"]
    extra_link_args = []
    define_macros = []

# Extension configuration
extensions = [
    Extension(
        "bp_estimation_cython",
        sources=["bp_estimation_cython.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        include_dirs=[],
        libraries=[],
        library_dirs=[],
    ),
    Extension(
        "dll_wrapper_cython",
        sources=["dll_wrapper_cython.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        include_dirs=[],
        libraries=[],
        library_dirs=[],
    )
]

# Cythonize with obfuscation settings
cython_extensions = cythonize(
    extensions,
    compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": True,  # Allow negative indexing
        "cdivision": True,
        "nonecheck": False,
        "embedsignature": False,
    }
)

# Setup configuration
setup(
    name="BloodPressureEstimationCython",
    version="1.0.0",
    description="Cython-based Blood Pressure Estimation DLL with code obfuscation",
    author="IKI Development Team",
    author_email="dev@iki.com",
    ext_modules=cython_extensions,
    install_requires=[
        "cython>=3.0.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "scipy>=1.11.0",
        "mediapipe>=0.10.0",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
