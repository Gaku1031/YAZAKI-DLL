#!/usr/bin/env python3
"""
Setup script to create pure Windows DLL from Cython code
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Define the Cython extension as a pure Windows DLL
extensions = [
    Extension(
        "BloodPressureEstimation",  # DLL name
        sources=["bp_estimation_cython.pyx"],
        include_dirs=[np.get_include()],
        libraries=["python311"],  # Link with Python library
        define_macros=[("MS_WIN64", None)],  # Windows 64-bit
        extra_compile_args=["/O2", "/MD"],  # Optimize and use MD runtime
        extra_link_args=["/DLL", "/EXPORT:InitializeDLL",
                         "/EXPORT:StartBloodPressureAnalysisRequest",
                         "/EXPORT:GetProcessingStatus",
                         "/EXPORT:CancelBloodPressureAnalysis",
                         "/EXPORT:GetVersionInfo",
                         "/EXPORT:GenerateRequestId",
                         "/EXPORT:DllMain"]  # Export C functions
    )
]

setup(
    name="BloodPressureEstimation",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'cdivision': True,
        'embedsignature': False
    }),
    zip_safe=False,
)
