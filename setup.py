"""
血圧推定DLLビルド用セットアップスクリプト
"""

from cx_Freeze import setup, Executable
import sys
import os

# ビルドオプション
build_exe_options = {
    "packages": [
        "cv2", "mediapipe", "numpy", "scipy", "sklearn", "pandas", 
        "joblib", "pywt", "threading", "ctypes", "logging"
    ],
    "excludes": ["tkinter", "matplotlib", "PyQt5"],
    "include_files": [
        ("models/", "models/"),
        ("sample-data/", "sample-data/")
    ],
    "zip_include_packages": "*",
    "zip_exclude_packages": [],
    "silent": True
}

# 32-bit DLL用の実行可能ファイル設定
dll_target = Executable(
    "bp_estimation_dll.py",
    base=None,
    target_name="BloodPressureEstimation.dll" if sys.platform == "win32" else "BloodPressureEstimation.so"
)

setup(
    name="BloodPressureEstimationDLL",
    version="1.0.0",
    description="血圧推定DLL - rPPGアルゴリズムによる動画からの血圧推定",
    options={"build_exe": build_exe_options},
    executables=[dll_target]
)