#!/usr/bin/env python3
"""
軽量Pythonランタイム作成スクリプト
必要なモジュールのみを含む最小構成のPythonランタイムを作成
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def create_lightweight_runtime():
    """軽量Pythonランタイムを作成"""

    print("Creating lightweight Python runtime...")

    # 必要なモジュールのみをリストアップ
    required_modules = [
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        'numpy.linalg._umath_linalg',
        'cv2',
        'sklearn.ensemble._forest',
        'sklearn.tree._utils',
        'sklearn.utils._cython_blas',
        'mediapipe.python.solutions.face_mesh',
        'mediapipe.python.solutions.drawing_utils',
        'scipy.special.cython_special',
        'joblib._parallel_backends',
        'joblib.externals.loky.backend',
        'BloodPressureEstimation'  # 作成したCythonモジュール
    ]

    # 軽量ランタイムディレクトリを作成
    runtime_dir = Path("lightweight_runtime")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(exist_ok=True)

    # Python実行ファイルをコピー
    python_exe = Path(sys.executable)
    shutil.copy2(python_exe, runtime_dir / "python.exe")

    # 必要なDLLをコピー
    python_dlls = [
        "python311.dll",
        "python3.dll",
        "vcruntime140.dll",
        "msvcp140.dll"
    ]

    python_dir = python_exe.parent
    for dll in python_dlls:
        dll_path = python_dir / dll
        if dll_path.exists():
            shutil.copy2(dll_path, runtime_dir / dll)

    # 最小限のライブラリディレクトリを作成
    lib_dir = runtime_dir / "Lib"
    lib_dir.mkdir(exist_ok=True)

    # 必要なモジュールのみをコピー
    site_packages = Path(sys.__path__[0]).parent / "site-packages"

    for module in required_modules:
        module_path = site_packages / module.replace('.', '/')
        if module_path.exists():
            # モジュールディレクトリを作成
            target_path = lib_dir / module.replace('.', '/')
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if module_path.is_dir():
                shutil.copytree(module_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(module_path, target_path)

    # 作成したCythonモジュールをコピー
    cython_module = Path("BloodPressureEstimation.dll")
    if cython_module.exists():
        shutil.copy2(cython_module, runtime_dir /
                     "BloodPressureEstimation.dll")

    # モデルファイルをコピー
    models_dir = runtime_dir / "models"
    if Path("models").exists():
        shutil.copytree("models", models_dir, dirs_exist_ok=True)

    print(f"Lightweight runtime created in: {runtime_dir}")
    print(
        f"Runtime size: {get_directory_size(runtime_dir) / (1024*1024):.1f} MB")

    return runtime_dir


def get_directory_size(path):
    """ディレクトリサイズを計算"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


if __name__ == "__main__":
    create_lightweight_runtime()
