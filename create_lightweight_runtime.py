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


def get_site_packages_path():
    """site-packagesディレクトリのパスを取得"""
    for path in sys.path:
        if 'site-packages' in path:
            return Path(path)
    # 見つからない場合は、Pythonのインストールディレクトリから推測
    python_dir = Path(sys.executable).parent
    return python_dir / "Lib" / "site-packages"


def create_lightweight_runtime():
    """軽量Pythonランタイムを作成"""

    print("Creating lightweight Python runtime...")

    # 必要なモジュールのリスト（より包括的に）
    required_modules = [
        'numpy',
        'cv2',
        'sklearn',
        'scipy',
        'mediapipe',
        'joblib',
        'PIL',  # Pillow
        'pandas',  # 必要に応じて
    ]

    # 軽量ランタイムディレクトリを作成
    runtime_dir = Path("lightweight_runtime")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(exist_ok=True)

    # Python実行ファイルをコピー
    python_exe = Path(sys.executable)
    shutil.copy2(python_exe, runtime_dir / "python.exe")
    print(f"Copied Python executable: {python_exe}")

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
            print(f"Copied DLL: {dll}")

    # 最小限のライブラリディレクトリを作成
    lib_dir = runtime_dir / "Lib"
    lib_dir.mkdir(exist_ok=True)

    # site-packagesディレクトリを作成
    site_packages = lib_dir / "site-packages"
    site_packages.mkdir(exist_ok=True)

    # 必要なモジュールをコピー
    source_site_packages = get_site_packages_path()
    print(f"Source site-packages: {source_site_packages}")

    for module in required_modules:
        module_path = source_site_packages / module
        if module_path.exists():
            target_path = site_packages / module
            if module_path.is_dir():
                shutil.copytree(module_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(module_path, target_path)
            print(f"Copied module: {module}")
        else:
            print(f"Module not found: {module} at {module_path}")

    # 作成したCythonモジュールをコピー
    cython_module = Path("BloodPressureEstimation.dll")
    if cython_module.exists():
        shutil.copy2(cython_module, runtime_dir /
                     "BloodPressureEstimation.dll")
        print("Copied Cython module: BloodPressureEstimation.dll")

    # モデルファイルをコピー
    models_dir = runtime_dir / "models"
    if Path("models").exists():
        shutil.copytree("models", models_dir, dirs_exist_ok=True)
        print("Copied models directory")

    # Pythonパッケージの初期化ファイルを作成
    init_file = site_packages / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print("Created __init__.py for site-packages")

    # 各モジュールディレクトリにも__init__.pyを作成
    for module in required_modules:
        module_dir = site_packages / module
        if module_dir.exists() and module_dir.is_dir():
            init_file = module_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    # テスト用のPythonスクリプトを作成
    test_script = runtime_dir / "test_import.py"
    test_content = '''#!/usr/bin/env python3
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import BloodPressureEstimation
    print("SUCCESS: BloodPressureEstimation module imported successfully")
    print(f"Module location: {BloodPressureEstimation.__file__}")
except Exception as e:
    print(f"ERROR: Failed to import BloodPressureEstimation: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("SUCCESS: NumPy imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import cv2
    print("SUCCESS: OpenCV imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import OpenCV: {e}")
    sys.exit(1)

try:
    import sklearn
    print("SUCCESS: scikit-learn imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import scikit-learn: {e}")
    sys.exit(1)

try:
    import mediapipe
    print("SUCCESS: MediaPipe imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import MediaPipe: {e}")
    sys.exit(1)

print("All modules imported successfully!")
'''

    with open(test_script, 'w') as f:
        f.write(test_content)
    print("Created test import script")

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
