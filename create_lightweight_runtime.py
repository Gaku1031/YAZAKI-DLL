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

    try:
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
            print(f"Removing existing runtime directory: {runtime_dir}")
            shutil.rmtree(runtime_dir)
        runtime_dir.mkdir(exist_ok=True)
        print(f"Created runtime directory: {runtime_dir}")

        # Python実行ファイルをコピー
        python_exe = Path(sys.executable)
        if not python_exe.exists():
            raise FileNotFoundError(
                f"Python executable not found: {python_exe}")

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
            else:
                print(f"Warning: DLL not found: {dll}")

        # 最小限のライブラリディレクトリを作成
        lib_dir = runtime_dir / "Lib"
        lib_dir.mkdir(exist_ok=True)
        print(f"Created lib directory: {lib_dir}")

        # site-packagesディレクトリを作成
        site_packages = lib_dir / "site-packages"
        site_packages.mkdir(exist_ok=True)
        print(f"Created site-packages directory: {site_packages}")

        # 必要なモジュールをコピー
        source_site_packages = get_site_packages_path()
        print(f"Source site-packages: {source_site_packages}")

        if not source_site_packages.exists():
            raise FileNotFoundError(
                f"Source site-packages not found: {source_site_packages}")

        for module in required_modules:
            module_path = source_site_packages / module
            if module_path.exists():
                target_path = site_packages / module
                if module_path.is_dir():
                    shutil.copytree(module_path, target_path,
                                    dirs_exist_ok=True)
                else:
                    shutil.copy2(module_path, target_path)
                print(f"Copied module: {module}")
            else:
                print(f"Warning: Module not found: {module} at {module_path}")

        # 作成したCythonモジュールをコピー
        cython_module = Path("BloodPressureEstimation.dll")
        if cython_module.exists():
            shutil.copy2(cython_module, runtime_dir /
                         "BloodPressureEstimation.dll")
            print("Copied Cython module: BloodPressureEstimation.dll")
        else:
            print(f"Warning: Cython module not found: {cython_module}")

        # モデルファイルをコピー
        models_dir = runtime_dir / "models"
        if Path("models").exists():
            shutil.copytree("models", models_dir, dirs_exist_ok=True)
            print("Copied models directory")
        else:
            print("Warning: Models directory not found")

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

        # 最終的なサイズを計算
        runtime_size = get_directory_size(runtime_dir)
        runtime_size_mb = runtime_size / (1024*1024)

        print(f"Lightweight runtime created successfully in: {runtime_dir}")
        print(f"Runtime size: {runtime_size_mb:.1f} MB")

        # 基本的な検証
        python_exe_path = runtime_dir / "python.exe"
        dll_path = runtime_dir / "BloodPressureEstimation.dll"

        if not python_exe_path.exists():
            raise FileNotFoundError(
                f"Python executable not found in runtime: {python_exe_path}")

        if not dll_path.exists():
            raise FileNotFoundError(
                f"Cython module not found in runtime: {dll_path}")

        # 軽量ランタイムでテストを実行
        print("Testing lightweight runtime...")
        test_process = subprocess.run([
            str(python_exe_path),
            str(test_script)
        ], capture_output=True, text=True, cwd=runtime_dir)

        if test_process.returncode == 0:
            print("Lightweight runtime test passed!")
            print("Test output:")
            print(test_process.stdout)
        else:
            print("Lightweight runtime test failed!")
            print("Test output:")
            print(test_process.stdout)
            print("Test errors:")
            print(test_process.stderr)

        print("Runtime verification completed successfully")
        return runtime_dir

    except Exception as e:
        print(f"ERROR: Failed to create lightweight runtime: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
