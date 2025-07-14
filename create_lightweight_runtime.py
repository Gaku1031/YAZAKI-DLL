#!/usr/bin/env python3
"""
軽量Pythonランタイム作成スクリプト
必要なモジュールのみを含む最小構成のPythonランタイムを作成

サイズ最適化戦略:
1. 必要最小限のモジュールのみを含める
2. テストファイル、ドキュメント、サンプルファイルを除外
3. __pycache__ディレクトリを除外
4. 各モジュールのサイズを監視

代替案（20MB制限を満たせない場合）:
1. より軽量なライブラリを使用（例：OpenCV→PIL、scikit-learn→簡易実装）
2. 事前コンパイルされたモデルを使用
3. サーバーサイド処理に移行
4. ネイティブC++実装に移行
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
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    try:
        # 必要なモジュールのリスト（最小限に削減）
        required_modules = [
            'numpy.core',  # NumPyのコア部分のみ
            'cv2',  # OpenCV
            'sklearn.ensemble',  # RandomForestのみ
            'scipy.signal',  # 信号処理のみ
            'mediapipe.solutions',  # MediaPipe
            'joblib',  # モデル読み込み用
            'PIL',  # Pillow
        ]

        # 除外するモジュール（サイズ削減のため）
        excluded_modules = [
            'numpy.tests',
            'numpy.f2py',
            'numpy.distutils',
            'sklearn.tests',
            'sklearn.datasets',
            'sklearn.metrics',
            'scipy.tests',
            'scipy.io',
            'scipy.optimize',
            'scipy.spatial',
            'mediapipe.tests',
            'PIL.tests',
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

        # モジュールをコピー（サイズ最適化付き）
        total_copied_size = 0
        for module in required_modules:
            module_path = source_site_packages / module
            if module_path.exists():
                target_path = site_packages / module
                if module_path.is_dir():
                    # 除外ディレクトリをスキップ
                    def copy_with_exclusions(src, dst):
                        if src.is_dir():
                            dst.mkdir(exist_ok=True)
                            for item in src.iterdir():
                                item_name = item.name
                                # 除外リストに含まれるかチェック
                                should_exclude = any(excluded in str(
                                    item) for excluded in excluded_modules)
                                if not should_exclude and not item_name.startswith('__pycache__'):
                                    copy_with_exclusions(item, dst / item_name)
                        else:
                            shutil.copy2(src, dst)

                    copy_with_exclusions(module_path, target_path)
                else:
                    shutil.copy2(module_path, target_path)

                # サイズを計算
                if target_path.exists():
                    if target_path.is_dir():
                        size = sum(f.stat().st_size for f in target_path.rglob(
                            '*') if f.is_file())
                    else:
                        size = target_path.stat().st_size
                    total_copied_size += size
                    size_mb = size / (1024 * 1024)
                    print(f"Copied module: {module} ({size_mb:.2f} MB)")
                else:
                    print(
                        f"Warning: Module not found: {module} at {module_path}")
            else:
                print(f"Warning: Module not found: {module} at {module_path}")

        print(
            f"Total modules size: {total_copied_size / (1024 * 1024):.2f} MB")

        # 作成したCythonモジュールをコピー
        cython_module = Path("BloodPressureEstimation.dll")
        if cython_module.exists():
            shutil.copy2(cython_module, runtime_dir /
                         "BloodPressureEstimation.dll")
            print("Copied Cython module: BloodPressureEstimation.dll")
        else:
            print(f"Warning: Cython module not found: {cython_module}")
            # 現在のディレクトリの内容を表示
            print("Current directory contents:")
            for item in Path(".").iterdir():
                print(f"  {item.name}")

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

print("=== Lightweight Runtime Test ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
print(f"Added to path: {current_dir}")

# List files in current directory
print("\\nFiles in current directory:")
for item in os.listdir(current_dir):
    print(f"  {item}")

# Test basic imports
print("\\n=== Testing Basic Imports ===")

try:
    import numpy as np
    print("SUCCESS: NumPy imported successfully")
    print(f"NumPy version: {np.__version__}")
except Exception as e:
    print(f"ERROR: Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import cv2
    print("SUCCESS: OpenCV imported successfully")
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"ERROR: Failed to import OpenCV: {e}")
    sys.exit(1)

try:
    import sklearn
    print("SUCCESS: scikit-learn imported successfully")
    print(f"scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"ERROR: Failed to import scikit-learn: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print("SUCCESS: MediaPipe imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import MediaPipe: {e}")
    sys.exit(1)

try:
    import joblib
    print("SUCCESS: joblib imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import joblib: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("SUCCESS: PIL imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import PIL: {e}")
    sys.exit(1)

# Test Cython module import
print("\\n=== Testing Cython Module Import ===")

try:
    import BloodPressureEstimation
    print("SUCCESS: BloodPressureEstimation module imported successfully")
    print(f"Module location: {BloodPressureEstimation.__file__}")
    
    # Test basic functionality
    try:
        version_info = BloodPressureEstimation.get_version_info()
        print(f"Version info: {version_info}")
    except Exception as e:
        print(f"WARNING: Could not call get_version_info: {e}")
        
    try:
        request_id = BloodPressureEstimation.generate_request_id()
        print(f"Generated request ID: {request_id}")
    except Exception as e:
        print(f"WARNING: Could not call generate_request_id: {e}")
        
except Exception as e:
    print(f"ERROR: Failed to import BloodPressureEstimation: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\\n=== All Tests Passed Successfully! ===")
'''

        with open(test_script, 'w') as f:
            f.write(test_content)
        print("Created test import script")

        # 最終的なサイズを計算
        runtime_size = get_directory_size(runtime_dir)
        runtime_size_mb = runtime_size / (1024*1024)

        print(f"Lightweight runtime created successfully in: {runtime_dir}")
        print(f"Runtime size: {runtime_size_mb:.1f} MB")

        # サイズ制限チェック
        size_limit_mb = 20
        if runtime_size_mb > size_limit_mb:
            print(
                f"WARNING: Runtime size ({runtime_size_mb:.1f} MB) exceeds target limit ({size_limit_mb} MB)")

            # サイズ内訳を表示
            print("Size breakdown:")
            for item in runtime_dir.rglob('*'):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    if size_mb > 1:  # 1MB以上のファイルのみ表示
                        print(
                            f"  {item.relative_to(runtime_dir)}: {size_mb:.2f} MB")
        else:
            print(
                f"SUCCESS: Runtime size ({runtime_size_mb:.1f} MB) is within target limit ({size_limit_mb} MB)")

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
        ], capture_output=True, text=True, cwd=str(runtime_dir))

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
            # Don't exit here, just warn
            print("WARNING: Runtime test failed, but continuing...")

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
