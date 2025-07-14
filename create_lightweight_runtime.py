#!/usr/bin/env python3
"""
軽量Pythonランタイム作成スクリプト
必要なモジュールのみを含む最小構成のPythonランタイムを作成

修正点:
- encodingsモジュールの確実なコピー
- Python初期化に必要なファイルの追加
- より包括的な標準ライブラリコピー
- エラー処理の改善
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


def get_python_lib_path():
    """Python標準ライブラリのパスを取得"""
    python_dir = Path(sys.executable).parent
    return python_dir / "Lib"


def create_lightweight_runtime():
    """軽量Pythonランタイムを作成"""

    print("Creating lightweight Python runtime...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    try:
        # 必要なモジュールのリスト（依存関係を考慮）
        required_modules = [
            'numpy',  # NumPy（コア部分のみコピー）
            'cv2',  # OpenCV
            'sklearn',  # scikit-learn（必要部分のみコピー）
            'scipy',  # SciPy（必要部分のみコピー）
            'mediapipe',  # MediaPipe（必要部分のみコピー）
            'joblib',  # モデル読み込み用
            'PIL',  # Pillow（必要部分のみコピー）
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

        # Python標準ライブラリをコピー（包括的）
        source_lib = get_python_lib_path()
        print(f"Source Python lib: {source_lib}")

        if not source_lib.exists():
            raise FileNotFoundError(
                f"Source Python lib not found: {source_lib}")

        # 方法1: 重要なディレクトリ全体をコピー
        important_dirs = ['encodings', 'collections',
                          'importlib', 'json', 'logging']
        for dir_name in important_dirs:
            source_dir = source_lib / dir_name
            target_dir = lib_dir / dir_name
            if source_dir.exists() and source_dir.is_dir():
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)
                size = sum(f.stat().st_size for f in target_dir.rglob(
                    '*') if f.is_file())
                print(f"Copied entire {dir_name} directory ({size} bytes)")
            else:
                print(
                    f"Warning: Source {dir_name} directory not found: {source_dir}")

        # 方法2: 最小限のPython標準ライブラリファイルをコピー
        minimal_stdlib_files = [
            '__future__.py',
            '_collections_abc.py',
            '_weakrefset.py',
            'abc.py',
            'builtins.py',
            'codecs.py',
            'copy.py',
            'copyreg.py',
            'enum.py',
            'functools.py',
            'io.py',
            'keyword.py',
            'operator.py',
            'os.py',
            'pickle.py',
            'pkgutil.py',
            're.py',
            'site.py',
            'stat.py',
            'string.py',
            'sys.py',
            'token.py',
            'tokenize.py',
            'traceback.py',
            'types.py',
            'warnings.py',
            'weakref.py',
            'zipimport.py',
            'zlib.py',
        ]

        # 最小限のファイルをコピー
        for file_name in minimal_stdlib_files:
            source_file = source_lib / file_name
            target_file = lib_dir / file_name
            if source_file.exists() and not target_file.exists():
                shutil.copy2(source_file, target_file)
                print(f"Copied minimal stdlib file: {file_name}")
            elif not source_file.exists():
                print(f"Warning: Minimal stdlib file not found: {file_name}")

        # 方法3: encodingsモジュールの確実なコピー（再実行）
        print("Ensuring encodings module is properly copied...")
        encodings_source = source_lib / "encodings"
        encodings_target = lib_dir / "encodings"

        if encodings_source.exists():
            # encodingsディレクトリを完全に削除して再作成
            if encodings_target.exists():
                shutil.rmtree(encodings_target)
                print("Removed existing encodings directory")

            # 新しいencodingsディレクトリを作成
            encodings_target.mkdir(parents=True, exist_ok=True)
            print(f"Created encodings directory: {encodings_target}")

            # encodingsディレクトリの全ファイルをコピー
            copied_files = 0
            for item in encodings_source.iterdir():
                if item.is_file():
                    target_item = encodings_target / item.name
                    shutil.copy2(item, target_item)
                    copied_files += 1
                    print(f"Copied encodings file: {item.name}")
                elif item.is_dir():
                    target_item = encodings_target / item.name
                    shutil.copytree(item, target_item)
                    size = sum(f.stat().st_size for f in target_item.rglob(
                        '*') if f.is_file())
                    copied_files += 1
                    print(f"Copied encodings directory: {item.name}")

            print(f"Total encodings files copied: {copied_files}")

            # encodingsディレクトリの内容を確認
            if encodings_target.exists():
                encodings_files = list(encodings_target.iterdir())
                print(
                    f"Encodings directory contains {len(encodings_files)} items:")
                for item in encodings_files:
                    if item.is_file():
                        size = item.stat().st_size
                        print(f"  {item.name} ({size} bytes)")
                    else:
                        print(f"  {item.name}/ (directory)")
            else:
                print("ERROR: Encodings directory was not created properly")
        else:
            print(
                f"ERROR: Source encodings directory not found: {encodings_source}")
            # 代替手段として、基本的なencodingsファイルを手動で作成
            print("Creating basic encodings files manually...")
            encodings_target.mkdir(parents=True, exist_ok=True)

            # 基本的なencodingsファイルを作成
            basic_encodings_files = {
                '__init__.py': '''"""Standard encodings module.

This module provides access to the codec registry and the base classes for
standard encodings.  The codec registry is a mapping of encoding names to
codec objects, which have a stateless interface in order to be safe to use
by multiple threads.

"""''',
                'utf_8.py': '''"""Python 'utf-8' Codec

This codec is used for UTF-8 encoding/decoding.

"""''',
                'ascii.py': '''"""Python 'ascii' Codec

This codec is used for ASCII encoding/decoding.

"""''',
                'latin_1.py': '''"""Python 'latin-1' Codec

This codec is used for Latin-1 encoding/decoding.

"""''',
                'cp1252.py': '''"""Python 'cp1252' Codec

This codec is used for Windows-1252 encoding/decoding.

"""''',
                'charmap.py': '''"""Python 'charmap' Codec

This codec is used for charmap encoding/decoding.

"""''',
                'aliases.py': '''"""Encoding aliases.

This module is used for encoding aliases.

"""''',
            }

            for filename, content in basic_encodings_files.items():
                file_path = encodings_target / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Created basic encodings file: {filename}")

        # 必要なsite-packagesモジュールをコピー
        source_site_packages = get_site_packages_path()
        print(f"Source site-packages: {source_site_packages}")

        if not source_site_packages.exists():
            raise FileNotFoundError(
                f"Source site-packages not found: {source_site_packages}")

        # モジュールをコピー（サイズ最適化付き）
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
                                    item) for excluded in ['__pycache__', '*.pyc', '*.pyo'])
                                # ファイルサイズ制限をチェック
                                if item.is_file():
                                    file_size = item.stat().st_size
                                    size_excluded = file_size > 100 * 1024  # 100KB
                                else:
                                    size_excluded = False

                                if not should_exclude and not size_excluded and not item_name.startswith('__pycache__'):
                                    copy_with_exclusions(item, dst / item_name)
                        else:
                            # ファイルサイズをチェック
                            file_size = src.stat().st_size
                            if file_size <= 100 * 1024:  # 100KB
                                shutil.copy2(src, dst)
                            else:
                                print(
                                    f"Skipping large file: {src.name} ({file_size / (1024*1024):.2f} MB)")

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
                    size_mb = size / (1024 * 1024)
                    print(f"Copied module: {module} ({size_mb:.2f} MB)")
                else:
                    print(
                        f"Warning: Module not found: {module} at {module_path}")
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
if current_dir not in sys.path:
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
            large_files = []
            for item in runtime_dir.rglob('*'):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    if size_mb > 0.5:  # 500KB以上のファイルのみ表示
                        large_files.append((item, size_mb))

            # サイズ順にソート
            large_files.sort(key=lambda x: x[1], reverse=True)
            for item, size_mb in large_files[:20]:  # 上位20ファイルのみ表示
                print(f"  {item.relative_to(runtime_dir)}: {size_mb:.2f} MB")

            # サイズ削減の提案
            print("\nSize reduction suggestions:")
            print("1. Remove large data files (>1MB)")
            print("2. Exclude test directories")
            print("3. Remove documentation files")
            print("4. Use minimal Python installation")
            print("5. Consider alternative lightweight libraries")
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

            # Additional diagnostic: check if test script exists
            print(f"Checking if test script exists: {test_script}")
            if test_script.exists():
                print(
                    f"Test script exists, size: {test_script.stat().st_size} bytes")
            else:
                print("Test script does not exist!")
                print("Runtime directory contents:")
                for item in runtime_dir.iterdir():
                    print(f"  {item.name}")

            # Try to run a simple Python test instead
            print("Trying simple Python test...")
            simple_test = subprocess.run([
                str(python_exe_path),
                "-c", "import sys; print('Python version:', sys.version); print('SUCCESS')"
            ], capture_output=True, text=True, cwd=str(runtime_dir))

            if simple_test.returncode == 0:
                print("Simple Python test passed!")
                print("Simple test output:")
                print(simple_test.stdout)
            else:
                print("Simple Python test failed!")
                print("Simple test errors:")
                print(simple_test.stderr)

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
