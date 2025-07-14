#!/usr/bin/env python3
"""
軽量Pythonランタイム作成スクリプト (完全修正版)
encodingsモジュールの問題を確実に解決
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path


def get_python_installation_path():
    """Python インストールパスを取得"""
    python_exe = Path(sys.executable)
    python_dir = python_exe.parent

    # 複数の候補を確認
    possible_paths = [
        python_dir,
        python_dir.parent,
        Path(sys.prefix),
        Path(sys.base_prefix) if hasattr(sys, 'base_prefix') else None
    ]

    for path in possible_paths:
        if path and path.exists():
            lib_path = path / "Lib"
            if lib_path.exists() and (lib_path / "encodings").exists():
                return path

    raise RuntimeError("Cannot find Python installation with Lib/encodings")


def create_python311_zip(runtime_dir, source_lib):
    """python311.zipを作成（標準ライブラリの圧縮版）"""
    zip_path = runtime_dir / "python311.zip"

    print(f"Creating {zip_path}...")

    # 重要なモジュールをzipに含める
    essential_modules = [
        'encodings',
        'codecs.py',
        'abc.py',
        '_collections_abc.py',
        'io.py',
        'os.py',
        'site.py',
        'stat.py',
        'ntpath.py',
        'posixpath.py',
        'genericpath.py',
        'types.py',
        'functools.py',
        'operator.py',
        'keyword.py',
        'warnings.py',
        'linecache.py',
        'reprlib.py',
        '_weakrefset.py',
        'weakref.py',
        'copyreg.py',
        'enum.py',
        'collections',
        'importlib'
    ]

    # encodingsディレクトリが存在することを確認
    encodings_source = source_lib / "encodings"
    if not encodings_source.exists():
        raise RuntimeError(
            f"Critical: encodings directory not found at {encodings_source}")
    else:
        print(f"Found encodings directory: {encodings_source}")
        encodings_files = list(encodings_source.rglob("*.py"))
        print(f"Found {len(encodings_files)} encodings files")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for module in essential_modules:
            source_path = source_lib / module

            if source_path.is_file():
                zf.write(source_path, module)
                print(f"  Added to zip: {module}")
            elif source_path.is_dir():
                for file_path in source_path.rglob("*.py"):
                    arcname = str(file_path.relative_to(source_lib))
                    zf.write(file_path, arcname)
                print(f"  Added directory to zip: {module}")

    print(f"Created python311.zip ({zip_path.stat().st_size / 1024:.1f} KB)")


def copy_essential_dlls(runtime_dir, python_installation):
    """必要なDLLファイルをコピー"""
    print("Copying essential DLLs...")

    # DLLの候補パス
    dll_search_paths = [
        python_installation,
        python_installation / "DLLs",
        Path(sys.executable).parent,
        Path(sys.executable).parent / "DLLs"
    ]

    essential_dlls = [
        "python311.dll",
        "python3.dll",
        "vcruntime140.dll",
        "msvcp140.dll"
    ]

    # DLLsディレクトリを作成
    dlls_dir = runtime_dir / "DLLs"
    dlls_dir.mkdir(exist_ok=True)

    for dll_name in essential_dlls:
        found = False
        for search_path in dll_search_paths:
            dll_path = search_path / dll_name
            if dll_path.exists():
                # ルートとDLLsディレクトリの両方にコピー
                shutil.copy2(dll_path, runtime_dir / dll_name)
                shutil.copy2(dll_path, dlls_dir / dll_name)
                print(f"  Copied DLL: {dll_name}")
                found = True
                break

        if not found:
            print(f"  Warning: DLL not found: {dll_name}")

    # 追加の.pydファイルをコピー
    system_dlls_dir = python_installation / "DLLs"
    if system_dlls_dir.exists():
        important_pyds = [
            "_ctypes.pyd",
            "_elementtree.pyd",
            "_hashlib.pyd",
            "_socket.pyd",
            "_ssl.pyd",
            "select.pyd",
            "_sqlite3.pyd",
            "_decimal.pyd"
        ]

        for pyd_name in important_pyds:
            pyd_path = system_dlls_dir / pyd_name
            if pyd_path.exists():
                shutil.copy2(pyd_path, dlls_dir / pyd_name)
                print(f"  Copied PYD: {pyd_name}")


def copy_stdlib_comprehensive(lib_dir, source_lib):
    """標準ライブラリを包括的にコピー"""
    print("Copying standard library comprehensively...")

    # 最重要: encodingsを最初にコピー
    encodings_source = source_lib / "encodings"
    encodings_target = lib_dir / "encodings"

    if encodings_source.exists():
        if encodings_target.exists():
            shutil.rmtree(encodings_target)

        shutil.copytree(encodings_source, encodings_target)
        files_count = len(list(encodings_target.rglob("*")))
        print(f"    Copied encodings directory ({files_count} files)")

        # encodingsディレクトリの内容を確認
        key_files = ['__init__.py', 'utf_8.py',
                     'ascii.py', 'latin_1.py', 'aliases.py']
        for key_file in key_files:
            key_path = encodings_target / key_file
            if key_path.exists():
                print(
                    f"    {key_file} exists ({key_path.stat().st_size} bytes)")
            else:
                print(f"    {key_file} missing")

        # encodingsディレクトリの内容を詳細に確認
        print(f"    Encodings directory contents:")
        for item in encodings_target.iterdir():
            if item.is_file():
                print(f"      File: {item.name} ({item.stat().st_size} bytes)")
            elif item.is_dir():
                print(f"      Dir: {item.name}")
                for subitem in item.iterdir():
                    if subitem.is_file():
                        print(
                            f"        {subitem.name} ({subitem.stat().st_size} bytes)")
    else:
        raise RuntimeError(
            f"Critical: encodings directory not found at {encodings_source}")

    # encodingsディレクトリが正しくコピーされたかテスト
    test_encodings = lib_dir / "encodings" / "__init__.py"
    if not test_encodings.exists():
        raise RuntimeError(
            f"Critical: encodings/__init__.py not found at {test_encodings}")
    else:
        print(
            f"    Verified: encodings/__init__.py exists ({test_encodings.stat().st_size} bytes)")

    # その他の重要なディレクトリ
    important_dirs = [
        'collections',
        'importlib',
        'json',
        'logging',
        'email',
        'urllib',
        'http',
        'xml'
    ]

    for dir_name in important_dirs:
        source_dir = source_lib / dir_name
        target_dir = lib_dir / dir_name

        if source_dir.exists() and source_dir.is_dir():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)
            files_count = len(list(target_dir.rglob("*")))
            print(f"    Copied {dir_name} ({files_count} files)")

    # 重要な単一ファイル
    important_files = [
        '__future__.py',
        '_collections_abc.py',
        '_weakrefset.py',
        'abc.py',
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
        'threading.py',
        'token.py',
        'tokenize.py',
        'traceback.py',
        'types.py',
        'warnings.py',
        'weakref.py',
        'zipimport.py',
        'ntpath.py',
        'posixpath.py',
        'genericpath.py',
        'sre_compile.py',
        'sre_constants.py',
        'sre_parse.py',
        'linecache.py',
        'reprlib.py',
        'locale.py',
        'heapq.py',
        'bisect.py',
        '_threading_local.py'
    ]

    for file_name in important_files:
        source_file = source_lib / file_name
        target_file = lib_dir / file_name

        if source_file.exists() and not target_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"    Copied {file_name}")


def copy_site_packages_optimized(site_packages_dir, source_site_packages):
    """site-packagesを最適化してコピー"""
    print("Copying site-packages (optimized)...")

    required_packages = {
        'numpy': ['core', 'lib', 'linalg', 'fft', 'random', 'ma', 'distutils'],
        'cv2': [],  # OpenCV
        'sklearn': ['ensemble', 'tree', 'linear_model', 'preprocessing', 'utils'],
        'scipy': ['sparse', 'linalg', 'interpolate', 'optimize'],
        'mediapipe': ['python', 'solutions'],
        'joblib': [],
        'PIL': []  # Pillow
    }

    for package_name, essential_parts in required_packages.items():
        source_package = source_site_packages / package_name
        target_package = site_packages_dir / package_name

        if source_package.exists():
            print(f"  Processing {package_name}...")

            if source_package.is_dir():
                target_package.mkdir(exist_ok=True)

                # __init__.pyを必ずコピー
                init_file = source_package / "__init__.py"
                if init_file.exists():
                    shutil.copy2(init_file, target_package / "__init__.py")

                # 基本的なファイルをコピー
                for item in source_package.iterdir():
                    if item.is_file() and item.suffix == '.py':
                        shutil.copy2(item, target_package / item.name)

                # 重要なサブディレクトリをコピー
                if essential_parts:
                    for part in essential_parts:
                        part_path = source_package / part
                        if part_path.exists():
                            target_part = target_package / part
                            if part_path.is_dir():
                                shutil.copytree(part_path, target_part,
                                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                            else:
                                shutil.copy2(part_path, target_part)
                else:
                    # 全体をコピー（除外パターン付き）
                    for item in source_package.iterdir():
                        if item.name.startswith('__pycache__'):
                            continue

                        target_item = target_package / item.name
                        if item.is_dir():
                            if not target_item.exists():
                                shutil.copytree(item, target_item,
                                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'tests'))
                        elif item.suffix not in ['.pyc', '.pyo']:
                            if not target_item.exists():
                                shutil.copy2(item, target_item)
            else:
                shutil.copy2(source_package, target_package)

            # サイズ確認
            if target_package.exists():
                if target_package.is_dir():
                    size = sum(f.stat().st_size for f in target_package.rglob(
                        '*') if f.is_file())
                else:
                    size = target_package.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"    Copied {package_name} ({size_mb:.2f} MB)")
        else:
            print(f"    Package not found: {package_name}")


def create_pyvenv_cfg(runtime_dir, python_installation):
    """pyvenv.cfgファイルを作成"""
    pyvenv_cfg = runtime_dir / "pyvenv.cfg"

    content = f"""home = {python_installation}
include-system-site-packages = false
version = {sys.version.split()[0]}
executable = {runtime_dir / 'python.exe'}
"""

    with open(pyvenv_cfg, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Created pyvenv.cfg")


def create_sitecustomize(runtime_dir):
    """sitecustomize.pyを作成してPythonパスを自動設定"""
    lib_dir = runtime_dir / "Lib"
    sitecustomize_path = lib_dir / "sitecustomize.py"

    content = '''"""
Site customization for lightweight runtime
"""
import sys
import os

# 現在のディレクトリをパスに追加
runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if runtime_dir not in sys.path:
    sys.path.insert(0, runtime_dir)

# site-packagesをパスに追加
site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site-packages')
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

print(f"Lightweight runtime initialized from: {runtime_dir}")
'''

    with open(sitecustomize_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Created sitecustomize.py")


def create_lightweight_runtime():
    """軽量Pythonランタイムを作成"""
    print("=== Creating Lightweight Python Runtime (Complete Fix) ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    try:
        # Python インストールパスを取得
        python_installation = get_python_installation_path()
        print(f"Python installation: {python_installation}")

        source_lib = python_installation / "Lib"
        source_site_packages = source_lib / "site-packages"

        print(f"Source Lib: {source_lib}")
        print(f"Source site-packages: {source_site_packages}")

        # ランタイムディレクトリを作成
        runtime_dir = Path("lightweight_runtime")
        if runtime_dir.exists():
            print(f"Removing existing runtime: {runtime_dir}")
            shutil.rmtree(runtime_dir)

        runtime_dir.mkdir()
        print(f"Created runtime directory: {runtime_dir}")

        # 1. Python実行ファイルをコピー
        python_exe = Path(sys.executable)
        shutil.copy2(python_exe, runtime_dir / "python.exe")
        print("Copied Python executable")

        # 2. 必要なDLLをコピー
        copy_essential_dlls(runtime_dir, python_installation)

        # 3. Libディレクトリを作成
        lib_dir = runtime_dir / "Lib"
        lib_dir.mkdir()

        # 4. site-packagesディレクトリを作成
        site_packages_dir = lib_dir / "site-packages"
        site_packages_dir.mkdir()

        # 5. 標準ライブラリをコピー (encodingsを含む)
        copy_stdlib_comprehensive(lib_dir, source_lib)

        # 6. python311.zipを作成
        create_python311_zip(runtime_dir, source_lib)

        # 7. site-packagesをコピー
        if source_site_packages.exists():
            copy_site_packages_optimized(
                site_packages_dir, source_site_packages)
        else:
            print(
                f"Warning: site-packages not found at {source_site_packages}")

        # 8. Cythonモジュールをコピー
        cython_files = ["BloodPressureEstimation.dll",
                        "BloodPressureEstimation.pyd"]
        for cython_file in cython_files:
            source_path = Path(cython_file)
            if source_path.exists():
                shutil.copy2(source_path, runtime_dir / cython_file)
                print(f"Copied Cython module: {cython_file}")

        # 9. modelsディレクトリをコピー
        models_candidates = ["models_compressed", "models"]
        for models_name in models_candidates:
            models_source = Path(models_name)
            if models_source.exists():
                models_target = runtime_dir / "models"
                shutil.copytree(models_source, models_target)
                print(f"Copied models from {models_name}")
                break

        # 10. 設定ファイルを作成
        create_pyvenv_cfg(runtime_dir, python_installation)
        create_sitecustomize(runtime_dir)

        # 11. __init__.pyファイルを作成
        init_files = [
            site_packages_dir / "__init__.py",
            lib_dir / "__init__.py"
        ]

        for init_file in init_files:
            if not init_file.exists():
                init_file.touch()

        # 12. テストスクリプトを作成
        create_test_script(runtime_dir)

        # 13. サイズ確認
        total_size = sum(
            f.stat().st_size for f in runtime_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"\nRuntime created successfully: {size_mb:.1f} MB")

        # 14. encodingsモジュールの最終確認
        print("\nFinal verification of encodings module...")
        encodings_init = runtime_dir / "Lib" / "encodings" / "__init__.py"
        if encodings_init.exists():
            print(
                f"✓ encodings/__init__.py exists ({encodings_init.stat().st_size} bytes)")
        else:
            raise RuntimeError(
                f"Critical: encodings/__init__.py not found at {encodings_init}")

        # encodingsディレクトリの内容を確認
        encodings_dir = runtime_dir / "Lib" / "encodings"
        if encodings_dir.exists():
            encodings_files = list(encodings_dir.rglob("*.py"))
            print(
                f"✓ encodings directory contains {len(encodings_files)} Python files")

            # 重要なファイルの存在確認
            important_encodings = ['__init__.py',
                                   'utf_8.py', 'ascii.py', 'latin_1.py']
            for important_file in important_encodings:
                file_path = encodings_dir / important_file
                if file_path.exists():
                    print(
                        f"✓ {important_file} exists ({file_path.stat().st_size} bytes)")
                else:
                    print(f"✗ {important_file} missing")
        else:
            raise RuntimeError(
                f"Critical: encodings directory not found at {encodings_dir}")

        # 14. 基本テストを実行
        test_runtime(runtime_dir)

        return runtime_dir

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_test_script(runtime_dir):
    """テストスクリプトを作成"""
    test_script = runtime_dir / "test_runtime.py"

    content = '''#!/usr/bin/env python3
import sys
import os

print("=== Lightweight Runtime Test ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

print("\\nPython path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# 重要なモジュールをテスト
print("\\n=== Testing Critical Imports ===")

# encodings (最重要)
try:
    import encodings
    print("encodings imported successfully")
    print(f"  Location: {encodings.__file__}")
except ImportError as e:
    print(f"encodings import failed: {e}")
    sys.exit(1)

# codecs
try:
    import codecs
    print("codecs imported successfully")
except ImportError as e:
    print(f"codecs import failed: {e}")
    sys.exit(1)

# その他の基本モジュール
basic_modules = ['os', 'sys', 'io', 'types', 'abc', 'collections']
for module in basic_modules:
    try:
        __import__(module)
        print(f"{module} imported successfully")
    except ImportError as e:
        print(f"{module} import failed: {e}")

print("\\n=== Testing Site-Packages ===")

# numpy
try:
    import numpy as np
    print(f"numpy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"numpy import failed: {e}")

# opencv
try:
    import cv2
    print(f"opencv imported successfully (version: {cv2.__version__})")
except ImportError as e:
    print(f"opencv import failed: {e}")

# scikit-learn
try:
    import sklearn
    print(f"sklearn imported successfully (version: {sklearn.__version__})")
except ImportError as e:
    print(f"sklearn import failed: {e}")

# BloodPressureEstimation
try:
    import BloodPressureEstimation
    print("BloodPressureEstimation imported successfully")
    print(f"  Location: {BloodPressureEstimation.__file__}")
except ImportError as e:
    print(f"BloodPressureEstimation import failed: {e}")

print("\\n=== Test Completed ===")
'''

    with open(test_script, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Created test script")


def test_runtime(runtime_dir):
    """ランタイムをテスト"""
    print("\n=== Testing Runtime ===")

    python_exe = runtime_dir / "python.exe"
    test_script = runtime_dir / "test_runtime.py"

    if not python_exe.exists():
        print("python.exe not found")
        return False

    if not test_script.exists():
        print("test script not found")
        return False

    # encodingsディレクトリの確認
    encodings_dir = runtime_dir / "Lib" / "encodings"
    if encodings_dir.exists():
        encodings_files = list(encodings_dir.glob("*.py"))
        print(f"encodings directory exists ({len(encodings_files)} files)")
    else:
        print("encodings directory missing")
        return False

    # テスト実行
    try:
        env = os.environ.copy()
        env['PYTHONHOME'] = str(runtime_dir.absolute())
        env['PYTHONPATH'] = str(runtime_dir.absolute())
        env['PYTHONUNBUFFERED'] = '1'

        result = subprocess.run(
            [str(python_exe), str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(runtime_dir),
            env=env,
            timeout=60
        )

        print("Test output:")
        print(result.stdout)

        if result.stderr:
            print("Test errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("Runtime test PASSED")
            return True
        else:
            print(f"Runtime test FAILED (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("Runtime test TIMEOUT")
        return False
    except Exception as e:
        print(f"Runtime test ERROR: {e}")
        return False


if __name__ == "__main__":
    try:
        runtime_dir = create_lightweight_runtime()
        print(f"\nSUCCESS: Lightweight runtime created at {runtime_dir}")

    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
