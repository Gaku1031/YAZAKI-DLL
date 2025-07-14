#!/usr/bin/env python3
"""
軽量ランタイム診断スクリプト
軽量ランタイムの内容を確認して問題を特定
"""

import os
import sys
from pathlib import Path


def debug_runtime():
    """軽量ランタイムの内容を診断"""

    runtime_dir = Path("lightweight_runtime")
    if not runtime_dir.exists():
        print("ERROR: lightweight_runtime directory not found")
        return

    print("=== Lightweight Runtime Diagnostic ===")
    print(f"Runtime directory: {runtime_dir}")

    # 基本ファイルの確認
    print("\n=== Basic Files ===")
    basic_files = ["python.exe", "BloodPressureEstimation.dll"]
    for file in basic_files:
        file_path = runtime_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"[OK] {file} ({size} bytes)")
        else:
            print(f"[ERROR] {file} (NOT FOUND)")

    # Libディレクトリの確認
    lib_dir = runtime_dir / "Lib"
    if lib_dir.exists():
        print(f"\n=== Lib Directory ===")
        print(f"Lib directory exists: {lib_dir}")

        # 標準ライブラリファイルの確認
        stdlib_files = [
            "encodings/__init__.py",
            "encodings/utf_8.py",
            "codecs.py",
            "sys.py",
            "os.py",
            "json/__init__.py",
            "pickle.py"
        ]

        print("\n=== Standard Library Files ===")
        for file in stdlib_files:
            file_path = lib_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"[OK] {file} ({size} bytes)")
            else:
                print(f"[ERROR] {file} (NOT FOUND)")

        # encodingsディレクトリの詳細確認
        encodings_dir = lib_dir / "encodings"
        if encodings_dir.exists():
            print(f"\n=== Encodings Directory Contents ===")
            for item in encodings_dir.iterdir():
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  {item.name} ({size} bytes)")
                else:
                    print(f"  {item.name}/ (directory)")
        else:
            print(f"\n[ERROR] Encodings directory not found: {encodings_dir}")

    # site-packagesディレクトリの確認
    site_packages = lib_dir / "site-packages"
    if site_packages.exists():
        print(f"\n=== Site-packages Directory ===")
        print(f"Site-packages exists: {site_packages}")

        # 主要モジュールの確認
        modules = ["numpy", "cv2", "sklearn",
                   "scipy", "mediapipe", "joblib", "PIL"]
        for module in modules:
            module_path = site_packages / module
            if module_path.exists():
                if module_path.is_dir():
                    size = sum(f.stat().st_size for f in module_path.rglob(
                        '*') if f.is_file())
                    print(f"[OK] {module}/ ({size} bytes)")
                else:
                    size = module_path.stat().st_size
                    print(f"[OK] {module} ({size} bytes)")
            else:
                print(f"[ERROR] {module} (NOT FOUND)")
    else:
        print(f"\n[ERROR] Site-packages directory not found: {site_packages}")

    # モデルディレクトリの確認
    models_dir = runtime_dir / "models"
    if models_dir.exists():
        print(f"\n=== Models Directory ===")
        for item in models_dir.iterdir():
            if item.is_file():
                size = item.stat().st_size
                print(f"  {item.name} ({size} bytes)")
            else:
                print(f"  {item.name}/ (directory)")
    else:
        print(f"\n[ERROR] Models directory not found: {models_dir}")

    # 全体のサイズ計算
    total_size = sum(
        f.stat().st_size for f in runtime_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n=== Total Runtime Size ===")
    print(f"Total size: {total_size_mb:.2f} MB")

    # Python実行テスト
    print(f"\n=== Python Execution Test ===")
    python_exe = runtime_dir / "python.exe"
    if python_exe.exists():
        try:
            import subprocess
            result = subprocess.run([
                str(python_exe),
                "-c",
                "import sys; print('Python version:', sys.version); print('Python executable:', sys.executable)"
            ], capture_output=True, text=True, cwd=str(runtime_dir))

            if result.returncode == 0:
                print("[OK] Python execution successful")
                print("Output:")
                print(result.stdout)
            else:
                print("[ERROR] Python execution failed")
                print("Error:")
                print(result.stderr)
        except Exception as e:
            print(f"[ERROR] Python execution error: {e}")
    else:
        print("[ERROR] Python executable not found")


if __name__ == "__main__":
    debug_runtime()
