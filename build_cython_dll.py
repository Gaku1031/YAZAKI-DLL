#!/usr/bin/env python3
"""
Build script to create a pure Windows DLL from Cython code
"""

import os
import sys
import subprocess
import shutil


def build_windows_dll():
    """Build a pure Windows DLL from Cython code"""

    print("Building Windows DLL from Cython...")

    # Clean previous builds
    for dir_name in ['build', 'dist', '__pycache__']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    # Remove any existing .pyd files
    for file in os.listdir('.'):
        if file.endswith('.pyd'):
            os.remove(file)

    # Use the existing setup_cython_dll.py to build
    print("Running setup_cython_dll.py...")
    result = subprocess.run([sys.executable, 'setup_cython_dll.py', 'build_ext', '--inplace'],
                            capture_output=True, text=True)

    if result.returncode != 0:
        print("Build failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print("Build completed successfully")

    # Find the built file and rename it to .dll
    built_files = []
    for root, dirs, files in os.walk('build'):
        for file in files:
            if file.startswith('BloodPressureEstimation') and file.endswith('.pyd'):
                built_files.append(os.path.join(root, file))

    if built_files:
        # Copy the .pyd file and rename it to .dll
        pyd_file = built_files[0]
        dll_file = 'BloodPressureEstimation.dll'

        shutil.copy2(pyd_file, dll_file)
        print(f"Created Windows DLL: {dll_file}")

        # Verify the DLL has the exported functions
        try:
            result = subprocess.run(['dumpbin', '/EXPORTS', dll_file],
                                    capture_output=True, text=True, shell=True)
            if 'InitializeDLL' in result.stdout:
                print("DLL exports verified")
            else:
                print("Warning: DLL exports not found")
        except:
            print("Could not verify DLL exports")

        return True
    else:
        print("No built files found")
        return False


if __name__ == "__main__":
    success = build_windows_dll()
    sys.exit(0 if success else 1)
