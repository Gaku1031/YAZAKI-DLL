#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cython DLL Builder for Blood Pressure Estimation
Builds obfuscated DLL for C# integration
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path


def print_step(message):
    """Print a build step message"""
    print(f"\n{'='*60}")
    print(f"STEP: {message}")
    print(f"{'='*60}")


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    if description:
        print(f"Description: {description}")

    try:
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True)
        print("Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def check_dependencies():
    """Check if all required dependencies are available"""
    print_step("Checking Dependencies")

    required_packages = [
        'cython', 'numpy', 'opencv-python-headless',
        'scikit-learn', 'joblib', 'scipy', 'mediapipe'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"{package} is available")
        except ImportError:
            print(f"{package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def clean_build_artifacts():
    """Clean up previous build artifacts"""
    print_step("Cleaning Build Artifacts")

    artifacts_to_remove = [
        'build', 'dist', '__pycache__', '*.egg-info',
        '*.pyd', '*.so', '*.dll', '*.c', '*.cpp'
    ]

    for artifact in artifacts_to_remove:
        if os.path.exists(artifact):
            try:
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                else:
                    os.remove(artifact)
                print(f"Removed {artifact}")
            except Exception as e:
                print(f"Could not remove {artifact}: {e}")


def build_cython_extension():
    """Build the Cython extension with obfuscation"""
    print_step("Building Cython Extension")

    # Set environment variables for optimization
    os.environ['CFLAGS'] = '-O3 -DNDEBUG'
    os.environ['CXXFLAGS'] = '-O3 -DNDEBUG'

    # Build command
    build_command = f"{sys.executable} setup_cython.py build_ext --inplace"

    if not run_command(build_command, "Building Cython extension with obfuscation"):
        return False

    return True


def create_dll_wrapper():
    """Create the DLL wrapper for C# integration"""
    print_step("Creating DLL Wrapper")

    # Check if the main extension was built
    if platform.system() == "Windows":
        extension_name = "bp_estimation_cython*.pyd"
    else:
        extension_name = "bp_estimation_cython*.so"

    extension_files = list(Path('.').glob(extension_name))
    if not extension_files:
        print("Main Cython extension not found")
        return False

    print(f"Found extension: {extension_files[0]}")

    # Build the DLL wrapper
    wrapper_command = f"{sys.executable} setup_cython.py build_ext --inplace"
    if not run_command(wrapper_command, "Building DLL wrapper"):
        return False

    return True


def test_cython_dll():
    """Test the built Cython DLL"""
    print_step("Testing Cython DLL")

    try:
        # Import and test the built module
        import bp_estimation_cython
        print("Main module imported successfully")

        # Test basic functionality
        if hasattr(bp_estimation_cython, 'initialize_dll'):
            print("initialize_dll function found")
        if hasattr(bp_estimation_cython, 'get_version_info'):
            version = bp_estimation_cython.get_version_info()
            print(f"Version info: {version}")

        # Test DLL wrapper if available
        try:
            import dll_wrapper_cython
            print("DLL wrapper imported successfully")

            # Test DLL wrapper functions
            if hasattr(dll_wrapper_cython, 'InitializeDLL'):
                print("InitializeDLL function found")
            if hasattr(dll_wrapper_cython, 'GetVersionInfo'):
                version = dll_wrapper_cython.GetVersionInfo().decode('utf-8')
                print(f"DLL wrapper version: {version}")

        except ImportError:
            print("DLL wrapper not available (this is normal for non-Windows)")

        print("All tests passed")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False


def create_distribution():
    """Create distribution package"""
    print_step("Creating Distribution Package")

    # Create dist directory
    dist_dir = Path("dist_cython")
    dist_dir.mkdir(exist_ok=True)

    # Copy built extensions
    extensions_to_copy = [
        "bp_estimation_cython*.pyd",
        "bp_estimation_cython*.so",
        "dll_wrapper_cython*.pyd",
        "dll_wrapper_cython*.so"
    ]

    for pattern in extensions_to_copy:
        files = list(Path('.').glob(pattern))
        for file in files:
            shutil.copy2(file, dist_dir)
            print(f"Copied {file} to distribution")

    # Copy models directory if it exists
    if os.path.exists("models"):
        shutil.copytree("models", dist_dir / "models", dirs_exist_ok=True)
        print("Copied models directory")

    # Create README for distribution
    readme_content = f"""# Blood Pressure Estimation DLL - Cython Version

## Build Information
- Built with Cython for code obfuscation
- Platform: {platform.system()} {platform.machine()}
- Python Version: {sys.version}
- Build Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Features
- Code obfuscation through Cython compilation
- C# integration support
- Optimized performance
- Reduced file size

## Files Included
- Cython-compiled extension files
- Model files (if available)
- This README

## Usage
1. Place the extension files in your application directory
2. Import and use the functions as documented
3. For C# integration, use the DLL wrapper functions

## Code Obfuscation
This build uses Cython to compile Python code to C++, providing:
- Source code protection
- Improved performance
- Reduced file size
- Better integration with native code

## Support
For issues or questions, please refer to the project documentation.
"""

    with open(dist_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("Created distribution package")
    return True


def main():
    """Main build process"""
    print("Cython DLL Builder for Blood Pressure Estimation")
    print("Building obfuscated DLL for C# integration")

    # Check dependencies
    if not check_dependencies():
        print("\nDependency check failed")
        return False

    # Clean previous builds
    clean_build_artifacts()

    # Build Cython extension
    if not build_cython_extension():
        print("\nCython extension build failed")
        return False

    # Create DLL wrapper
    if not create_dll_wrapper():
        print("\nDLL wrapper creation failed")
        return False

    # Test the built DLL
    if not test_cython_dll():
        print("\nDLL test failed")
        return False

    # Create distribution
    if not create_distribution():
        print("\nDistribution creation failed")
        return False

    print("\n" + "="*60)
    print("Cython DLL build completed successfully!")
    print("="*60)
    print("\nBuild artifacts:")
    print("- Cython-compiled extension files")
    print("- DLL wrapper for C# integration")
    print("- Distribution package in dist_cython/")
    print("\nCode obfuscation features:")
    print("- Source code compiled to C++")
    print("- Python bytecode not visible")
    print("- Optimized performance")
    print("- Reduced file size")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
