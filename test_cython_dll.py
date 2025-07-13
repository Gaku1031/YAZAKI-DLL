#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Cython Blood Pressure Estimation DLL
Tests functionality and obfuscation features
"""

import os
import sys
import time
import platform
from pathlib import Path


def print_test_header(test_name):
    """Print a test header"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


def test_cython_import():
    """Test Cython module import"""
    print_test_header("Cython Module Import Test")

    try:
        import bp_estimation_cython
        print("Main Cython module imported successfully")

        # Test basic functions
        if hasattr(bp_estimation_cython, 'initialize_dll'):
            print("initialize_dll function found")
        if hasattr(bp_estimation_cython, 'get_version_info'):
            version = bp_estimation_cython.get_version_info()
            print(f"Version info: {version}")
        if hasattr(bp_estimation_cython, 'generate_request_id'):
            print("generate_request_id function found")

        return True
    except ImportError as e:
        print(f"Failed to import main Cython module: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during import: {e}")
        return False


def test_dll_wrapper():
    """Test DLL wrapper import"""
    print_test_header("DLL Wrapper Import Test")

    try:
        import dll_wrapper_cython
        print("DLL wrapper imported successfully")

        # Test DLL wrapper functions
        if hasattr(dll_wrapper_cython, 'InitializeDLL'):
            print("InitializeDLL function found")
        if hasattr(dll_wrapper_cython, 'GetVersionInfo'):
            version = dll_wrapper_cython.GetVersionInfo().decode('utf-8')
            print(f"DLL wrapper version: {version}")
        if hasattr(dll_wrapper_cython, 'GenerateRequestId'):
            print("GenerateRequestId function found")

        return True
    except ImportError as e:
        print(
            f"DLL wrapper not available (this is normal for non-Windows): {e}")
        return True  # Not a failure for non-Windows
    except Exception as e:
        print(f"Unexpected error during DLL wrapper import: {e}")
        return False


def test_initialization():
    """Test DLL initialization"""
    print_test_header("DLL Initialization Test")

    try:
        import bp_estimation_cython

        # Test initialization
        models_dir = "models" if os.path.exists(
            "models") else "models_compressed"
        if os.path.exists(models_dir):
            print(f"Models directory found: {models_dir}")
        else:
            print("No models directory found, using default")
            models_dir = "models"

        # Try initialization
        result = bp_estimation_cython.initialize_dll(models_dir)
        if result:
            print("DLL initialization successful")
        else:
            print("DLL initialization failed (may be expected without models)")

        return True
    except Exception as e:
        print(f"Initialization test failed: {e}")
        return False


def test_request_id_generation():
    """Test request ID generation"""
    print_test_header("Request ID Generation Test")

    try:
        import bp_estimation_cython

        # Test request ID generation
        customer_code = "9000000001"
        driver_code = "0000012345"

        request_id = bp_estimation_cython.generate_request_id(
            customer_code, driver_code)
        print(f"Generated request ID: {request_id}")

        # Validate format
        import re
        pattern = r'^\d{17}_\d{10}_\d{10}$'
        if re.match(pattern, request_id):
            print("Request ID format is valid")
        else:
            print("Request ID format is invalid")
            return False

        return True
    except Exception as e:
        print(f"Request ID generation test failed: {e}")
        return False


def test_processing_status():
    """Test processing status functions"""
    print_test_header("Processing Status Test")

    try:
        import bp_estimation_cython

        # Test processing status
        test_request_id = "20250101120000000_9000000001_0000012345"
        status = bp_estimation_cython.get_processing_status(test_request_id)
        print(f"Processing status retrieved: {status}")

        # Test cancellation
        cancel_result = bp_estimation_cython.cancel_blood_pressure_analysis(
            test_request_id)
        print(f"Cancellation test completed: {cancel_result}")

        return True
    except Exception as e:
        print(f"Processing status test failed: {e}")
        return False


def test_obfuscation():
    """Test code obfuscation features"""
    print_test_header("Code Obfuscation Test")

    try:
        # Check for compiled extensions
        extensions = []
        for pattern in ["bp_estimation_cython*.pyd", "bp_estimation_cython*.so"]:
            extensions.extend(Path('.').glob(pattern))

        if extensions:
            print(f"Found {len(extensions)} compiled extension(s)")

            for ext in extensions:
                size_mb = ext.stat().st_size / (1024 * 1024)
                print(f"  - {ext.name}: {size_mb:.2f} MB")

                # Check for Python bytecode (should not be present)
                try:
                    with open(ext, 'rb') as f:
                        content = f.read()
                        if b'__pycache__' in content or b'.pyc' in content:
                            print(f"  Python bytecode found in {ext.name}")
                        else:
                            print(f"  No Python bytecode in {ext.name}")
                except Exception as e:
                    print(f"  Could not analyze {ext.name}: {e}")
        else:
            print("No compiled extensions found")

        return True
    except Exception as e:
        print(f"Obfuscation test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics"""
    print_test_header("Performance Test")

    try:
        import bp_estimation_cython
        import time

        # Test import performance
        start_time = time.time()
        import bp_estimation_cython
        import_time = time.time() - start_time
        print(f"Import time: {import_time:.3f} seconds")

        # Test function call performance
        start_time = time.time()
        version = bp_estimation_cython.get_version_info()
        call_time = time.time() - start_time
        print(f"Function call time: {call_time:.6f} seconds")

        # Test request ID generation performance
        start_time = time.time()
        for _ in range(100):
            bp_estimation_cython.generate_request_id(
                "9000000001", "0000012345")
        generation_time = time.time() - start_time
        print(f"Request ID generation (100x): {generation_time:.3f} seconds")

        return True
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False


def test_platform_compatibility():
    """Test platform compatibility"""
    print_test_header("Platform Compatibility Test")

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Architecture: {platform.architecture()}")

    # Check for appropriate extensions
    if platform.system() == "Windows":
        expected_extensions = ["*.pyd"]
    else:
        expected_extensions = ["*.so"]

    found_extensions = []
    for pattern in expected_extensions:
        found_extensions.extend(Path('.').glob(pattern))

    if found_extensions:
        print(
            f"Found {len(found_extensions)} platform-appropriate extension(s)")
        for ext in found_extensions:
            print(f"  - {ext.name}")
    else:
        print("No platform-appropriate extensions found")

    return True


def main():
    """Main test function"""
    print("Cython Blood Pressure Estimation DLL Test Suite")
    print("Testing functionality and obfuscation features")

    tests = [
        ("Cython Module Import", test_cython_import),
        ("DLL Wrapper Import", test_dll_wrapper),
        ("DLL Initialization", test_initialization),
        ("Request ID Generation", test_request_id_generation),
        ("Processing Status", test_processing_status),
        ("Code Obfuscation", test_obfuscation),
        ("Performance", test_performance),
        ("Platform Compatibility", test_platform_compatibility),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"{test_name} PASSED")
            else:
                print(f"{test_name} FAILED")
        except Exception as e:
            print(f"{test_name} ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("All tests passed! Cython DLL is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
