#!/usr/bin/env python3
"""
Pythonランタイムテストスクリプト
軽量ランタイムが正しく動作するかをテスト
"""

import sys
import os

print("=== Python Runtime Test ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test basic imports
print("\n=== Testing Basic Imports ===")

try:
    import encodings
    print("SUCCESS: encodings imported")
except Exception as e:
    print(f"ERROR: Failed to import encodings: {e}")
    sys.exit(1)

try:
    import sys
    print("SUCCESS: sys imported")
except Exception as e:
    print(f"ERROR: Failed to import sys: {e}")
    sys.exit(1)

try:
    import os
    print("SUCCESS: os imported")
except Exception as e:
    print(f"ERROR: Failed to import os: {e}")
    sys.exit(1)

try:
    import json
    print("SUCCESS: json imported")
except Exception as e:
    print(f"ERROR: Failed to import json: {e}")
    sys.exit(1)

try:
    import pickle
    print("SUCCESS: pickle imported")
except Exception as e:
    print(f"ERROR: Failed to import pickle: {e}")
    sys.exit(1)

# Test required modules
print("\n=== Testing Required Modules ===")

try:
    import numpy as np
    print("SUCCESS: NumPy imported")
except Exception as e:
    print(f"ERROR: Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import cv2
    print("SUCCESS: OpenCV imported")
except Exception as e:
    print(f"ERROR: Failed to import OpenCV: {e}")
    sys.exit(1)

try:
    import sklearn
    print("SUCCESS: scikit-learn imported")
except Exception as e:
    print(f"ERROR: Failed to import scikit-learn: {e}")
    sys.exit(1)

try:
    import mediapipe
    print("SUCCESS: MediaPipe imported")
except Exception as e:
    print(f"ERROR: Failed to import MediaPipe: {e}")
    sys.exit(1)

# Test Cython module
print("\n=== Testing Cython Module ===")

try:
    import BloodPressureEstimation
    print("SUCCESS: BloodPressureEstimation imported")
    print(f"Module file: {BloodPressureEstimation.__file__}")
except Exception as e:
    print(f"ERROR: Failed to import BloodPressureEstimation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All Python Tests Passed! ===")
