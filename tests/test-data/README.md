# Test Data

This directory contains test video files for the Blood Pressure Estimation DLL tests.

## Files

- `sample.webm` - Sample WebM video file for testing (if available)
- `dummy.webm` - Minimal dummy file for basic functionality testing

## Usage

The C# integration tests will automatically:
1. Look for `.webm` files in this directory
2. Use the first available file for testing
3. Create a dummy file if no real test data is available

## Adding Real Test Data

To add real test video data:
1. Place WebM format video files in this directory
2. Ensure files are 30-second duration, 1280x720 resolution (recommended)
3. Files should contain facial video suitable for rPPG analysis

## Note

For CI/CD purposes, the tests are designed to work with dummy data when real video files are not available.