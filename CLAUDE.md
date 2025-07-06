# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a blood pressure estimation DLL project for IKI Japan/Yazaki. The project creates a 32-bit DLL that estimates blood pressure from 30-second WebM video files using remote photoplethysmography (rPPG) techniques.

## Architecture

### Core Components

**DLL Interface Requirements:**
- **Input**: Video file path (WebM format), user parameters (height, weight, sex), request ID
- **Output**: Estimated systolic/diastolic blood pressure, PPG raw data (CSV format ~20KB), error information via callback
- **Processing**: Asynchronous analysis with callback-based result delivery

**Key Functions to Implement:**
- Blood pressure analysis request handler
- Processing interruption by request ID
- Processing status retrieval (`"none"` or `"processing"`)
- Version information retrieval
- Callback interface (IF004 specification)

### Reference Implementation

The `reference/` directory contains Python implementations that demonstrate the algorithms:

**bp-estimation-by-video.py:**
- Complete GUI application for video-based blood pressure estimation
- Implements POS (Plane Orthogonal to Skin) algorithm for rPPG signal extraction
- Handles both daytime (color) and nighttime (grayscale/infrared) video processing
- MediaPipe-based facial landmark detection and ROI extraction
- Real-time signal processing with bandpass filtering, PCA, and wavelet denoising
- 5-second segment averaging for blood pressure calculation
- CSV export functionality with RRI (R-R interval) data

**bp-estimation.py:**
- Machine learning approach using Random Forest regression
- Feature extraction from RRI data (mean, std, min, max)
- User demographic integration (BMI, sex)
- Pre-trained model loading for SBP/DBP prediction

### Data Flow

1. **Video Input**: 30-second WebM files (1280x720, 30fps, VP8 encoding, 2.5Mbps)
2. **Face Detection**: MediaPipe FaceMesh for facial landmark identification
3. **ROI Extraction**: Specific facial regions for pulse signal extraction
4. **Signal Processing**: 
   - Daytime: POS algorithm on RGB channels
   - Nighttime: PCA + wavelet denoising on grayscale intensity
5. **Blood Pressure Estimation**: RRI-based calculation with calibration parameters
6. **Output**: Callback with BP values, PPG data, and error information

## Technical Specifications

### Input Parameters
- `requestId`: Format `${yyyyMMddHHmmssfff}_${driverCode}`
- `height`: Integer (cm)
- `weight`: Integer (kg) 
- `sex`: 1=male, 2=female
- `measurementMoviePath`: Full path to WebM file

### Error Codes
- 1001: DLL not initialized
- 1002: Device connection failure
- 1003: Calibration incomplete
- 1004: Invalid input parameters
- 1005: Request during processing
- 1006: Internal processing error

### Performance Requirements
- 32-bit DLL compilation
- Asynchronous processing with callback mechanism
- CSV output approximately 20KB in size
- Support for concurrent request management

## Development Notes

### Video Processing Algorithms
- **POS Algorithm**: Used for color video processing, extracts pulse signals from RGB channels
- **MediaPipe Integration**: Face mesh detection with specific landmark indices for ROI definition
- **Signal Filtering**: Bandpass filtering (0.7-3.0 Hz) for heart rate range isolation
- **Peak Detection**: RRI calculation from detected pulse peaks for blood pressure estimation

### Machine Learning Components
- Pre-trained Random Forest models for SBP/DBP prediction
- Feature engineering from RRI statistics and user demographics
- Model files: `models/model_sbp.pkl` and `models/model_dbp.pkl`

### Dependencies (Reference Implementation)
- OpenCV for video processing
- MediaPipe for face detection
- NumPy/SciPy for signal processing
- scikit-learn for machine learning
- PyQt5 for GUI (reference only)
- pywt for wavelet transforms

## Implementation Strategy

The final DLL should be implemented in C/C++ based on the algorithms demonstrated in the Python reference code. Key considerations:

1. **Algorithm Translation**: Convert POS and signal processing algorithms from Python to C/C++
2. **MediaPipe Integration**: Use MediaPipe C++ API for face detection
3. **Callback Management**: Implement thread-safe callback mechanism
4. **Error Handling**: Comprehensive error reporting per specification
5. **Memory Management**: Efficient handling of video data and signal processing buffers
6. **Performance Optimization**: Real-time processing capabilities for 30-second video analysis
