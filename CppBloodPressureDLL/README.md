# C++ Blood Pressure Estimation DLL

This directory contains the C++ implementation of the blood pressure estimation DLL. The DLL uses remote photoplethysmography (rPPG) techniques to estimate blood pressure from video data.

## Architecture

The C++ implementation consists of several key components:

### Core Components

1. **BloodPressureDLL.cpp/h** - Main DLL interface and exported functions
2. **rppg.cpp/h** - Remote photoplethysmography signal processing
3. **bp_estimator.cpp/h** - Blood pressure estimation using machine learning models
4. **peak_detect.cpp/h** - Peak detection for pulse signal analysis

### Key Features

- **Face Detection**: Uses OpenCV Haar cascades (with MediaPipe fallback support)
- **rPPG Signal Processing**: Implements POS (Plane Orthogonal to Skin) algorithm
- **Machine Learning**: Uses ONNX runtime for blood pressure prediction
- **Asynchronous Processing**: Thread-based processing with callback mechanism
- **CSV Export**: Generates detailed PPG data in CSV format

## Dependencies

### Required Libraries

- **OpenCV 4.x** - Computer vision and image processing
- **Eigen3** - Linear algebra operations
- **ONNX Runtime** - Machine learning model inference
- **MediaPipe** (optional) - Advanced face detection and landmarks

### Build Tools

- **CMake 3.15+** - Build system
- **Visual Studio 2019+** - C++ compiler (Windows)
- **vcpkg** - Package manager (recommended)

## Building the DLL

### Option 1: Using vcpkg (Recommended)

1. Install vcpkg:
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
```

2. Install dependencies:
```bash
vcpkg install opencv4:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install onnxruntime:x64-windows
```

3. Build the project:
```bash
cd CppBloodPressureDLL
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

### Option 2: Manual Installation

1. Download and install OpenCV, Eigen3, and ONNX Runtime manually
2. Set environment variables:
   - `OPENCV_DIR` - OpenCV installation directory
   - `EIGEN3_ROOT` - Eigen3 installation directory
   - `ONNXRUNTIME_ROOT_PATH` - ONNX Runtime installation directory

3. Build:
```bash
cd CppBloodPressureDLL
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Model Conversion

Before building, you need to convert the scikit-learn models to ONNX format:

```bash
cd CppBloodPressureDLL/models
python convert_to_onnx.py
```

This will create:
- `model_sbp.onnx` - Systolic blood pressure model
- `model_dbp.onnx` - Diastolic blood pressure model

## DLL Interface

The DLL exports the following functions:

### Core Functions

```cpp
// Initialize the DLL with model directory
int InitializeBP(const char* modelDir);

// Start blood pressure analysis
const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback);

// Get processing status
const char* GetProcessingStatus(const char* requestId);

// Cancel processing
int CancelBloodPressureAnalysis(const char* requestId);

// Get version information
const char* GetVersionInfo();

// Generate unique request ID
const char* GenerateRequestId();
```

### Callback Function

```cpp
typedef void(*BPCallback)(
    const char* requestId,
    int maxBloodPressure,
    int minBloodPressure,
    const char* measureRowData,
    const char* errorsJson
);
```

## Usage Example (C#)

```csharp
using System;
using System.Runtime.InteropServices;

public class BloodPressureDll
{
    [DllImport("BloodPressureDLL.dll")]
    public static extern int InitializeBP(string modelDir);

    [DllImport("BloodPressureDLL.dll")]
    public static extern string StartBloodPressureAnalysisRequest(
        string requestId, int height, int weight, int sex,
        string moviePath, BPCallback callback);

    // ... other imports

    public static void Main()
    {
        // Initialize DLL
        int result = InitializeBP("models");
        
        if (result == 1)
        {
            // Start analysis
            string error = StartBloodPressureAnalysisRequest(
                "test_request", 170, 70, 1, "video.webm", MyCallback);
            
            if (error == null)
            {
                Console.WriteLine("Analysis started successfully");
            }
        }
    }

    public static void MyCallback(string requestId, int sbp, int dbp, 
                                 string csvData, string errors)
    {
        Console.WriteLine($"Result: {sbp}/{dbp} mmHg");
        // Save CSV data if needed
    }
}
```

## Testing

### C# Test Application

The project includes a C# test application:

```bash
cd CppBloodPressureDLL/test
dotnet run --project CSharpTest.csproj
```

### Manual Testing

1. Build the DLL
2. Copy the DLL to the test directory
3. Ensure model files are available
4. Run the C# test application

## Input Requirements

### Video Format

- **Container**: WebM
- **Codec**: VP8
- **Duration**: 30 seconds
- **Frame Rate**: 30 FPS
- **Resolution**: 1280x720 (1MP)
- **Bitrate**: ~2.5 Mbps

### Input Parameters

- **Height**: Integer (cm)
- **Weight**: Integer (kg)
- **Sex**: 1 (male) or 2 (female)
- **Request ID**: Format `${yyyyMMddHHmmssfff}_${customerCode}_${driverCode}`

## Output Format

### Blood Pressure Values

- **Systolic**: Integer (mmHg), max 999
- **Diastolic**: Integer (mmHg), max 999

### CSV Data Format

```csv
Time(s),rPPG_Signal,Peak_Flag
0.000,0.123456,0
0.033,0.234567,0
0.067,0.345678,1
...
```

## Error Codes

- **1001**: DLL not initialized
- **1002**: Device connection failed
- **1003**: Calibration incomplete
- **1004**: Invalid input parameters
- **1005**: Request during processing
- **1006**: Internal processing error

## Performance Considerations

- **Memory Usage**: ~50-100MB during processing
- **Processing Time**: 10-30 seconds for 30-second video
- **Thread Safety**: Uses mutex for thread-safe operations
- **Model Size**: ONNX models are ~1-5MB each

## Algorithm Details

### rPPG Signal Processing

1. **Face Detection**: OpenCV Haar cascade or MediaPipe
2. **ROI Extraction**: Facial region of interest
3. **Skin Segmentation**: YCbCr color space filtering
4. **POS Algorithm**: Plane orthogonal to skin signal extraction
5. **Signal Filtering**: Bandpass filtering (0.7-3.0 Hz)

### Blood Pressure Estimation

1. **Peak Detection**: Find pulse peaks in rPPG signal
2. **RRI Calculation**: R-R interval extraction
3. **Feature Engineering**: Statistical features from RRI
4. **ML Prediction**: Random Forest models via ONNX Runtime

## GitHub Actions

The project includes automated builds:

```yaml
# .github/workflows/build-cpp-dll.yml
- Builds DLL on Windows x64
- Runs integration tests
- Creates distribution packages
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required libraries are installed
2. **Model Files**: Check that ONNX models are in the models directory
3. **Face Detection**: Verify OpenCV data files are available
4. **Memory Issues**: Increase available memory for large videos

### Debug Build

For debugging, build with Debug configuration:

```bash
cmake --build build --config Debug
```

## License

This project is part of the YAZAKI blood pressure estimation system for IKI Japan.