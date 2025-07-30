param(
    [string]$ConfigName,
    [string]$Description,
    [string]$SizeMB,
    [string]$GitHubRunId
)

$packageInfo = @"
# BloodPressureDLL Optimized Package

## Configuration: $ConfigName
**Description**: $Description
**Build Date**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**GitHub Run ID**: $GitHubRunId

## Size Optimization
- **Original Size**: 60MB (opencv480_world.dll)
- **Optimized Size**: $SizeMB MB
- **Size Reduction**: $([math]::Round((60 - [double]$SizeMB) / 60 * 100, 1))%

## Included Files
- BloodPressureDLL.dll (main library)
- Individual OpenCV libraries (if applicable)
- dlib19.24.0_release_64bit_msvc1929.dll
- onnxruntime.dll
- Models directory (ONNX, dlib, OpenCV models)
- Include directory (header files)
- Documentation (README.md, INTEGRATION_GUIDE.md)

## Usage Instructions
1. Extract all files to your application directory
2. Ensure all DLL files are in the same directory as your executable
3. Copy the models directory to your application directory
4. Include the header files in your project

## Dependencies
This package includes all necessary runtime dependencies:
- OpenCV individual libraries (core, imgproc, imgcodecs, objdetect, dnn)
- dlib face detection library
- ONNX Runtime for machine learning inference

## Performance Optimizations
- GitHub Actions optimized compilation (/O2 /GL)
- Link-time code generation (/LTCG)
- Memory reuse optimizations
- dlib-based face detection (faster than OpenCV DNN)
- Individual OpenCV libraries (smaller than world library)

## Support
For integration support, see INTEGRATION_GUIDE.md
"@

$downloadReadme = @"
# BloodPressureDLL - Optimized Download Package

## Quick Start
1. Download this package
2. Extract to your project directory
3. Copy all DLL files to your executable directory
4. Include the header files in your project
5. Copy the models directory to your application directory

## What's Included
- **BloodPressureDLL.dll**: Main blood pressure estimation library
- **OpenCV Libraries**: Individual optimized libraries (not the 60MB world library)
- **dlib Library**: High-performance face detection
- **ONNX Runtime**: Machine learning inference engine
- **Models**: Pre-trained models for face detection and blood pressure estimation
- **Headers**: C++ header files for integration
- **Documentation**: Integration guide and usage instructions

## Size Comparison
- **Original**: 60MB (opencv480_world.dll)
- **This Package**: $SizeMB MB
- **Savings**: $([math]::Round((60 - [double]$SizeMB) / 60 * 100, 1))% smaller

## Configuration: $ConfigName
$Description

## Build Information
- **Build Date**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
- **GitHub Run ID**: $GitHubRunId
- **Optimization Level**: GitHub Actions optimized
- **Compiler**: Visual Studio 2022 with /O2 /GL flags

## Integration
See INTEGRATION_GUIDE.md for detailed integration instructions.
"@

# パッケージ情報ファイルを出力
$packageInfo | Out-File -FilePath "package_$ConfigName\PACKAGE_INFO.md" -Encoding UTF8
$downloadReadme | Out-File -FilePath "package_$ConfigName\README_DOWNLOAD.md" -Encoding UTF8

Write-Host "Package info files created for $ConfigName" 
