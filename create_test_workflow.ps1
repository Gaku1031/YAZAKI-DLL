# Create Blood Pressure Estimation Test Workflow
# This script generates the GitHub Actions workflow file

$workflowContent = @'
name: Test Blood Pressure Estimation with Real Video

on:
  workflow_dispatch:
    inputs:
      test_video:
        description: "Test video file path (relative to sample-data/)"
        required: true
        default: "sample_1M.webm"
      test_duration:
        description: "Test duration in seconds (0 for full video)"
        required: false
        default: "30"
        type: number
      performance_monitoring:
        description: "Enable detailed performance monitoring"
        required: false
        default: "true"
        type: boolean

env:
  TEST_VIDEO: ${{ github.event.inputs.test_video || 'sample_1M.webm' }}
  TEST_DURATION: ${{ github.event.inputs.test_duration || '30' }}
  PERFORMANCE_MONITORING: ${{ github.event.inputs.performance_monitoring || 'true' }}

jobs:
  test-blood-pressure-estimation:
    runs-on: windows-latest
    name: Test Blood Pressure Estimation with Performance Metrics

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup .NET
        uses: actions/setup-dotnet@v3
        with:
          dotnet-version: "6.0.x"

      - name: Verify required files exist
        run: |
          Write-Host "Verifying required files for blood pressure estimation test..." -ForegroundColor Yellow
          
          # Check main DLL
          if (!(Test-Path "BloodPressureDLL.dll")) {
            Write-Host "ERROR: BloodPressureDLL.dll not found in root directory" -ForegroundColor Red
            exit 1
          }
          $dllSize = [math]::Round((Get-Item "BloodPressureDLL.dll").Length / 1MB, 2)
          Write-Host "✓ BloodPressureDLL.dll found ($dllSize MB)" -ForegroundColor Green
          
          # Check critical dependency DLLs
          $criticalDlls = @(
            "opencv_core4.dll",
            "opencv_dnn4.dll", 
            "opencv_imgcodecs4.dll",
            "opencv_imgproc4.dll",
            "onnxruntime.dll"
          )
          
          foreach ($dll in $criticalDlls) {
            if (Test-Path $dll) {
              $size = [math]::Round((Get-Item $dll).Length / 1MB, 2)
              Write-Host "✓ $dll found ($size MB)" -ForegroundColor Green
            } else {
              Write-Host "ERROR: $dll not found" -ForegroundColor Red
              exit 1
            }
          }
          
          # Check model files
          $modelFiles = @(
            "CppBloodPressureDLL/models/systolicbloodpressure.onnx",
            "CppBloodPressureDLL/models/diastolicbloodpressure.onnx",
            "CppBloodPressureDLL/models/opencv_face_detector.pbtxt",
            "CppBloodPressureDLL/models/opencv_face_detector_uint8.pb"
          )
          
          foreach ($model in $modelFiles) {
            if (Test-Path $model) {
              $size = [math]::Round((Get-Item $model).Length / 1MB, 2)
              Write-Host "✓ $model found ($size MB)" -ForegroundColor Green
            } else {
              Write-Host "ERROR: $model not found" -ForegroundColor Red
              exit 1
            }
          }
          
          # Check header file
          if (Test-Path "CppBloodPressureDLL/include/BloodPressureDLL.h") {
            Write-Host "✓ BloodPressureDLL.h found" -ForegroundColor Green
          } else {
            Write-Host "ERROR: BloodPressureDLL.h not found" -ForegroundColor Red
            exit 1
          }
          
          Write-Host "All required files verified successfully!" -ForegroundColor Green
        shell: powershell

      - name: Create test application
        run: |
          Write-Host "Creating blood pressure estimation test application..." -ForegroundColor Yellow
          
          # Create test directory
          New-Item -ItemType Directory -Path "test_bp_estimation" -Force | Out-Null
          cd test_bp_estimation
          
          # Create C# test application
          $testCode = @'
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace BloodPressureTest
{
    public class BloodPressureEstimator
    {
        // DLL imports
        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CreateBloodPressureEstimator();
        
        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void DestroyBloodPressureEstimator(IntPtr estimator);
        
        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int EstimateBloodPressure(IntPtr estimator, string videoPath, 
            out double systolic, out double diastolic, out double confidence);
        
        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetLastError(IntPtr estimator, StringBuilder errorMessage, int maxLength);
        
        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetPerformanceMetrics(IntPtr estimator, 
            out double initializationTime, out double processingTime, out double totalTime,
            out int frameCount, out double fps);
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Blood Pressure Estimation Test ===");
            Console.WriteLine($"Test Video: {Environment.GetEnvironmentVariable("TEST_VIDEO")}");
            Console.WriteLine($"Test Duration: {Environment.GetEnvironmentVariable("TEST_DURATION")} seconds");
            Console.WriteLine($"Performance Monitoring: {Environment.GetEnvironmentVariable("PERFORMANCE_MONITORING")}");
            Console.WriteLine();
            
            // Performance monitoring setup
            var stopwatch = Stopwatch.StartNew();
            var process = Process.GetCurrentProcess();
            var initialMemory = process.WorkingSet64;
            
            try
            {
                // Copy DLLs to test directory
                Console.WriteLine("Setting up test environment...");
                CopyRequiredFiles();
                
                // Create estimator
                Console.WriteLine("Initializing blood pressure estimator...");
                var estimator = BloodPressureEstimator.CreateBloodPressureEstimator();
                
                if (estimator == IntPtr.Zero)
                {
                    Console.WriteLine("ERROR: Failed to create blood pressure estimator");
                    return;
                }
                
                // Get video path
                var videoPath = Path.Combine("..", "sample-data", Environment.GetEnvironmentVariable("TEST_VIDEO"));
                if (!File.Exists(videoPath))
                {
                    Console.WriteLine($"ERROR: Test video not found: {videoPath}");
                    return;
                }
                
                Console.WriteLine($"Processing video: {videoPath}");
                var fileInfo = new FileInfo(videoPath);
                Console.WriteLine($"Video size: {fileInfo.Length / 1024 / 1024:F2} MB");
                
                // Estimate blood pressure
                double systolic, diastolic, confidence;
                var result = BloodPressureEstimator.EstimateBloodPressure(estimator, videoPath, 
                    out systolic, out diastolic, out confidence);
                
                if (result == 0)
                {
                    // Get performance metrics
                    double initTime, procTime, totalTime;
                    int frameCount;
                    double fps;
                    BloodPressureEstimator.GetPerformanceMetrics(estimator, 
                        out initTime, out procTime, out totalTime, out frameCount, out fps);
                    
                    // Get memory usage
                    process.Refresh();
                    var finalMemory = process.WorkingSet64;
                    var memoryUsed = (finalMemory - initialMemory) / 1024 / 1024;
                    
                    // Get CPU usage (approximate)
                    var cpuTime = process.TotalProcessorTime;
                    var cpuUsage = (cpuTime.TotalMilliseconds / stopwatch.ElapsedMilliseconds) * 100;
                    
                    // Output results
                    Console.WriteLine();
                    Console.WriteLine("=== RESULTS ===");
                    Console.WriteLine($"Success: true");
                    Console.WriteLine($"Systolic Blood Pressure: {systolic:F1} mmHg");
                    Console.WriteLine($"Diastolic Blood Pressure: {diastolic:F1} mmHg");
                    Console.WriteLine($"Confidence: {confidence:F2}");
                    Console.WriteLine();
                    Console.WriteLine("=== PERFORMANCE METRICS ===");
                    Console.WriteLine($"Total Time: {totalTime:F2} seconds");
                    Console.WriteLine($"Initialization Time: {initTime:F2} seconds");
                    Console.WriteLine($"Processing Time: {procTime:F2} seconds");
                    Console.WriteLine($"Frame Count: {frameCount}");
                    Console.WriteLine($"Frames Per Second: {fps:F2}");
                    Console.WriteLine($"Memory Usage: {memoryUsed:F2} MB");
                    Console.WriteLine($"CPU Usage: {cpuUsage:F1}%");
                    Console.WriteLine($"Video File Size: {fileInfo.Length / 1024 / 1024:F2} MB");
                    Console.WriteLine();
                    
                    // Performance analysis
                    Console.WriteLine("=== PERFORMANCE ANALYSIS ===");
                    var processingEfficiency = (procTime / totalTime) * 100;
                    Console.WriteLine($"Processing Efficiency: {processingEfficiency:F1}%");
                    
                    var throughput = frameCount / totalTime;
                    Console.WriteLine($"Throughput: {throughput:F2} frames/second");
                    
                    var memoryEfficiency = fileInfo.Length / 1024 / 1024 / memoryUsed;
                    Console.WriteLine($"Memory Efficiency: {memoryEfficiency:F2} MB video per MB RAM");
                    
                    // Bottleneck analysis
                    Console.WriteLine();
                    Console.WriteLine("=== BOTTLENECK ANALYSIS ===");
                    if (initTime > procTime * 0.1)
                    {
                        Console.WriteLine("• Initialization time is significant (>10% of processing time)");
                        Console.WriteLine("  - Consider lazy initialization or caching");
                    }
                    
                    if (fps < 25)
                    {
                        Console.WriteLine("• Frame rate is below real-time threshold (<25 FPS)");
                        Console.WriteLine("  - Consider frame skipping or resolution reduction");
                    }
                    
                    if (memoryUsed > 1024)
                    {
                        Console.WriteLine("• High memory usage (>1GB)");
                        Console.WriteLine("  - Consider memory optimization or streaming");
                    }
                    
                    // Recommendations
                    Console.WriteLine();
                    Console.WriteLine("=== RECOMMENDATIONS ===");
                    if (processingEfficiency < 80)
                    {
                        Console.WriteLine("• Low processing efficiency detected");
                        Console.WriteLine("  - Optimize initialization and cleanup");
                    }
                    
                    if (throughput < 30)
                    {
                        Console.WriteLine("• Low throughput detected");
                        Console.WriteLine("  - Consider parallel processing or GPU acceleration");
                    }
                    
                    if (memoryEfficiency < 1.0)
                    {
                        Console.WriteLine("• Low memory efficiency detected");
                        Console.WriteLine("  - Consider memory pooling or streaming");
                    }
                    
                    Console.WriteLine();
                    Console.WriteLine("=== TEST COMPLETED SUCCESSFULLY ===");
                }
                else
                {
                    var errorMessage = new StringBuilder(1024);
                    BloodPressureEstimator.GetLastError(estimator, errorMessage, 1024);
                    Console.WriteLine($"ERROR: Blood pressure estimation failed: {errorMessage}");
                }
                
                // Cleanup
                BloodPressureEstimator.DestroyBloodPressureEstimator(estimator);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            finally
            {
                stopwatch.Stop();
                Console.WriteLine($"Total test time: {stopwatch.ElapsedMilliseconds / 1000:F2} seconds");
            }
        }
        
        static void CopyRequiredFiles()
        {
            var files = new[]
            {
                "BloodPressureDLL.dll",
                "opencv_core4.dll",
                "opencv_dnn4.dll", 
                "opencv_imgcodecs4.dll",
                "opencv_imgproc4.dll",
                "opencv_objdetect4.dll",
                "onnxruntime.dll",
                "abseil_dll.dll",
                "jpeg62.dll",
                "libgcc_s_seh-1.dll",
                "libgfortran-5.dll",
                "liblapack.dll",
                "liblzma.dll",
                "libpng16.dll",
                "libprotobuf.dll",
                "libquadmath-0.dll",
                "libsharpyuv.dll",
                "libwebp.dll",
                "libwebpdecoder.dll",
                "libwebpdemux.dll",
                "libwebpmux.dll",
                "libwinpthread-1.dll",
                "openblas.dll",
                "tiff.dll",
                "zlib1.dll"
            };
            
            foreach (var file in files)
            {
                if (File.Exists(Path.Combine("..", file)))
                {
                    File.Copy(Path.Combine("..", file), file, true);
                    Console.WriteLine($"Copied: {file}");
                }
                else
                {
                    Console.WriteLine($"Warning: {file} not found");
                }
            }
            
            // Copy models directory
            var modelsDir = Path.Combine("..", "CppBloodPressureDLL", "models");
            if (Directory.Exists(modelsDir))
            {
                if (Directory.Exists("models"))
                    Directory.Delete("models", true);
                Directory.CreateDirectory("models");
                
                foreach (var file in Directory.GetFiles(modelsDir))
                {
                    var fileName = Path.GetFileName(file);
                    File.Copy(file, Path.Combine("models", fileName), true);
                    Console.WriteLine($"Copied model: {fileName}");
                }
            }
        }
    }
}
'@
          
          # Create project file
          $projectFile = @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <PlatformTarget>x64</PlatformTarget>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
</Project>
'@
          
          # Write files
          [System.IO.File]::WriteAllText("Program.cs", $testCode)
          [System.IO.File]::WriteAllText("BloodPressureTest.csproj", $projectFile)
          
          Write-Host "Test application created successfully" -ForegroundColor Green
        shell: powershell

      - name: Build test application
        run: |
          cd test_bp_estimation
          dotnet restore
          dotnet build -c Release
          
          if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to build test application" -ForegroundColor Red
            exit 1
          }
          
          Write-Host "Test application built successfully" -ForegroundColor Green
        shell: powershell

      - name: Run blood pressure estimation test
        run: |
          cd test_bp_estimation
          
          # Set environment variables
          $env:TEST_VIDEO = "${{ env.TEST_VIDEO }}"
          $env:TEST_DURATION = "${{ env.TEST_DURATION }}"
          $env:PERFORMANCE_MONITORING = "${{ env.PERFORMANCE_MONITORING }}"
          
          # Create results directory
          New-Item -ItemType Directory -Path "results" -Force | Out-Null
          
          Write-Host "Running blood pressure estimation test..." -ForegroundColor Yellow
          Write-Host "Test video: $env:TEST_VIDEO" -ForegroundColor Cyan
          Write-Host "Test duration: $env:TEST_DURATION seconds" -ForegroundColor Cyan
          Write-Host "Performance monitoring: $env:PERFORMANCE_MONITORING" -ForegroundColor Cyan
          
          # Run test with output capture
          $output = dotnet run -c Release 2>&1
          $exitCode = $LASTEXITCODE
          
          # Save output to file
          $output | Out-File -FilePath "results/test_output.txt" -Encoding UTF8
          
          # Display output
          Write-Host "=== TEST OUTPUT ===" -ForegroundColor Green
          $output
          
          if ($exitCode -ne 0) {
            Write-Host "ERROR: Test failed with exit code $exitCode" -ForegroundColor Red
            exit 1
          }
          
          Write-Host "Test completed successfully" -ForegroundColor Green
        shell: powershell

      - name: Analyze test results
        run: |
          cd test_bp_estimation
          
          if (Test-Path "results/test_output.txt") {
            Write-Host "=== TEST RESULTS ANALYSIS ===" -ForegroundColor Green
            
            $output = Get-Content "results/test_output.txt" -Raw
            
            # Extract key metrics
            $systolic = [regex]::Match($output, "Systolic Blood Pressure: ([\d.]+)").Groups[1].Value
            $diastolic = [regex]::Match($output, "Diastolic Blood Pressure: ([\d.]+)").Groups[1].Value
            $confidence = [regex]::Match($output, "Confidence: ([\d.]+)").Groups[1].Value
            $totalTime = [regex]::Match($output, "Total Time: ([\d.]+)").Groups[1].Value
            $fps = [regex]::Match($output, "Frames Per Second: ([\d.]+)").Groups[1].Value
            $memoryUsed = [regex]::Match($output, "Memory Usage: ([\d.]+)").Groups[1].Value
            $cpuUsage = [regex]::Match($output, "CPU Usage: ([\d.]+)").Groups[1].Value
            
            Write-Host "Blood Pressure Results:" -ForegroundColor Yellow
            Write-Host "  Systolic: ${systolic} mmHg" -ForegroundColor Cyan
            Write-Host "  Diastolic: ${diastolic} mmHg" -ForegroundColor Cyan
            Write-Host "  Confidence: ${confidence}" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Performance Metrics:" -ForegroundColor Yellow
            Write-Host "  Total Time: ${totalTime} seconds" -ForegroundColor Cyan
            Write-Host "  FPS: ${fps}" -ForegroundColor Cyan
            Write-Host "  Memory Usage: ${memoryUsed} MB" -ForegroundColor Cyan
            Write-Host "  CPU Usage: ${cpuUsage}%" -ForegroundColor Cyan
            
            # Performance assessment
            Write-Host ""
            Write-Host "Performance Assessment:" -ForegroundColor Yellow
            
            if ($fps -and [double]$fps -ge 25) {
              Write-Host "  ✓ Frame rate is acceptable (≥25 FPS)" -ForegroundColor Green
            } elseif ($fps) {
              Write-Host "  ⚠ Frame rate is below threshold (<25 FPS)" -ForegroundColor Yellow
            }
            
            if ($memoryUsed -and [double]$memoryUsed -le 1024) {
              Write-Host "  ✓ Memory usage is reasonable (≤1GB)" -ForegroundColor Green
            } elseif ($memoryUsed) {
              Write-Host "  ⚠ High memory usage (>1GB)" -ForegroundColor Yellow
            }
            
            if ($cpuUsage -and [double]$cpuUsage -le 80) {
              Write-Host "  ✓ CPU usage is acceptable (≤80%)" -ForegroundColor Green
            } elseif ($cpuUsage) {
              Write-Host "  ⚠ High CPU usage (>80%)" -ForegroundColor Yellow
            }
            
            # Blood pressure validation
            Write-Host ""
            Write-Host "Blood Pressure Validation:" -ForegroundColor Yellow
            
            if ($systolic -and [double]$systolic -ge 80 -and [double]$systolic -le 200) {
              Write-Host "  ✓ Systolic pressure is within normal range (80-200 mmHg)" -ForegroundColor Green
            } elseif ($systolic) {
              Write-Host "  ⚠ Systolic pressure is outside normal range" -ForegroundColor Yellow
            }
            
            if ($diastolic -and [double]$diastolic -ge 40 -and [double]$diastolic -le 120) {
              Write-Host "  ✓ Diastolic pressure is within normal range (40-120 mmHg)" -ForegroundColor Green
            } elseif ($diastolic) {
              Write-Host "  ⚠ Diastolic pressure is outside normal range" -ForegroundColor Yellow
            }
            
            if ($confidence -and [double]$confidence -ge 0.7) {
              Write-Host "  ✓ Confidence level is good (≥0.7)" -ForegroundColor Green
            } elseif ($confidence) {
              Write-Host "  ⚠ Low confidence level (<0.7)" -ForegroundColor Yellow
            }
            
          } else {
            Write-Host "ERROR: Test output file not found" -ForegroundColor Red
          }
        shell: powershell

      - name: Upload test results
        uses: actions/upload-artifact@v4
        with:
          name: blood-pressure-test-results-${{ github.sha }}
          path: test_bp_estimation/results/
          retention-days: 30
          if-no-files-found: error

      - name: Upload test application
        uses: actions/upload-artifact@v4
        with:
          name: blood-pressure-test-app-${{ github.sha }}
          path: test_bp_estimation/bin/Release/net6.0/
          retention-days: 30
          if-no-files-found: error

      - name: Generate test report
        run: |
          Write-Host "=== BLOOD PRESSURE ESTIMATION TEST REPORT ===" -ForegroundColor Green
          Write-Host "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
          Write-Host "Commit: ${{ github.sha }}" -ForegroundColor Cyan
          Write-Host "Test Video: ${{ env.TEST_VIDEO }}" -ForegroundColor Cyan
          Write-Host "Test Duration: ${{ env.TEST_DURATION }} seconds" -ForegroundColor Cyan
          Write-Host ""
          Write-Host "Test Status: COMPLETED" -ForegroundColor Green
          Write-Host ""
          Write-Host "Files Generated:" -ForegroundColor Yellow
          Write-Host "- Test application: test_bp_estimation/bin/Release/net6.0/" -ForegroundColor Cyan
          Write-Host "- Test results: test_bp_estimation/results/" -ForegroundColor Cyan
          Write-Host "- Performance logs: Available in artifacts" -ForegroundColor Cyan
          Write-Host ""
          Write-Host "Next Steps:" -ForegroundColor Yellow
          Write-Host "1. Download test results artifact for detailed analysis" -ForegroundColor Cyan
          Write-Host "2. Review performance metrics and recommendations" -ForegroundColor Cyan
          Write-Host "3. Optimize based on bottleneck analysis" -ForegroundColor Cyan
          Write-Host "4. Run additional tests with different video files" -ForegroundColor Cyan
        shell: powershell
'@

# Create .github/workflows directory if it doesn't exist
if (!(Test-Path ".github/workflows")) {
    New-Item -ItemType Directory -Path ".github/workflows" -Force | Out-Null
}

# Write the workflow file
[System.IO.File]::WriteAllText(".github/workflows/test-blood-pressure-estimation.yml", $workflowContent)

Write-Host "Blood Pressure Estimation Test Workflow created successfully!" -ForegroundColor Green
Write-Host "File location: .github/workflows/test-blood-pressure-estimation.yml" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use this workflow:" -ForegroundColor Yellow
Write-Host "1. Commit and push the changes to GitHub" -ForegroundColor Cyan
Write-Host "2. Go to Actions tab in your repository" -ForegroundColor Cyan
Write-Host "3. Select 'Test Blood Pressure Estimation with Real Video'" -ForegroundColor Cyan
Write-Host "4. Click 'Run workflow' and configure parameters" -ForegroundColor Cyan
Write-Host "5. Monitor the test execution and download results" -ForegroundColor Cyan 
