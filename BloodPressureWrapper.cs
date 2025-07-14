using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace BloodPressureEstimation
{
    public class BloodPressureWrapper : IDisposable
    {
        private Process pythonProcess;
        private string runtimePath;
        private bool isInitialized = false;

        public BloodPressureWrapper(string runtimeDirectory = "lightweight_runtime")
        {
            runtimePath = runtimeDirectory;
        }

        public bool Initialize(string modelDir = "models")
        {
            try
            {
                Console.WriteLine($"Initializing with runtime path: {runtimePath}");
                Console.WriteLine($"Current working directory: {Directory.GetCurrentDirectory()}");
                
                // Check if runtime directory exists
                if (!Directory.Exists(runtimePath))
                {
                    Console.WriteLine($"ERROR: Runtime directory not found: {runtimePath}");
                    return false;
                }

                // Check if python.exe exists
                string pythonExe = Path.Combine(runtimePath, "python.exe");
                if (!File.Exists(pythonExe))
                {
                    Console.WriteLine($"ERROR: Python executable not found: {pythonExe}");
                    return false;
                }

                // Check if BloodPressureEstimation.dll exists
                string dllPath = Path.Combine(runtimePath, "BloodPressureEstimation.dll");
                if (!File.Exists(dllPath))
                {
                    Console.WriteLine($"ERROR: BloodPressureEstimation.dll not found: {dllPath}");
                    return false;
                }

                // Test Python import first
                var testImportScript = $@"
import sys
import os
print('Python version:', sys.version)
print('Python executable:', sys.executable)
print('Current directory:', os.getcwd())
print('Python path:', sys.path)

try:
    import BloodPressureEstimation
    print('SUCCESS: BloodPressureEstimation imported')
    print('Module file:', BloodPressureEstimation.__file__)
except Exception as e:
    print('ERROR importing BloodPressureEstimation:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import numpy
    print('SUCCESS: NumPy imported')
except Exception as e:
    print('ERROR importing NumPy:', e)
    sys.exit(1)

try:
    import cv2
    print('SUCCESS: OpenCV imported')
except Exception as e:
    print('ERROR importing OpenCV:', e)
    sys.exit(1)

print('All imports successful')
";

                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"-c \"{testImportScript}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                using (var process = Process.Start(startInfo))
                {
                    var output = process.StandardOutput.ReadToEnd();
                    var error = process.StandardError.ReadToEnd();
                    process.WaitForExit();
                    
                    Console.WriteLine("Python test output:");
                    Console.WriteLine(output);
                    if (!string.IsNullOrEmpty(error))
                    {
                        Console.WriteLine("Python test errors:");
                        Console.WriteLine(error);
                    }
                    
                    isInitialized = process.ExitCode == 0;
                    Console.WriteLine($"Python test exit code: {process.ExitCode}");
                }

                return isInitialized;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Initialization failed with exception: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                return false;
            }
        }

        public async Task<string> StartBloodPressureAnalysisAsync(
            string requestId, int height, int weight, int sex, string moviePath)
        {
            if (!isInitialized)
                return "ERROR: Not initialized";

            try
            {
                var script = $@"
import BloodPressureEstimation
import sys

try:
    result = BloodPressureEstimation.StartBloodPressureAnalysisRequest(
        '{requestId}', {height}, {weight}, {sex}, '{moviePath}'
    )
    print(result)
except Exception as e:
    print(f'ERROR: {{e}}')
    import traceback
    traceback.print_exc()
";

                var startInfo = new ProcessStartInfo
                {
                    FileName = Path.Combine(runtimePath, "python.exe"),
                    Arguments = $"-c \"{script}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                using (var process = Process.Start(startInfo))
                {
                    var output = await process.StandardOutput.ReadToEndAsync();
                    var error = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    
                    if (!string.IsNullOrEmpty(error))
                    {
                        Console.WriteLine($"Python error: {error}");
                    }
                    
                    return output.Trim();
                }
            }
            catch (Exception ex)
            {
                return $"ERROR: {ex.Message}";
            }
        }

        public async Task<string> GetProcessingStatusAsync(string requestId)
        {
            if (!isInitialized)
                return "ERROR: Not initialized";

            try
            {
                var script = $@"
import BloodPressureEstimation

try:
    result = BloodPressureEstimation.GetProcessingStatus('{requestId}')
    print(result)
except Exception as e:
    print(f'ERROR: {{e}}')
";

                var startInfo = new ProcessStartInfo
                {
                    FileName = Path.Combine(runtimePath, "python.exe"),
                    Arguments = $"-c \"{script}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                using (var process = Process.Start(startInfo))
                {
                    var output = await process.StandardOutput.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    
                    return output.Trim();
                }
            }
            catch (Exception ex)
            {
                return $"ERROR: {ex.Message}";
            }
        }

        public async Task<bool> CancelBloodPressureAnalysisAsync(string requestId)
        {
            if (!isInitialized)
                return false;

            try
            {
                var script = $@"
import BloodPressureEstimation

try:
    result = BloodPressureEstimation.CancelBloodPressureAnalysis('{requestId}')
    print('OK' if result else 'FAILED')
except Exception as e:
    print('ERROR')
";

                var startInfo = new ProcessStartInfo
                {
                    FileName = Path.Combine(runtimePath, "python.exe"),
                    Arguments = $"-c \"{script}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                using (var process = Process.Start(startInfo))
                {
                    var output = await process.StandardOutput.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    
                    return output.Trim() == "OK";
                }
            }
            catch
            {
                return false;
            }
        }

        public async Task<string> GetVersionInfoAsync()
        {
            if (!isInitialized)
                return "ERROR: Not initialized";

            try
            {
                var script = @"
import BloodPressureEstimation

try:
    result = BloodPressureEstimation.GetVersionInfo()
    print(result)
except Exception as e:
    print(f'ERROR: {e}')
";

                var startInfo = new ProcessStartInfo
                {
                    FileName = Path.Combine(runtimePath, "python.exe"),
                    Arguments = $"-c \"{script}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                using (var process = Process.Start(startInfo))
                {
                    var output = await process.StandardOutput.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    
                    return output.Trim();
                }
            }
            catch (Exception ex)
            {
                return $"ERROR: {ex.Message}";
            }
        }

        public void Dispose()
        {
            pythonProcess?.Dispose();
        }
    }
} 
