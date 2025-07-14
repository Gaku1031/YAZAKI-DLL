using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Linq; // Added for .Take()
using System.Collections.Generic; // Added for BitConverter

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
                    Console.WriteLine("Available directories:");
                    foreach (var dir in Directory.GetDirectories("."))
                    {
                        Console.WriteLine($"  {dir}");
                    }
                    return false;
                }

                // List runtime directory contents
                Console.WriteLine("Runtime directory contents:");
                foreach (var item in Directory.GetFileSystemEntries(runtimePath))
                {
                    var info = new FileInfo(item);
                    Console.WriteLine($"  {Path.GetFileName(item)} ({(info.Exists ? info.Length : 0)} bytes)");
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

                // Test Python import first with detailed output
                var testImportScript = @"
import sys
import os
print('Python version:', sys.version)
print('Python executable:', sys.executable)
print('Current directory:', os.getcwd())
print('Python path:', sys.path)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
print('Updated Python path:', sys.path)

# List files in current directory
print('Files in current directory:')
for item in os.listdir(current_dir):
    print('  ' + item)

try:
    import BloodPressureEstimation
    print('SUCCESS: BloodPressureEstimation imported')
    print('Module file:', BloodPressureEstimation.__file__)
    
    # Test basic functionality
    try:
        version_info = BloodPressureEstimation.get_version_info()
        print('Version info:', version_info)
    except Exception as e:
        print('WARNING: Could not call get_version_info:', e)
        
    try:
        request_id = BloodPressureEstimation.generate_request_id()
        print('Generated request ID:', request_id)
    except Exception as e:
        print('WARNING: Could not call generate_request_id:', e)
        
except Exception as e:
    print('ERROR importing BloodPressureEstimation:', e)
    print('Error type:', type(e).__name__)
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import numpy
    print('SUCCESS: NumPy imported')
    print('NumPy version:', numpy.__version__)
except Exception as e:
    print('ERROR importing NumPy:', e)
    sys.exit(1)

try:
    import cv2
    print('SUCCESS: OpenCV imported')
    print('OpenCV version:', cv2.__version__)
except Exception as e:
    print('ERROR importing OpenCV:', e)
    sys.exit(1)

try:
    import sklearn
    print('SUCCESS: scikit-learn imported')
    print('scikit-learn version:', sklearn.__version__)
except Exception as e:
    print('ERROR importing scikit-learn:', e)
    sys.exit(1)

try:
    import mediapipe
    print('SUCCESS: MediaPipe imported')
except Exception as e:
    print('ERROR importing MediaPipe:', e)
    sys.exit(1)

try:
    import joblib
    print('SUCCESS: joblib imported')
except Exception as e:
    print('ERROR importing joblib:', e)
    sys.exit(1)

try:
    from PIL import Image
    print('SUCCESS: PIL imported')
except Exception as e:
    print('ERROR importing PIL:', e)
    sys.exit(1)

print('All imports successful')
";

                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"-c \"{testImportScript.Replace("\"", "\\\"")}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                Console.WriteLine($"Running Python test with: {startInfo.FileName} {startInfo.Arguments}");

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
                
                if (!isInitialized)
                {
                    Console.WriteLine("Python import test failed. Detailed analysis:");
                    Console.WriteLine("1. Check if all required modules are available");
                    Console.WriteLine("2. Check if BloodPressureEstimation.dll is properly copied");
                    Console.WriteLine("3. Check if Python path is correctly set");
                    Console.WriteLine("4. Check if working directory is correct");
                    
                    // Check runtime directory contents
                    Console.WriteLine("Runtime directory contents:");
                    try
                    {
                        var runtimeFiles = Directory.GetFileSystemEntries(runtimePath);
                        foreach (var file in runtimeFiles)
                        {
                            var info = new FileInfo(file);
                            Console.WriteLine($"  {Path.GetFileName(file)} ({(info.Exists ? info.Length : 0)} bytes)");
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"Error listing runtime contents: {e.Message}");
                    }
                    
                    // Check if BloodPressureEstimation.dll exists and is readable
                    var dllPath = Path.Combine(runtimePath, "BloodPressureEstimation.dll");
                    if (File.Exists(dllPath))
                    {
                        var dllInfo = new FileInfo(dllPath);
                        Console.WriteLine($"BloodPressureEstimation.dll found: {dllInfo.Length} bytes");
                        
                        // Try to read the DLL file
                        try
                        {
                            var dllBytes = File.ReadAllBytes(dllPath);
                            Console.WriteLine($"DLL file is readable, first 100 bytes: {BitConverter.ToString(dllBytes.Take(100).ToArray())}");
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine($"Error reading DLL file: {e.Message}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("BloodPressureEstimation.dll NOT FOUND");
                    }
                    
                    // Additional diagnostic: try to run a simple Python test
                    Console.WriteLine("Running additional diagnostic test...");
                    var simpleTest = @"
import sys
import os
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Current directory:', os.getcwd())
print('Python path:', sys.path)
print('Available modules:')
for module in ['numpy', 'cv2', 'sklearn', 'mediapipe', 'joblib', 'PIL']:
    try:
        __import__(module)
        print(f'  {module}: OK')
    except ImportError as e:
        print(f'  {module}: FAILED - {e}')
print('Trying to import BloodPressureEstimation...')
try:
    import BloodPressureEstimation
    print('BloodPressureEstimation: OK')
    print('Module file:', BloodPressureEstimation.__file__)
except ImportError as e:
    print('BloodPressureEstimation: FAILED -', e)
    import traceback
    traceback.print_exc()
";
                    
                    var diagnosticStartInfo = new ProcessStartInfo
                    {
                        FileName = Path.Combine(runtimePath, "python.exe"),
                        Arguments = $"-c \"{simpleTest.Replace("\"", "\\\"")}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = runtimePath
                    };
                    
                    using (var diagnosticProcess = Process.Start(diagnosticStartInfo))
                    {
                        var diagnosticOutput = diagnosticProcess.StandardOutput.ReadToEnd();
                        var diagnosticError = diagnosticProcess.StandardError.ReadToEnd();
                        diagnosticProcess.WaitForExit();
                        
                        Console.WriteLine("Diagnostic test output:");
                        Console.WriteLine(diagnosticOutput);
                        if (!string.IsNullOrEmpty(diagnosticError))
                        {
                            Console.WriteLine("Diagnostic test errors:");
                            Console.WriteLine(diagnosticError);
                        }
                    }
                }
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
                    Arguments = $"-c \"{script.Replace("\"", "\\\"")}\"",
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
                    Arguments = $"-c \"{script.Replace("\"", "\\\"")}\"",
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
                    Arguments = $"-c \"{script.Replace("\"", "\\\"")}\"",
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
                    Arguments = $"-c \"{script.Replace("\"", "\\\"")}\"",
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
