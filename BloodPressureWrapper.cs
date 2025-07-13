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
                // Pythonプロセスを起動
                var startInfo = new ProcessStartInfo
                {
                    FileName = Path.Combine(runtimePath, "python.exe"),
                    Arguments = $"-c \"import BloodPressureEstimation; print('OK')\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = runtimePath
                };

                using (var process = Process.Start(startInfo))
                {
                    process.WaitForExit();
                    isInitialized = process.ExitCode == 0;
                }

                return isInitialized;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Initialization failed: {ex.Message}");
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
