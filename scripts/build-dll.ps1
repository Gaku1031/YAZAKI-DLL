# PowerShell script for building Blood Pressure Estimation DLL
# This script is designed to run in GitHub Actions Windows environment

param(
    [string]$Configuration = "Release",
    [string]$OutputDir = "build\dist"
)

Write-Host "=== Blood Pressure DLL Build Script ===" -ForegroundColor Green
Write-Host "Configuration: $Configuration"
Write-Host "Output Directory: $OutputDir"
Write-Host "Working Directory: $(Get-Location)"

# Set ErrorActionPreference to handle cl.exe behavior properly
$ErrorActionPreference = "Continue"

# Check prerequisites
Write-Host "`n1. Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found" -ForegroundColor Red
    exit 1
}

# Check Visual Studio environment - GitHub Actions specific approach
Write-Host "`n2. Checking Visual Studio environment..." -ForegroundColor Yellow

# In GitHub Actions, the msvc-dev-cmd action should have already set up the environment
# Let's verify the required environment variables are set
$requiredEnvVars = @('VCINSTALLDIR', 'INCLUDE', 'LIB', 'PATH')
$missingVars = @()

foreach ($varName in $requiredEnvVars) {
    $varValue = [Environment]::GetEnvironmentVariable($varName)
    if ([string]::IsNullOrEmpty($varValue)) {
        $missingVars += $varName
        Write-Host "$varName is NOT set" -ForegroundColor Red
    } else {
        Write-Host "$varName is set" -ForegroundColor Green
    }
}

# Check if cl.exe is available in PATH - using the same method as the workflow
Write-Host "`n3. Verifying C++ compiler..." -ForegroundColor Yellow

# cl.exe は引数なしで実行するとエラーコード1で終了するが、これは正常な動作
$clOutput = cl.exe 2>&1 | Out-String

if ($clOutput -match "Microsoft.*Compiler") {
    Write-Host "Visual C++ Compiler found and working" -ForegroundColor Green
    Write-Host "Compiler version info:" -ForegroundColor Cyan
    # Extract and display version info
    $versionLine = ($clOutput -split "`n" | Where-Object { $_ -match "Microsoft.*Compiler" })[0]
    Write-Host "  $versionLine" -ForegroundColor White
} else {
    Write-Host "Visual C++ Compiler not available in PATH" -ForegroundColor Red
    Write-Host "Compiler output: $clOutput" -ForegroundColor Red
    
    # デバッグ情報を出力
    Write-Host "PATH環境変数の確認:" -ForegroundColor Yellow
    $env:PATH -split ';' | Where-Object { $_ -like '*Visual Studio*' -or $_ -like '*MSVC*' } | ForEach-Object {
        Write-Host "  $_"
    }
    
    Write-Host "cl.exe の場所を検索:" -ForegroundColor Yellow
    Get-Command cl.exe -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Found at: $($_.Source)"
    }
    
    Write-Host "Environment variables:" -ForegroundColor Yellow
    Write-Host "VCINSTALLDIR: $env:VCINSTALLDIR"
    if ($env:INCLUDE) {
        Write-Host ("INCLUDE: " + $env:INCLUDE.Split(';')[0] + "...")
    }
    if ($env:LIB) {
        Write-Host ("LIB: " + $env:LIB.Split(';')[0] + "...")
    }
    
    exit 1
}

# Check required source files
$requiredFiles = @(
    "BloodPressureEstimation_Fixed.cpp",
    "BloodPressureEstimation_Fixed.h",
    "BloodPressureEstimation.def",
    "bp_estimation_simple.py"
)

Write-Host "`n4. Checking source files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        Write-Host "$file - Found ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "$file - NOT FOUND" -ForegroundColor Red
        Write-Host "Run 'python create_cpp_wrapper_dll.py' first" -ForegroundColor Red
        exit 1
    }
}

# Get Python configuration
Write-Host "`n5. Getting Python configuration..." -ForegroundColor Yellow
try {
    $pythonInclude = python -c "import sysconfig; print(sysconfig.get_path('include'))"
    $pythonPrefix = python -c "import sys; print(sys.base_prefix)"
    $pythonLibs = "$pythonPrefix\libs"
    $pythonVersionRaw = python -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')"
    
    # Remove any trailing whitespace/newlines
    $pythonVersion = $pythonVersionRaw.Trim()
    
    Write-Host "Python Include: $pythonInclude" -ForegroundColor Cyan
    Write-Host "Python Libs: $pythonLibs" -ForegroundColor Cyan
    Write-Host "Python Lib Name: $pythonVersion" -ForegroundColor Cyan
    
    # Verify paths exist
    if (-not (Test-Path $pythonInclude)) {
        Write-Host "Python include directory not found: $pythonInclude" -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path $pythonLibs)) {
        Write-Host "Python libs directory not found: $pythonLibs" -ForegroundColor Red
        exit 1
    }
    
    $pythonLibFile = "$pythonLibs\$pythonVersion.lib"
    if (-not (Test-Path $pythonLibFile)) {
        Write-Host "Python library file not found: $pythonLibFile" -ForegroundColor Red
        # List available .lib files for debugging
        Write-Host "Available .lib files in ${pythonLibs}:" -ForegroundColor Yellow
        $availableLibs = Get-ChildItem -Path $pythonLibs -Filter "*.lib" | ForEach-Object { 
            Write-Host "  $($_.Name)" -ForegroundColor Cyan
            $_.Name
        }
        
        # Try to find the correct Python library file
        $correctLib = $null
        if ($availableLibs -contains "python311.lib") {
            $correctLib = "python311"
            Write-Host "Using python311.lib" -ForegroundColor Green
        } elseif ($availableLibs -contains "python3.lib") {
            $correctLib = "python3"
            Write-Host "Using python3.lib" -ForegroundColor Green
        } else {
            Write-Host "No suitable Python library found" -ForegroundColor Red
            exit 1
        }
        
        # Update pythonVersion to use the correct library
        $pythonVersion = $correctLib
        $pythonLibFile = "$pythonLibs\$pythonVersion.lib"
        
        if (-not (Test-Path $pythonLibFile)) {
            Write-Host "Selected Python library file still not found: $pythonLibFile" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "Python configuration verified successfully" -ForegroundColor Green
    Write-Host "Final Python Lib Name: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Failed to get Python configuration: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Python executable path: $(Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)" -ForegroundColor Yellow
    exit 1
}

# Create output directory
Write-Host "`n6. Creating output directory..." -ForegroundColor Yellow
if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force
    Write-Host "Cleaned existing output directory" -ForegroundColor Green
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Host "Created output directory: $OutputDir" -ForegroundColor Green

# Compile the DLL
Write-Host "`n7. Compiling C++ DLL..." -ForegroundColor Yellow

$compileArgs = @(
    '/nologo',                                  # Suppress startup banner
    '/EHsc',                                    # Exception handling
    '/MD',                                      # Multi-threaded DLL runtime
    '/O2',                                      # Optimization
    "/I`"$pythonInclude`"",                    # Python include directory
    '/DBLOODPRESSURE_EXPORTS',                 # Export macro
    '/LD',                                      # Create DLL
    'BloodPressureEstimation_Fixed.cpp',       # Source file
    '/link',                                    # Linker options start
    '/nologo',                                  # Suppress linker banner
    "/DEF:BloodPressureEstimation.def",        # Export definition file
    "/LIBPATH:`"$pythonLibs`"",                # Python library path
    "$pythonVersion.lib",                       # Python library
    "/OUT:$OutputDir\BloodPressureEstimation.dll"  # Output file
)

Write-Host "Compile command:" -ForegroundColor Cyan
Write-Host "cl.exe $($compileArgs -join ' ')" -ForegroundColor White
Write-Host "`nCurrent directory: $(Get-Location)" -ForegroundColor Cyan

# Show relevant PATH entries for debugging
$relevantPaths = $env:PATH.Split(';') | Where-Object { $_ -like '*Visual Studio*' -or $_ -like '*VC*' -or $_ -like '*MSVC*' }
if ($relevantPaths) {
    Write-Host "Relevant PATH entries:" -ForegroundColor Cyan
    $relevantPaths | Select-Object -First 3 | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
}

try {
    # Run the compilation
    $compileResult = & cl.exe $compileArgs 2>&1
    $compileOutput = $compileResult | Out-String
    
    Write-Host "`nCompiler output:" -ForegroundColor Yellow
    Write-Host $compileOutput
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful" -ForegroundColor Green
    } else {
        Write-Host "Compilation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        
        # Additional debugging information
        Write-Host "`nDebugging information:" -ForegroundColor Yellow
        Write-Host "Files in current directory:" -ForegroundColor Cyan
        Get-ChildItem -Name | ForEach-Object { Write-Host "  $_" }
        
        exit 1
    }
} catch {
    Write-Host "Compilation error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Verify DLL was created
Write-Host "`n8. Verifying DLL..." -ForegroundColor Yellow
$dllPath = "$OutputDir\BloodPressureEstimation.dll"

if (Test-Path $dllPath) {
    $dllSize = (Get-Item $dllPath).Length
    $dllSizeMB = [math]::Round($dllSize / 1MB, 2)
    Write-Host "DLL created: $dllPath" -ForegroundColor Green
    Write-Host "  Size: $dllSizeMB MB" -ForegroundColor Cyan
    
    # List all files in output directory
    Write-Host "`nFiles in output directory:" -ForegroundColor Cyan
    Get-ChildItem $OutputDir | ForEach-Object { 
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $($_.Name) - $sizeMB MB" -ForegroundColor White
    }
    
    # Check exports using dumpbin if available
    try {
        Write-Host "`n9. Checking DLL exports..." -ForegroundColor Yellow
        $exports = & dumpbin /exports $dllPath 2>&1 | Out-String
        
        $expectedExports = @(
            "InitializeDLL",
            "StartBloodPressureAnalysisRequest",
            "GetProcessingStatus", 
            "CancelBloodPressureAnalysis",
            "GetVersionInfo"
        )
        
        $foundExports = @()
        
        foreach ($export in $expectedExports) {
            if ($exports -match $export) {
                $foundExports += $export
                Write-Host "  Export found: $export" -ForegroundColor Green
            } else {
                Write-Host "  Export missing: $export" -ForegroundColor Red
            }
        }
        
        if ($foundExports.Count -eq $expectedExports.Count) {
            Write-Host "All required exports found" -ForegroundColor Green
        } else {
            Write-Host "Missing exports detected. Full export list:" -ForegroundColor Yellow
            Write-Host $exports
        }
    } catch {
        Write-Host "Warning: Could not verify exports (dumpbin not available): $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "DLL not found: $dllPath" -ForegroundColor Red
    Write-Host "Files in output directory:" -ForegroundColor Yellow
    if (Test-Path $OutputDir) {
        Get-ChildItem $OutputDir | ForEach-Object { Write-Host "  $($_.Name)" }
    }
    
    # Show current directory contents for debugging
    Write-Host "Files in current directory:" -ForegroundColor Yellow
    Get-ChildItem | ForEach-Object { Write-Host "  $($_.Name)" }
    
    exit 1
}

Write-Host "`n=== Build completed successfully ===" -ForegroundColor Green
Write-Host "DLL Location: $dllPath" -ForegroundColor Cyan
Write-Host "DLL Size: $dllSizeMB MB" -ForegroundColor Cyan
