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

# Check Visual Studio compiler
try {
    $clVersion = cl.exe 2>&1 | Select-String "Version"
    Write-Host "Visual C++ Compiler: $clVersion" -ForegroundColor Green
} catch {
    Write-Host "Visual C++ Compiler not found" -ForegroundColor Red
    Write-Host "Make sure Visual Studio environment is set up" -ForegroundColor Red
    exit 1
}

# Check required source files
$requiredFiles = @(
    "BloodPressureEstimation_Fixed.cpp",
    "BloodPressureEstimation_Fixed.h",
    "BloodPressureEstimation.def",
    "bp_estimation_simple.py"
)

Write-Host "`n2. Checking source files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "$file" -ForegroundColor Green
    } else {
        Write-Host "$file not found" -ForegroundColor Red
        Write-Host "Run 'python create_cpp_wrapper_dll.py' first" -ForegroundColor Red
        exit 1
    }
}

# Get Python configuration
Write-Host "`n3. Getting Python configuration..." -ForegroundColor Yellow
try {
    $pythonInclude = python -c "import sysconfig; print(sysconfig.get_path('include'))"
    $pythonPrefix = python -c "import sys; print(sys.base_prefix)"
    $pythonLibs = "$pythonPrefix\libs"
    $pythonVersion = python -c 'import sys; print("python" + str(sys.version_info.major) + str(sys.version_info.minor))'
    
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
        exit 1
    }
    
    Write-Host "Python configuration verified" -ForegroundColor Green
} catch {
    Write-Host "Failed to get Python configuration" -ForegroundColor Red
    exit 1
}

# Create output directory
Write-Host "`n4. Creating output directory..." -ForegroundColor Yellow
if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force
    Write-Host "Cleaned existing output directory" -ForegroundColor Green
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Host "Created output directory: $OutputDir" -ForegroundColor Green

# Compile the DLL
Write-Host "`n5. Compiling C++ DLL..." -ForegroundColor Yellow

$compileArgs = @(
    '/EHsc',                                    # Exception handling
    '/MD',                                      # Multi-threaded DLL runtime
    "/I`"$pythonInclude`"",                    # Python include directory
    '/DBLOODPRESSURE_EXPORTS',                 # Export macro
    '/LD',                                      # Create DLL
    'BloodPressureEstimation_Fixed.cpp',       # Source file
    '/link',                                    # Linker options start
    "/DEF:BloodPressureEstimation.def",        # Export definition file
    "/LIBPATH:`"$pythonLibs`"",                # Python library path
    "$pythonVersion.lib",                       # Python library
    "/OUT:$OutputDir\BloodPressureEstimation.dll"  # Output file
)

Write-Host "Compile command: cl.exe $($compileArgs -join ' ')" -ForegroundColor Cyan

try {
    $compileResult = & cl.exe $compileArgs 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful" -ForegroundColor Green
    } else {
        Write-Host "Compilation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Compiler output:" -ForegroundColor Yellow
        $compileResult | Write-Host
        exit 1
    }
} catch {
    Write-Host "Compilation error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Verify DLL was created
Write-Host "`n6. Verifying DLL..." -ForegroundColor Yellow
$dllPath = "$OutputDir\BloodPressureEstimation.dll"

if (Test-Path $dllPath) {
    $dllSize = (Get-Item $dllPath).Length
    $dllSizeMB = [math]::Round($dllSize / 1MB, 2)
    Write-Host "DLL created: $dllPath" -ForegroundColor Green
    Write-Host "  Size: $dllSizeMB MB" -ForegroundColor Cyan
    
    # Check exports using dumpbin if available
    try {
        Write-Host "`n7. Checking DLL exports..." -ForegroundColor Yellow
        $exports = dumpbin /exports $dllPath 2>&1
        
        $expectedExports = @(
            "InitializeDLL",
            "StartBloodPressureAnalysisRequest",
            "GetProcessingStatus", 
            "CancelBloodPressureAnalysis",
            "GetVersionInfo"
        )
        
        $exportText = $exports -join "`n"
        $foundExports = @()
        
        foreach ($export in $expectedExports) {
            if ($exportText -match $export) {
                $foundExports += $export
                Write-Host "  Export found: $export" -ForegroundColor Green
            } else {
                Write-Host "  Export missing: $export" -ForegroundColor Red
            }
        }
        
        if ($foundExports.Count -eq $expectedExports.Count) {
            Write-Host "All required exports found" -ForegroundColor Green
        } else {
            Write-Host "Missing exports detected" -ForegroundColor Red
            Write-Host "Full dumpbin output:" -ForegroundColor Yellow
            $exports | Write-Host
            exit 1
        }
    } catch {
        Write-Host "Warning: Could not verify exports (dumpbin not available)" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "DLL not found: $dllPath" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Build completed successfully ===" -ForegroundColor Green
Write-Host "DLL Location: $dllPath" -ForegroundColor Cyan
Write-Host "DLL Size: $dllSizeMB MB" -ForegroundColor Cyan
