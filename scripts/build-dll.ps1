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

# Check Visual Studio environment - GitHub Actions specific approach
Write-Host "`n2. Checking Visual Studio environment..." -ForegroundColor Yellow

# In GitHub Actions, the msvc-dev-cmd action should have already set up the environment
# Let's verify the required environment variables are set
$requiredEnvVars = @('VCINSTALLDIR', 'VS_ROOT', 'INCLUDE', 'LIB', 'PATH')
$missingVars = @()

foreach ($varName in $requiredEnvVars) {
    $varValue = [Environment]::GetEnvironmentVariable($varName)
    if ([string]::IsNullOrEmpty($varValue)) {
        $missingVars += $varName
    } else {
        Write-Host "$varName is set" -ForegroundColor Green
    }
}

# Check if cl.exe is available in PATH
try {
    $clVersion = & cl.exe 2>&1 | Out-String
    if ($clVersion -match "Microsoft.*Compiler") {
        Write-Host "Visual C++ Compiler found and working" -ForegroundColor Green
    } else {
        throw "Compiler check failed"
    }
} catch {
    Write-Host "Visual C++ Compiler not available in PATH" -ForegroundColor Red
    
    # Try to manually locate and set up the compiler
    Write-Host "Attempting to locate Visual Studio tools..." -ForegroundColor Yellow
    
    # Common paths for Visual Studio 2022 in GitHub Actions
    $vsPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        "${env:VCINSTALLDIR}Tools\MSVC"
    )
    
    $compilerFound = $false
    foreach ($basePath in $vsPaths) {
        if (Test-Path $basePath) {
            Write-Host "Checking VS path: $basePath" -ForegroundColor Cyan
            $versions = Get-ChildItem -Path $basePath -Directory | Sort-Object Name -Descending
            
            foreach ($version in $versions) {
                $clPath = Join-Path $version.FullName "bin\Hostx64\x64\cl.exe"
                if (Test-Path $clPath) {
                    $toolsDir = $version.FullName
                    $binDir = Join-Path $toolsDir "bin\Hostx64\x64"
                    
                    # Add compiler to PATH
                    $env:PATH = "$binDir;$env:PATH"
                    
                    # Set required environment variables
                    $env:INCLUDE = "$(Join-Path $toolsDir 'include');$(Join-Path $toolsDir 'ATLMFC\include');$env:INCLUDE"
                    $env:LIB = "$(Join-Path $toolsDir 'lib\x64');$(Join-Path $toolsDir 'ATLMFC\lib\x64');$env:LIB"
                    
                    Write-Host "Found and configured compiler at: $clPath" -ForegroundColor Green
                    $compilerFound = $true
                    break
                }
            }
            if ($compilerFound) { break }
        }
    }
    
    if (-not $compilerFound) {
        Write-Host "Could not locate Visual Studio compiler" -ForegroundColor Red
        Write-Host "Environment debugging information:" -ForegroundColor Yellow
        Write-Host "VCINSTALLDIR: $env:VCINSTALLDIR"
        Write-Host "VS_ROOT: $env:VS_ROOT"
        Write-Host "PATH contains cl.exe: $($env:PATH -like '*cl.exe*')"
        
        # List available Visual Studio installations
        Write-Host "`nSearching for Visual Studio installations..." -ForegroundColor Yellow
        Get-ChildItem -Path "${env:ProgramFiles}\Microsoft Visual Studio" -Recurse -Name "cl.exe" -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object {
            Write-Host "Found: $_" -ForegroundColor Cyan
        }
        exit 1
    }
}

# Verify compiler is working
try {
    $testCompile = & cl.exe 2>&1 | Out-String
    if ($testCompile -match "Microsoft.*Compiler") {
        Write-Host "Compiler verification successful" -ForegroundColor Green
    }
} catch {
    Write-Host "Compiler verification failed" -ForegroundColor Red
    exit 1
}

# Check required source files
$requiredFiles = @(
    "BloodPressureEstimation_Fixed.cpp",
    "BloodPressureEstimation_Fixed.h",
    "BloodPressureEstimation.def",
    "bp_estimation_simple.py"
)

Write-Host "`n3. Checking source files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "$file - Found" -ForegroundColor Green
    } else {
        Write-Host "$file - NOT FOUND" -ForegroundColor Red
        Write-Host "Run 'python create_cpp_wrapper_dll.py' first" -ForegroundColor Red
        exit 1
    }
}

# Get Python configuration
Write-Host "`n4. Getting Python configuration..." -ForegroundColor Yellow
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
    Write-Host "Failed to get Python configuration: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Create output directory
Write-Host "`n5. Creating output directory..." -ForegroundColor Yellow
if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force
    Write-Host "Cleaned existing output directory" -ForegroundColor Green
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Host "Created output directory: $OutputDir" -ForegroundColor Green

# Compile the DLL
Write-Host "`n6. Compiling C++ DLL..." -ForegroundColor Yellow

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
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "Environment PATH includes: $($env:PATH.Split(';') | Where-Object { $_ -like '*Visual Studio*' -or $_ -like '*VC*' } | Select-Object -First 3)" -ForegroundColor Cyan

try {
    $compileResult = & cl.exe $compileArgs 2>&1
    $compileOutput = $compileResult | Out-String
    
    Write-Host "Compiler output:" -ForegroundColor Yellow
    Write-Host $compileOutput
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful" -ForegroundColor Green
    } else {
        Write-Host "Compilation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Compilation error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Verify DLL was created
Write-Host "`n7. Verifying DLL..." -ForegroundColor Yellow
$dllPath = "$OutputDir\BloodPressureEstimation.dll"

if (Test-Path $dllPath) {
    $dllSize = (Get-Item $dllPath).Length
    $dllSizeMB = [math]::Round($dllSize / 1MB, 2)
    Write-Host "DLL created: $dllPath" -ForegroundColor Green
    Write-Host "  Size: $dllSizeMB MB" -ForegroundColor Cyan
    
    # Check exports using dumpbin if available
    try {
        Write-Host "`n8. Checking DLL exports..." -ForegroundColor Yellow
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
    exit 1
}

Write-Host "`n=== Build completed successfully ===" -ForegroundColor Green
Write-Host "DLL Location: $dllPath" -ForegroundColor Cyan
Write-Host "DLL Size: $dllSizeMB MB" -ForegroundColor Cyan
