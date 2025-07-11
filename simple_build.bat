@echo off
echo === Simple C++ Wrapper DLL Build ===
echo.

REM 現在のディレクトリ確認
echo Current directory: %CD%
echo.

REM 必要なファイルの存在確認
echo Checking required files...
if not exist "BloodPressureEstimation_Fixed.cpp" (
    echo ERROR: BloodPressureEstimation_Fixed.cpp not found
    echo Please run 'python create_cpp_wrapper_dll.py' first
    pause
    exit /b 1
)

if not exist "BloodPressureEstimation_Fixed.h" (
    echo ERROR: BloodPressureEstimation_Fixed.h not found
    pause
    exit /b 1
)

if not exist "BloodPressureEstimation.def" (
    echo ERROR: BloodPressureEstimation.def not found
    pause
    exit /b 1
)

echo ✓ All required files found
echo.

REM Python環境確認
echo Checking Python environment...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Python開発ヘッダー確認
python -c "import sysconfig; print('Python include:', sysconfig.get_path('include'))"
python -c "import sys; print('Python library path:', sys.base_prefix + '\\libs')"
echo.

REM Visual Studio環境設定
echo Setting up Visual Studio environment...
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    echo ✓ Visual Studio Community 2022 environment loaded
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    echo ✓ Visual Studio Professional 2022 environment loaded
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    echo ✓ Visual Studio Enterprise 2022 environment loaded
) else (
    echo ERROR: Visual Studio 2022 not found
    echo Please install Visual Studio 2022 with C++ development tools
    pause
    exit /b 1
)
echo.

REM ビルドディレクトリ準備
echo Preparing build directory...
if exist build (
    echo Cleaning existing build directory...
    rmdir /s /q build
)
mkdir build
mkdir build\dist
echo ✓ Build directory prepared
echo.

REM Python情報取得
echo Getting Python configuration...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.base_prefix + '\\libs')"') do set PYTHON_LIBS=%%i
for /f "tokens=*" %%i in ('python -c "import sys; print('python' + str(sys.version_info.major) + str(sys.version_info.minor))"') do set PYTHON_LIB_NAME=%%i

echo Python include path: %PYTHON_INCLUDE%
echo Python library path: %PYTHON_LIBS%
echo Python library name: %PYTHON_LIB_NAME%
echo.

REM 直接コンパイル（CMakeを使わない）
echo Compiling C++ source...
cl.exe /EHsc /MD /I"%PYTHON_INCLUDE%" ^
    /DBLOODPRESSURE_EXPORTS ^
    /LD BloodPressureEstimation_Fixed.cpp ^
    /link /DEF:BloodPressureEstimation.def ^
    /LIBPATH:"%PYTHON_LIBS%" %PYTHON_LIB_NAME%.lib ^
    /OUT:build\dist\BloodPressureEstimation.dll

if %ERRORLEVEL% EQU 0 (
    echo ✓ Compilation successful
    echo.
    
    REM 結果確認
    echo Checking DLL exports...
    dumpbin /exports build\dist\BloodPressureEstimation.dll
    echo.
    
    echo ✓ DLL created successfully: build\dist\BloodPressureEstimation.dll
    echo.
    
    REM ファイルサイズ確認
    for %%F in (build\dist\BloodPressureEstimation.dll) do (
        set /a size=%%~zF/1024
        echo DLL size: !size! KB
    )
    
    echo.
    echo Next steps:
    echo 1. Copy build\dist\BloodPressureEstimation.dll to your C# project
    echo 2. Copy bp_estimation_simple.py to the same directory
    echo 3. Compile and run CSharpCppWrapperTest.cs
    
) else (
    echo ✗ Compilation failed
    echo Check the error messages above
)

echo.
pause