@echo off
echo Testing GitHub Actions optimization settings...

:: テスト設定
set TEST_CONFIGS=(
    "Individual Libraries + dlib"
    "Lightweight OpenCV + dlib"
    "Static Linking + dlib"
)

:: 各設定でテスト実行
for %%c in (%TEST_CONFIGS%) do (
    echo.
    echo ========================================
    echo Testing: %%c
    echo ========================================
    
    :: ビルドディレクトリをクリア
    if exist build rmdir /s /q build
    mkdir build
    cd build
    
    :: CMake設定（GitHub Actions最適化付き）
    if "%%c"=="Individual Libraries + dlib" (
        cmake -G "Visual Studio 17 2022" -A x64 -DUSE_INDIVIDUAL_OPENCV_LIBS=ON -DUSE_GITHUB_ACTIONS_OPTIMIZED=ON ..
    ) else if "%%c"=="Lightweight OpenCV + dlib" (
        cmake -G "Visual Studio 17 2022" -A x64 -DUSE_LIGHTWEIGHT_OPENCV=ON -DUSE_GITHUB_ACTIONS_OPTIMIZED=ON ..
    ) else if "%%c"=="Static Linking + dlib" (
        cmake -G "Visual Studio 17 2022" -A x64 -DUSE_STATIC_OPENCV=ON -DUSE_GITHUB_ACTIONS_OPTIMIZED=ON ..
    )
    
    if %ERRORLEVEL% neq 0 (
        echo CMake configuration failed for %%c
        cd ..
        continue
    )
    
    :: ビルド実行
    cmake --build . --config Release --parallel 4
    
    if %ERRORLEVEL% neq 0 (
        echo Build failed for %%c
        cd ..
        continue
    )
    
    :: DLLサイズ測定
    if exist Release\BloodPressureDLL.dll (
        for %%F in (Release\BloodPressureDLL.dll) do (
            set /a size=%%~zF/1024/1024
            echo DLL size: !size! MB
        )
    )
    
    :: パフォーマンステスト実行
    if exist Release\test_app.exe (
        echo Running performance test...
        cd Release
        test_app.exe "sample-data\sample_1M.webm" --performance-test
        cd ..
    )
    
    cd ..
    echo Test completed for %%c
)

echo.
echo ========================================
echo All optimization tests completed!
echo ======================================== 
