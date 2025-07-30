@echo off
echo Testing FFmpeg H.264 support for webm files...

REM Check if ffmpeg.exe exists
if not exist "ffmpeg.exe" (
    echo Error: ffmpeg.exe not found
    echo Please run update_ffmpeg.bat first
    pause
    exit /b 1
)

REM Test ffmpeg version and supported decoders
echo.
echo === FFmpeg Version ===
ffmpeg.exe -version

echo.
echo === Supported Decoders ===
ffmpeg.exe -decoders | findstr "h264\|vp9"

echo.
echo === Testing with sample webm file ===
if exist "sample-data\sample_1M.webm" (
    echo Testing with sample_1M.webm...
    ffmpeg.exe -i "sample-data\sample_1M.webm" -f image2 test_frame_%%05d.jpg -vframes 5
    if %errorlevel% equ 0 (
        echo SUCCESS: FFmpeg successfully processed the webm file with H.264
        echo Generated test frames: test_frame_*.jpg
    ) else (
        echo ERROR: FFmpeg failed to process the webm file
    )
) else (
    echo Warning: sample-data\sample_1M.webm not found
    echo Please place a test webm file in the sample-data directory
)

echo.
echo Test completed.
pause 
