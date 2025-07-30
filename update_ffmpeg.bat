@echo off
echo Updating ffmpeg.exe with H.264 support...

REM Check if ffmpeg-ultra-slim artifact is available
if not exist "ffmpeg-ultra-slim\ffmpeg.exe" (
    echo Error: ffmpeg-ultra-slim\ffmpeg.exe not found
    echo Please download the artifact from GitHub Actions first
    echo 1. Go to Actions tab in your repository
    echo 2. Find the "Build FFmpeg with H.264 Support" workflow
    echo 3. Download the "ffmpeg-ultra-slim" artifact
    echo 4. Extract it to the current directory
    pause
    exit /b 1
)

REM Backup existing ffmpeg.exe if it exists
if exist "ffmpeg.exe" (
    echo Backing up existing ffmpeg.exe...
    move ffmpeg.exe ffmpeg.exe.backup
)

REM Copy new ffmpeg.exe
echo Copying new ffmpeg.exe with H.264 support...
copy "ffmpeg-ultra-slim\ffmpeg.exe" "ffmpeg.exe"

REM Also copy to CppBloodPressureDLL/package/ if it exists
if exist "CppBloodPressureDLL\package\" (
    echo Copying to CppBloodPressureDLL\package\...
    copy "ffmpeg-ultra-slim\ffmpeg.exe" "CppBloodPressureDLL\package\ffmpeg.exe"
)

echo.
echo FFmpeg update completed successfully!
echo The new ffmpeg.exe supports H.264 decoding for webm files.
echo.
echo To test the new ffmpeg:
echo ffmpeg.exe -i your_video.webm -f image2 frame_%%05d.jpg
pause 
