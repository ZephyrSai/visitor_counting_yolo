@echo off
REM Installation script for Multi-Stream People Counter on Windows

echo ======================================
echo Multi-Stream People Counter Installer
echo ======================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Show Python version
python --version

REM Check for NVIDIA GPU
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected!
    nvidia-smi --query-gpu=name --format=csv,noheader
    set HAS_GPU=1
) else (
    echo No NVIDIA GPU detected. Will install CPU version.
    set HAS_GPU=0
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install base dependencies
echo.
echo Installing base dependencies...
pip install ultralytics opencv-python numpy paho-mqtt

REM Install PyTorch
echo.
if %HAS_GPU%==1 (
    echo Installing PyTorch with GPU support...
    echo Detecting CUDA version...
    
    REM Try CUDA 12.1 first (for newer GPUs)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    REM Verify GPU is detected
    python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
) else (
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)

REM Download YOLO model
echo.
echo Downloading YOLO model...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt' -OutFile 'yolov11n.pt'"
echo Downloaded yolov11n.pt

REM Create sample cameras.txt
echo.
echo Creating sample cameras.txt...
(
echo # Sample RTSP URLs - Replace with your camera URLs
echo # Format: rtsp://username:password@ip:port/path
echo # Uncomment and modify the lines below:
echo.
echo # Hikvision example:
echo # rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
echo.
echo # Dahua example:
echo # rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1^&subtype=0
echo.
echo # For testing with webcam, use:
echo # 0
) > cameras_sample.txt

REM Create run scripts
echo.
echo Creating run scripts...

REM Run with ThingsBoard
(
echo @echo off
echo call venv\Scripts\activate.bat
echo python multi_stream_thingsboard.py --urls-file cameras.txt %%*
) > run_with_thingsboard.bat

REM Run standalone
(
echo @echo off
echo call venv\Scripts\activate.bat
echo python multi_stream_thingsboard.py --urls-file cameras.txt --no-thingsboard %%*
) > run_standalone.bat

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch, cv2; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA:', torch.cuda.is_available())"

echo.
echo ======================================
echo Installation Complete!
echo ======================================
echo.
echo Next steps:
echo 1. Copy cameras_sample.txt to cameras.txt and add your RTSP URLs
echo    copy cameras_sample.txt cameras.txt
echo    notepad cameras.txt
echo.
echo 2. Run with ThingsBoard:
echo    run_with_thingsboard.bat
echo.
echo 3. Run standalone (no ThingsBoard):
echo    run_standalone.bat
echo.
echo 4. For custom settings, add arguments:
echo    run_standalone.bat --line-position 0.3 --process-fps 10
echo.

if %HAS_GPU%==0 (
    echo Note: No GPU detected. The system will run on CPU.
    echo For better performance, consider using a CUDA-capable GPU.
    echo.
)

echo For more information, see README.md
echo ======================================
pause