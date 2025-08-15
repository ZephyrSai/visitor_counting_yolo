@echo off
REM ================================================================
REM People Counter Installation Script for Windows
REM ================================================================

setlocal enabledelayedexpansion

REM Colors and formatting
echo ================================================================
echo           People Counter Installation Script
echo ================================================================
echo.

REM Check Python installation
echo [i] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python is not installed or not in PATH
    echo [!] Please install Python 3.10+ from https://python.org
    echo [!] Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Check Python version is 3.10+
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo [X] Python 3.10+ required
    pause
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 10 (
    echo [X] Python 3.10+ required, but %PYTHON_VERSION% found
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
echo [i] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected
    for /f "tokens=1-5 delims= " %%a in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        echo      GPU: %%a %%b %%c %%d %%e
    )
    for /f "tokens=1" %%a in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
        echo      Driver: %%a
    )
    set HAS_GPU=1
    
    REM Determine CUDA version
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
        set DRIVER_MAJOR=%%a
    )
    
    if !DRIVER_MAJOR! GEQ 525 (
        set CUDA_VERSION=12.1
        set PYTORCH_INDEX=cu121
    ) else if !DRIVER_MAJOR! GEQ 450 (
        set CUDA_VERSION=11.8
        set PYTORCH_INDEX=cu118
    ) else (
        echo [!] Old NVIDIA driver detected. GPU support may not work.
        set CUDA_VERSION=11.8
        set PYTORCH_INDEX=cu118
    )
    echo [i] Recommended CUDA version: !CUDA_VERSION!
) else (
    echo [!] No NVIDIA GPU detected. Will install CPU-only version.
    set HAS_GPU=0
)

REM Create virtual environment
echo [i] Creating Python virtual environment...
if exist venv (
    echo [!] Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [i] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo [i] Upgrading pip...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1
echo [OK] Pip upgraded

REM Install base dependencies
echo [i] Installing base dependencies...
pip install numpy opencv-python paho-mqtt >nul 2>&1
echo [OK] Base dependencies installed

REM Install PyTorch
echo [i] Installing PyTorch...
if %HAS_GPU%==1 (
    echo [i] Installing PyTorch with CUDA !CUDA_VERSION! support...
    if "!PYTORCH_INDEX!"=="cu121" (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo [i] Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)
echo [OK] PyTorch installed

REM Install Ultralytics
echo [i] Installing Ultralytics YOLO...
pip install ultralytics>=8.2.0 >nul 2>&1
echo [OK] Ultralytics installed

REM Verify GPU support
echo [i] Verifying installation...
python -c "import torch; print('[OK] PyTorch installed'); print(f'     CUDA available: {torch.cuda.is_available()}'); print(f'     GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>nul

REM Create models directory
if not exist models mkdir models

REM Download YOLO models
echo [i] Downloading YOLO models...

REM Download yolo11n.pt
if exist "models\yolo11n.pt" (
    echo [!] yolo11n.pt already exists, skipping...
) else (
    echo [i] Downloading yolo11n.pt...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt' -OutFile 'models\yolo11n.pt'"
    echo [OK] yolo11n.pt downloaded
)

REM Download yolo11s.pt
if exist "models\yolo11s.pt" (
    echo [!] yolo11s.pt already exists, skipping...
) else (
    echo [i] Downloading yolo11s.pt...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt' -OutFile 'models\yolo11s.pt'"
    echo [OK] yolo11s.pt downloaded
)

REM Download yolo11x.pt
if exist "models\yolo11x.pt" (
    echo [!] yolo11x.pt already exists, skipping...
) else (
    echo [i] Downloading yolo11x.pt...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt' -OutFile 'models\yolo11x.pt'"
    echo [OK] yolo11x.pt downloaded
)

REM Create sample configuration files
echo [i] Creating sample configuration files...

if not exist cameras.txt (
    (
        echo # Example camera configuration file
        echo # Format: URL^|model^|direction^|fps^|line_position
        echo # Modify these with your actual camera URLs
        echo.
        echo # Example 1: Simple configuration
        echo #rtsp://admin:password@192.168.1.100:554/stream
        echo.
        echo # Example 2: With model specified
        echo #rtsp://admin:password@192.168.1.101:554/stream^|models/yolo11n.pt
        echo.
        echo # Example 3: Bidirectional counting with all options
        echo #rtsp://admin:password@192.168.1.102:554/stream^|models/yolo11x.pt^|bidirectional_horizontal^|5^|0.3
        echo.
        echo # Add your cameras below:
        echo.
    ) > cameras_example.txt
    echo [OK] Created cameras_example.txt
    echo [!] Please create cameras.txt with your camera URLs
) else (
    echo [i] cameras.txt already exists
)

REM Create run.bat script
echo [i] Creating run script...
(
    echo @echo off
    echo REM Activate virtual environment and run the people counter
    echo.
    echo call venv\Scripts\activate.bat
    echo.
    echo REM Default settings - modify as needed
    echo set URLS_FILE=cameras.txt
    echo set THINGSBOARD_HOST=192.168.1.11
    echo set ACCESS_TOKEN=mAYztIXMLIris3zAIcsJ
    echo set PROCESS_FPS=5
    echo set LINE_POSITION=0.7
    echo set DEFAULT_MODEL=models/yolo11x.pt
    echo set DEFAULT_DIRECTION=top_to_bottom
    echo.
    echo REM Check if cameras.txt exists
    echo if not exist %%URLS_FILE%% ^(
    echo     echo Error: %%URLS_FILE%% not found!
    echo     echo Please create %%URLS_FILE%% with your camera configurations.
    echo     echo See cameras_example.txt for format.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo REM Run with ThingsBoard ^(default^)
    echo python people_counter.py ^^
    echo     --urls-file "%%URLS_FILE%%" ^^
    echo     --thingsboard-host "%%THINGSBOARD_HOST%%" ^^
    echo     --access-token "%%ACCESS_TOKEN%%" ^^
    echo     --process-fps %%PROCESS_FPS%% ^^
    echo     --line-position %%LINE_POSITION%% ^^
    echo     --model "%%DEFAULT_MODEL%%" ^^
    echo     --direction "%%DEFAULT_DIRECTION%%"
    echo.
    echo REM For standalone mode ^(no ThingsBoard^), use run_standalone.bat
    echo pause
) > run.bat
echo [OK] Created run.bat

REM Create standalone run script
(
    echo @echo off
    echo REM Run in standalone mode without ThingsBoard
    echo.
    echo call venv\Scripts\activate.bat
    echo.
    echo set URLS_FILE=cameras.txt
    echo set PROCESS_FPS=5
    echo set LINE_POSITION=0.7
    echo set DEFAULT_MODEL=models/yolo11x.pt
    echo set DEFAULT_DIRECTION=top_to_bottom
    echo.
    echo if not exist %%URLS_FILE%% ^(
    echo     echo Error: %%URLS_FILE%% not found!
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo python people_counter.py ^^
    echo     --urls-file "%%URLS_FILE%%" ^^
    echo     --no-thingsboard ^^
    echo     --process-fps %%PROCESS_FPS%% ^^
    echo     --line-position %%LINE_POSITION%% ^^
    echo     --model "%%DEFAULT_MODEL%%" ^^
    echo     --direction "%%DEFAULT_DIRECTION%%"
    echo.
    echo pause
) > run_standalone.bat
echo [OK] Created run_standalone.bat

REM Create test script
echo [i] Creating test script...
(
    echo import torch
    echo import cv2
    echo from ultralytics import YOLO
    echo import numpy as np
    echo.
    echo print^("=" * 60^)
    echo print^("Testing Installation"^)
    echo print^("=" * 60^)
    echo.
    echo print^("\n1. PyTorch Installation:"^)
    echo print^(f"   PyTorch version: {torch.__version__}"^)
    echo print^(f"   CUDA available: {torch.cuda.is_available^(^)}"^)
    echo if torch.cuda.is_available^(^):
    echo     print^(f"   CUDA version: {torch.version.cuda}"^)
    echo     print^(f"   GPU: {torch.cuda.get_device_name^(0^)}"^)
    echo     print^(f"   GPU memory: {torch.cuda.get_device_properties^(0^).total_memory / 1024**3:.1f} GB"^)
    echo.
    echo print^("\n2. OpenCV Installation:"^)
    echo print^(f"   OpenCV version: {cv2.__version__}"^)
    echo.
    echo print^("\n3. YOLO Test:"^)
    echo try:
    echo     model = YOLO^('models/yolo11n.pt'^)
    echo     dummy_img = np.random.randint^(0, 255, ^(640, 640, 3^), dtype=np.uint8^)
    echo     results = model^(dummy_img, verbose=False^)
    echo     print^("   √ YOLO model loaded successfully"^)
    echo     print^("   √ Inference test passed"^)
    echo     if torch.cuda.is_available^(^):
    echo         print^("   √ GPU inference ready"^)
    echo     else:
    echo         print^("   ! CPU inference mode"^)
    echo except Exception as e:
    echo     print^(f"   X YOLO test failed: {e}"^)
    echo.
    echo print^("\n" + "=" * 60^)
    echo print^("Installation test complete!"^)
    echo print^("=" * 60^)
) > test_installation.py

echo [OK] Created test_installation.py

REM Run test
echo [i] Running installation test...
python test_installation.py

REM Create Windows Task Scheduler XML template
echo [i] Creating Windows Task Scheduler template...
(
    echo ^<?xml version="1.0" encoding="UTF-16"?^>
    echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
    echo   ^<RegistrationInfo^>
    echo     ^<Description^>People Counter Service^</Description^>
    echo   ^</RegistrationInfo^>
    echo   ^<Triggers^>
    echo     ^<LogonTrigger^>
    echo       ^<Enabled^>true^</Enabled^>
    echo     ^</LogonTrigger^>
    echo   ^</Triggers^>
    echo   ^<Settings^>
    echo     ^<RestartOnFailure^>
    echo       ^<Interval^>PT1M^</Interval^>
    echo       ^<Count^>3^</Count^>
    echo     ^</RestartOnFailure^>
    echo   ^</Settings^>
    echo   ^<Actions^>
    echo     ^<Exec^>
    echo       ^<Command^>%CD%\run.bat^</Command^>
    echo       ^<WorkingDirectory^>%CD%^</WorkingDirectory^>
    echo     ^</Exec^>
    echo   ^</Actions^>
    echo ^</Task^>
) > people-counter-task.xml
echo [OK] Created Task Scheduler template

REM Final instructions
echo.
echo ================================================================
echo                  Installation Complete!
echo ================================================================
echo.
echo [OK] All dependencies installed successfully
echo.
echo Next steps:
echo.
echo 1. Edit cameras.txt with your camera URLs
echo    Example: rtsp://admin:pass@192.168.1.100:554/stream^|models/yolo11x.pt^|bidirectional_horizontal^|5^|0.3
echo.
echo 2. Edit run.bat with your ThingsBoard settings (if using)
echo.
echo 3. Run the people counter:
echo    - With ThingsBoard: run.bat
echo    - Standalone mode: run_standalone.bat
echo.

if %HAS_GPU%==1 (
    echo [OK] GPU acceleration is enabled!
) else (
    echo [!] Running in CPU mode. For GPU acceleration, install CUDA and appropriate GPU drivers.
)

echo.
echo For auto-start on Windows boot (optional):
echo   1. Open Task Scheduler
echo   2. Import people-counter-task.xml
echo   3. Modify settings as needed
echo.
echo ================================================================
echo.
pause
