#!/bin/bash

# Installation script for Multi-Stream People Counter
# Detects GPU and installs appropriate PyTorch version

echo "======================================"
echo "Multi-Stream People Counter Installer"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -n "Checking Python version... "
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Found Python $PYTHON_VERSION"
else
    echo "Python 3 not found! Please install Python 3.8 or higher."
    exit 1
fi

# Check if nvidia-smi is available
echo -n "Checking for NVIDIA GPU... "
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Found: $GPU_INFO"
    HAS_GPU=true
    
    # Detect CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "CUDA Version: $CUDA_VERSION"
else
    echo "No NVIDIA GPU detected. Will install CPU version."
    HAS_GPU=false
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install base dependencies
echo ""
echo "Installing base dependencies..."
pip install ultralytics>=8.2.0 opencv-python>=4.8.0 numpy>=1.24.0 paho-mqtt

# Install PyTorch based on GPU availability
echo ""
if [ "$HAS_GPU" = true ]; then
    echo "Installing PyTorch with GPU support..."
    
    # Determine CUDA version for PyTorch
    case $CUDA_VERSION in
        "12."*)
            echo "Installing PyTorch for CUDA 12.1..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        "11."*)
            echo "Installing PyTorch for CUDA 11.8..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        *)
            echo "Unknown CUDA version. Installing default GPU PyTorch..."
            pip install torch torchvision torchaudio
            ;;
    esac
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 << EOF
import torch
import cv2
from ultralytics import YOLO

print("✓ PyTorch version:", torch.__version__)
print("✓ OpenCV version:", cv2.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ GPU:", torch.cuda.get_device_name(0))
    print("✓ CUDA version:", torch.version.cuda)
EOF

# Download default YOLO model
echo ""
echo "Downloading YOLO models..."
wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt
echo "✓ Downloaded yolov11n.pt"

# Create sample cameras.txt
echo ""
echo "Creating sample cameras.txt..."
cat > cameras_sample.txt << EOF
# Sample RTSP URLs - Replace with your camera URLs
# Format: rtsp://username:password@ip:port/path
# Uncomment and modify the lines below:

# Hikvision example:
# rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua example:
# rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0

# Generic RTSP:
# rtsp://username:password@192.168.1.102:554/stream1

# For testing with webcam, use:
# 0
EOF
echo "✓ Created cameras_sample.txt"

# Create run scripts
echo ""
echo "Creating run scripts..."

# Create run script for ThingsBoard mode
cat > run_with_thingsboard.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python multi_stream_thingsboard.py --urls-file cameras.txt "$@"
EOF
chmod +x run_with_thingsboard.sh

# Create run script for standalone mode
cat > run_standalone.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python multi_stream_thingsboard.py --urls-file cameras.txt --no-thingsboard "$@"
EOF
chmod +x run_standalone.sh

echo "✓ Created run scripts"

# Final instructions
echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Copy cameras_sample.txt to cameras.txt and add your RTSP URLs"
echo "   cp cameras_sample.txt cameras.txt"
echo "   nano cameras.txt"
echo ""
echo "2. Run with ThingsBoard:"
echo "   ./run_with_thingsboard.sh"
echo ""
echo "3. Run standalone (no ThingsBoard):"
echo "   ./run_standalone.sh"
echo ""
echo "4. For custom settings, add arguments:"
echo "   ./run_standalone.sh --line-position 0.3 --process-fps 10"
echo ""

if [ "$HAS_GPU" = false ]; then
    echo "⚠ Note: No GPU detected. The system will run on CPU."
    echo "  For better performance, consider using a CUDA-capable GPU."
fi

echo ""
echo "For more information, see README.md"
echo "======================================"