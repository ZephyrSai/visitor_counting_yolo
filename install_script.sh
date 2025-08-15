#!/bin/bash

# ================================================================
# People Counter Installation Script for Linux/Ubuntu
# ================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Banner
echo "================================================================"
echo "          People Counter Installation Script"
echo "================================================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is for Linux systems only."
    exit 1
fi

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_VERSION_NUM=$(python3 -c 'import sys; print(sys.version_info[0]*10 + sys.version_info[1])')
    
    if [ "$PYTHON_VERSION_NUM" -ge 310 ]; then
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, but $PYTHON_VERSION found"
        print_info "Please install Python 3.10 or newer"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check for NVIDIA GPU
print_info "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    print_status "NVIDIA GPU detected: $GPU_NAME"
    print_status "Driver version: $DRIVER_VERSION"
    HAS_GPU=true
    
    # Determine CUDA version based on driver
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -ge 525 ]; then
        CUDA_VERSION="12.1"
        PYTORCH_INDEX="cu121"
    elif [ "$DRIVER_MAJOR" -ge 450 ]; then
        CUDA_VERSION="11.8"
        PYTORCH_INDEX="cu118"
    else
        print_warning "Old NVIDIA driver detected. GPU support may not work properly."
        CUDA_VERSION="11.8"
        PYTORCH_INDEX="cu118"
    fi
    print_info "Recommended CUDA version: $CUDA_VERSION"
else
    print_warning "No NVIDIA GPU detected. Will install CPU-only version."
    HAS_GPU=false
fi

# Create virtual environment
print_info "Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
print_status "Pip upgraded"

# Install base dependencies
print_info "Installing base dependencies..."
pip install numpy opencv-python paho-mqtt > /dev/null 2>&1
print_status "Base dependencies installed"

# Install PyTorch
print_info "Installing PyTorch..."
if [ "$HAS_GPU" = true ]; then
    print_info "Installing PyTorch with CUDA $CUDA_VERSION support..."
    if [ "$PYTORCH_INDEX" = "cu121" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    print_info "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi
print_status "PyTorch installed"

# Install Ultralytics
print_info "Installing Ultralytics YOLO..."
pip install ultralytics>=8.2.0 > /dev/null 2>&1
print_status "Ultralytics installed"

# Verify GPU support
print_info "Verifying installation..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME_TORCH=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_status "GPU support verified: $GPU_NAME_TORCH"
else
    if [ "$HAS_GPU" = true ]; then
        print_warning "GPU detected but PyTorch cannot use it. Check CUDA installation."
    else
        print_status "CPU-only installation verified"
    fi
fi

# Download YOLO models
print_info "Downloading YOLO models..."
mkdir -p models

# Function to download with progress
download_model() {
    MODEL_NAME=$1
    MODEL_URL=$2
    
    if [ -f "models/$MODEL_NAME" ]; then
        print_warning "$MODEL_NAME already exists, skipping..."
    else
        print_info "Downloading $MODEL_NAME..."
        wget -q --show-progress -O "models/$MODEL_NAME" "$MODEL_URL"
        print_status "$MODEL_NAME downloaded"
    fi
}

# Download models
download_model "yolo11n.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
download_model "yolo11s.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
download_model "yolo11x.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"

# Create sample configuration files
print_info "Creating sample configuration files..."

# Create sample cameras.txt if it doesn't exist
if [ ! -f "cameras.txt" ]; then
    cat > cameras_example.txt << 'EOF'
# Example camera configuration file
# Format: URL|model|direction|fps|line_position
# Modify these with your actual camera URLs

# Example 1: Simple configuration
#rtsp://admin:password@192.168.1.100:554/stream

# Example 2: With model specified
#rtsp://admin:password@192.168.1.101:554/stream|models/yolo11n.pt

# Example 3: Bidirectional counting with all options
#rtsp://admin:password@192.168.1.102:554/stream|models/yolo11x.pt|bidirectional_horizontal|5|0.3

# Add your cameras below:

EOF
    print_status "Created cameras_example.txt"
    print_warning "Please create cameras.txt with your camera URLs"
else
    print_info "cameras.txt already exists"
fi

# Create run script
print_info "Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash
# Activate virtual environment and run the people counter

source venv/bin/activate

# Default settings - modify as needed
URLS_FILE="cameras.txt"
THINGSBOARD_HOST="192.168.1.11"
ACCESS_TOKEN="mAYztIXMLIris3zAIcsJ"
PROCESS_FPS=5
LINE_POSITION=0.7
DEFAULT_MODEL="models/yolo11x.pt"
DEFAULT_DIRECTION="top_to_bottom"

# Check if cameras.txt exists
if [ ! -f "$URLS_FILE" ]; then
    echo "Error: $URLS_FILE not found!"
    echo "Please create $URLS_FILE with your camera configurations."
    echo "See cameras_example.txt for format."
    exit 1
fi

# Run with ThingsBoard (default)
python3 people_counter.py \
    --urls-file "$URLS_FILE" \
    --thingsboard-host "$THINGSBOARD_HOST" \
    --access-token "$ACCESS_TOKEN" \
    --process-fps $PROCESS_FPS \
    --line-position $LINE_POSITION \
    --model "$DEFAULT_MODEL" \
    --direction "$DEFAULT_DIRECTION"

# For standalone mode (no ThingsBoard), uncomment below and comment above:
# python3 people_counter.py \
#     --urls-file "$URLS_FILE" \
#     --no-thingsboard \
#     --process-fps $PROCESS_FPS \
#     --line-position $LINE_POSITION \
#     --model "$DEFAULT_MODEL" \
#     --direction "$DEFAULT_DIRECTION"
EOF

chmod +x run.sh
print_status "Created run.sh"

# Create test script
print_info "Creating test script..."
cat > test_gpu.py << 'EOF'
#!/usr/bin/env python3
"""Test GPU and YOLO installation"""

import torch
import cv2
from ultralytics import YOLO
import numpy as np

print("=" * 60)
print("Testing Installation")
print("=" * 60)

# Test PyTorch
print("\n1. PyTorch Installation:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test OpenCV
print("\n2. OpenCV Installation:")
print(f"   OpenCV version: {cv2.__version__}")

# Test YOLO
print("\n3. YOLO Test:")
try:
    model = YOLO('models/yolo11n.pt')
    
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = model(dummy_img, verbose=False)
    
    print("   ✓ YOLO model loaded successfully")
    print("   ✓ Inference test passed")
    
    if torch.cuda.is_available():
        print("   ✓ GPU inference ready")
    else:
        print("   ! CPU inference mode")
        
except Exception as e:
    print(f"   ✗ YOLO test failed: {e}")

print("\n" + "=" * 60)
print("Installation test complete!")
print("=" * 60)
EOF

print_status "Created test_gpu.py"

# Run test
print_info "Running installation test..."
python3 test_gpu.py

# Create systemd service file (optional)
print_info "Creating systemd service template..."
cat > people-counter.service << EOF
[Unit]
Description=People Counter Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/venv/bin/python $(pwd)/people_counter.py --urls-file cameras.txt
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_status "Created people-counter.service template"

# Final instructions
echo ""
echo "================================================================"
echo "                 Installation Complete!"
echo "================================================================"
echo ""
print_status "All dependencies installed successfully"
echo ""
echo "Next steps:"
echo "1. Edit cameras.txt with your camera URLs"
echo "   Example: rtsp://admin:pass@192.168.1.100:554/stream|models/yolo11x.pt|bidirectional_horizontal|5|0.3"
echo ""
echo "2. Edit run.sh with your ThingsBoard settings (if using)"
echo ""
echo "3. Run the people counter:"
echo "   ./run.sh"
echo ""
echo "4. For standalone mode (no ThingsBoard):"
echo "   source venv/bin/activate"
echo "   python3 people_counter.py --urls-file cameras.txt --no-thingsboard"
echo ""

if [ "$CUDA_AVAILABLE" = "True" ]; then
    print_status "GPU acceleration is enabled!"
else
    print_warning "Running in CPU mode. For GPU acceleration, check CUDA installation."
fi

echo ""
echo "For auto-start on boot (optional):"
echo "  sudo cp people-counter.service /etc/systemd/system/"
echo "  sudo systemctl enable people-counter"
echo "  sudo systemctl start people-counter"
echo ""
echo "================================================================"
