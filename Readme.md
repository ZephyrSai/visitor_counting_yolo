# 🎯 Multi-Stream People Counter with YOLO and ByteTrack

A high-performance, GPU-accelerated people counting system for multiple RTSP streams. Features directional counting, ThingsBoard IoT integration, and real-time visualization.

## 🌟 Features

- **Multi-Stream Support**: Process multiple RTSP cameras simultaneously
- **GPU Acceleration**: NVIDIA CUDA support for high-performance inference
- **Directional Counting**: Count people moving in specific directions (up/down/left/right)
- **Smart Tracking**: ByteTrack algorithm for robust multi-object tracking
- **IoT Integration**: Optional ThingsBoard telemetry for real-time dashboards
- **Configurable Line Position**: Adjustable counting line placement
- **Frame Rate Optimization**: Process only N frames per second while maintaining smooth tracking
- **Automatic Reconnection**: Handles network interruptions gracefully
- **Individual Visualization**: Separate window for each camera stream

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU (recommended) or CPU
- CUDA-compatible GPU for acceleration
- RTSP-enabled IP cameras
- Ubuntu 20.04+ or Windows 10/11

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/people-counter.git
cd people-counter
```

### 2. Install Base Dependencies

```bash
pip install ultralytics>=8.2.0 opencv-python>=4.8.0 numpy>=1.24.0 paho-mqtt
```

### 3. Install PyTorch with GPU Support

#### Step 1: Check Your NVIDIA GPU

```bash
nvidia-smi
```

Note your GPU model and driver version.

#### Step 2: Determine CUDA Version

Based on your GPU generation:

| GPU Generation | Recommended CUDA | Compute Capability |
|----------------|------------------|-------------------|
| **RTX 40 Series** (4090, 4080, 4070) | CUDA 12.1+ | 8.9 |
| **RTX 30 Series** (3090, 3080, 3070, 3060, 3050) | CUDA 11.8+ | 8.6 |
| **RTX 20 Series** (2080 Ti, 2080, 2070, 2060) | CUDA 11.8+ | 7.5 |
| **GTX 16 Series** (1660, 1650) | CUDA 11.8+ | 7.5 |
| **GTX 10 Series** (1080 Ti, 1080, 1070, 1060) | CUDA 11.8+ | 6.1 |
| **Older GPUs** | Check [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus) | Varies |

#### Step 3: Install PyTorch

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the latest commands.

**For RTX 30/40 Series (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For RTX 20 Series and older (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**
```bash
pip install torch torchvision torchaudio
```

#### Step 4: Verify GPU Installation

```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### 4. Download YOLO Model

The model will auto-download on first run, or manually:

```bash
# Download YOLOv11 nano model (recommended for speed)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt

# For better accuracy (slower)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11s.pt
```

## 📁 Configuration

### 1. Create Camera List File

Create `cameras.txt` with your RTSP URLs (one per line):

```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101
rtsp://admin:password@192.168.1.102:554/Streaming/Channels/101
```

### 2. RTSP URL Formats

**Hikvision:**
```
rtsp://username:password@IP:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://username:password@IP:554/cam/realmonitor?channel=1&subtype=0
```

**Axis:**
```
rtsp://username:password@IP/axis-media/media.amp
```

## 🎮 Usage

### Basic Usage
```bash
python multi_stream_thingsboard.py --urls-file cameras.txt
```

### Standalone Mode (No ThingsBoard)
```bash
python multi_stream_thingsboard.py --urls-file cameras.txt --no-thingsboard
```

### Custom Configuration
```bash
python multi_stream_thingsboard.py \
  --urls-file cameras.txt \
  --line-position 0.3 \
  --process-fps 10 \
  --direction top_to_bottom \
  --model yolo11s.pt
```

### All Options
| Option | Default | Description |
|--------|---------|-------------|
| `--urls-file` | Required | Path to file with RTSP URLs |
| `--thingsboard-host` | 192.168.1.11 | ThingsBoard server IP |
| `--access-token` | (built-in) | ThingsBoard device token |
| `--no-thingsboard` | False | Disable IoT telemetry |
| `--process-fps` | 5 | Frames to process per second |
| `--direction` | top_to_bottom | Counting direction |
| `--model` | yolo12n.pt | YOLO model file |
| `--line-position` | 0.75 | Line position (0.0-1.0) |

## 📊 Display Information

### With ThingsBoard (default):
```
Total Count: 234        # All-time total
This Hour: 45          # Current hour count
This Minute: 3         # Current minute count
Active Tracks: 2       # Currently tracking
Source FPS: 25.1       # Camera stream FPS
Process FPS: 5.0 / 5   # Actual/Target
Runtime: 01:45         # Time since start
```

### Standalone Mode:
```
Total Count: 234        # All-time total
This Hour: 45          # Current hour count
Active Tracks: 2       # Currently tracking
Source FPS: 25.1       # Camera stream FPS
Process FPS: 5.0 / 5   # Actual/Target
Runtime: 01:45         # Time since start
```

## 🌐 ThingsBoard Integration

### Data Format

**Minute Updates (when count > 0):**
```json
{
  "cam_192_168_1_100_count": 5,      // This minute
  "cam_192_168_1_100_total": 127,    // Total since start
  "total_count": 8,                   // All cameras this minute
  "all_cameras_total": 216            // All cameras total
}
```

**Hourly Updates:**
```json
{
  "cam_192_168_1_100_hourly": 45,    // This hour
  "total_hourly": 77                  // All cameras this hour
}
```

### Dashboard Setup

1. Create a device in ThingsBoard
2. Copy the access token
3. Create widgets for:
   - `{camera}_count` - Minute counts
   - `{camera}_total` - Running totals
   - `{camera}_hourly` - Hourly counts
   - `all_cameras_total` - Combined total

## 🎯 Line Position Guide

| Position | Value | Use Case |
|----------|-------|----------|
| Top/Left | 0.0-0.3 | Entrance detection |
| Center | 0.4-0.6 | General counting |
| Bottom/Right | 0.7-1.0 | Exit detection |

## 🚨 Troubleshooting

### GPU Not Detected

1. **Check NVIDIA Driver:**
   ```bash
   nvidia-smi
   ```
   
2. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Stream Connection Issues

1. **Test RTSP URL:**
   ```bash
   ffplay "rtsp://username:password@ip:port/path"
   # or
   vlc "rtsp://username:password@ip:port/path"
   ```

2. **Check Network:**
   ```bash
   ping camera_ip
   ```

3. **Verify Credentials:** Ensure username/password are correct

### Performance Issues

1. **Reduce Processing FPS:**
   ```bash
   --process-fps 3
   ```

2. **Use Lighter Model:**
   ```bash
   --model yolo11n.pt
   ```

3. **Monitor GPU Usage:**
   ```bash
   nvidia-smi -l 1
   ```

### Recommended Settings

**High Traffic Areas:**
- Process FPS: 8-10
- Model: yolo11s or yolo11m
- Line Position: 0.3 (early detection)

**Normal Traffic:**
- Process FPS: 5
- Model: yolo11n
- Line Position: 0.5

**Low Traffic:**
- Process FPS: 2-3
- Model: yolo11n
- Line Position: 0.7

## 🔒 Security Considerations

1. **Secure Storage:**
   ```bash
   chmod 600 cameras.txt  # Restrict file permissions
   ```

2. **Environment Variables:** For production, use environment variables:
   ```bash
   export TB_HOST="192.168.1.11"
   export TB_TOKEN="your_token"
   ```

3. **Network Security:** Use VPN or secure network for RTSP streams

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for tracking algorithm
- [ThingsBoard](https://thingsboard.io/) for IoT platform

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review closed issues on GitHub
3. Open a new issue with details

---

**Note:** Always ensure you have permission to monitor areas with cameras and comply with local privacy laws.
