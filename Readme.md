# 🎯 Advanced Multi-Stream People Counter with Bidirectional Tracking

A high-performance, GPU-accelerated people counting system for multiple RTSP streams. Features bidirectional counting (IN/OUT), per-camera model selection, ThingsBoard IoT integration, and real-time visualization.

## 🌟 Key Features

- **Multi-Stream Support**: Process multiple RTSP cameras simultaneously
- **Bidirectional Counting**: Track people entering AND exiting in a single camera
- **Per-Camera Model Selection**: Use different YOLO models for different cameras
- **GPU Acceleration**: NVIDIA CUDA support for high-performance inference
- **Smart Tracking**: ByteTrack algorithm for robust multi-object tracking
- **IoT Integration**: Real-time ThingsBoard telemetry with separate IN/OUT metrics
- **Configurable Line Position**: Adjustable counting line placement
- **Frame Rate Optimization**: Per-camera FPS settings
- **Automatic Reconnection**: Handles network interruptions gracefully
- **Net Flow Calculation**: Automatic calculation of net people flow (IN - OUT)

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
| **RTX 30 Series** (3090, 3080, 3070, 3060) | CUDA 11.8+ | 8.6 |
| **RTX 20 Series** (2080 Ti, 2080, 2070, 2060) | CUDA 11.8+ | 7.5 |
| **GTX 16 Series** (1660, 1650) | CUDA 11.8+ | 7.5 |
| **GTX 10 Series** (1080 Ti, 1080, 1070, 1060) | CUDA 11.8+ | 6.1 |

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

### 4. Download YOLO Models

Models auto-download on first run, or manually download:

```bash
# Lightweight model for simple scenes
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt

# Standard model for general use
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt

# Large model for complex/crowded scenes
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11l.pt
```

## 📁 Configuration

### Camera Configuration File Format

Create `cameras.txt` with flexible per-camera settings:

```bash
# Format: URL|model|direction|fps|line_position
# All fields except URL are optional

# BIDIRECTIONAL COUNTING with custom line positions
# Main entrance - early detection at 30%
rtsp://admin:pass@192.168.1.100:554/entrance|yolo11x.pt|bidirectional_horizontal|5|0.3

# Hallway - center line at 50%
rtsp://admin:pass@192.168.1.101:554/hallway|yolo11x.pt|bidirectional_vertical|5|0.5

# SINGLE DIRECTION with various line positions
# Entry door - very early detection at 20%
rtsp://admin:pass@192.168.1.102:554/entry|yolo11n.pt|top_to_bottom|10|0.2

# Exit door - late detection at 80%
rtsp://admin:pass@192.168.1.103:554/exit|yolo11n.pt|bottom_to_top|10|0.8

# Complex entrance - large model, slow FPS, line at 40%
rtsp://admin:pass@192.168.1.104:554/main_door|yolo11l.pt|bidirectional_horizontal|3|0.4

# Using defaults from command line (no line position specified)
rtsp://admin:pass@192.168.1.105:554/stream
```

### Configuration Options Per Camera

| Field | Options | Description | Example |
|-------|---------|-------------|---------|
| **URL** | RTSP URL | Camera stream URL (required) | rtsp://admin:pass@192.168.1.100:554/stream |
| **Model** | yolo11n.pt, yolo11x.pt, yolo11l.pt | YOLO model to use | yolo11x.pt |
| **Direction** | See direction options below | Counting direction | bidirectional_horizontal |
| **FPS** | 1-30 | Frames to process per second | 5 |
| **Line Position** | 0.0-1.0 | Position of counting line | 0.3 (30% from top/left) |

### Direction Options

| Direction | Description | Use Case |
|-----------|-------------|----------|
| `top_to_bottom` | Count downward movement | Entrances |
| `bottom_to_top` | Count upward movement | Exits |
| `left_to_right` | Count rightward movement | Corridors |
| `right_to_left` | Count leftward movement | Corridors |
| **`bidirectional_horizontal`** | Count BOTH up/down | Doors, entrances |
| **`bidirectional_vertical`** | Count BOTH left/right | Hallways, corridors |

## 🎮 Usage

### Basic Usage
```bash
python people_counter.py --urls-file cameras.txt
```

### Standalone Mode (No ThingsBoard)
```bash
python people_counter.py --urls-file cameras.txt --no-thingsboard
```

### Custom Default Settings
```bash
python people_counter.py \
  --urls-file cameras.txt \
  --line-position 0.5 \
  --process-fps 5 \
  --direction bidirectional_horizontal \
  --model yolo11x.pt
```

### All Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--urls-file` | Required | Path to camera configuration file |
| `--thingsboard-host` | 192.168.1.11 | ThingsBoard server IP |
| `--access-token` | (built-in) | ThingsBoard device token |
| `--no-thingsboard` | False | Run without IoT telemetry |
| `--process-fps` | 5 | Default FPS (can override per camera) |
| `--direction` | top_to_bottom | Default direction (can override per camera) |
| `--model` | yolo11x.pt | Default model (can override per camera) |
| `--line-position` | 0.7 | Counting line position (0.0-1.0) |

## 📊 Display Information

### Bidirectional Camera Display
```
Model: yolo11x.pt
═══ Down↓ (IN) ═══
Total: 234
Hour: 45
Minute: 3
═══ Up↑ (OUT) ═══
Total: 189
Hour: 32
Minute: 2
═══ NET: 45 ═══
Active Tracks: 2
Runtime: 01:45
```

### Single Direction Display
```
Model: yolo11n.pt
Total Count: 234
This Hour: 45
This Minute: 3
Active Tracks: 2
Runtime: 01:45
```

## 🌐 ThingsBoard Integration

### Telemetry Data Format

**Bidirectional Cameras (per minute):**
```json
{
  "cam_192_168_1_100_in": 5,           // People entered this minute
  "cam_192_168_1_100_out": 3,          // People exited this minute
  "cam_192_168_1_100_net": 2,          // Net flow (in - out)
  "cam_192_168_1_100_total_in": 234,   // Total entered since start
  "cam_192_168_1_100_total_out": 189,  // Total exited since start
  "cam_192_168_1_100_total_net": 45,   // Total net flow
  "total_in": 8,                       // All cameras IN this minute
  "total_out": 5,                      // All cameras OUT this minute
  "total_net": 3                       // All cameras NET this minute
}
```

**Single Direction Cameras (per minute):**
```json
{
  "cam_192_168_1_102_count": 5,        // This minute
  "cam_192_168_1_102_total": 127,      // Total since start
  "total_count": 8                     // All single-direction cameras
}
```

**Hourly Updates:**
```json
{
  "cam_192_168_1_100_hourly_in": 45,   // Entered this hour
  "cam_192_168_1_100_hourly_out": 32,  // Exited this hour
  "cam_192_168_1_100_hourly_net": 13,  // Net this hour
  "total_hourly_in": 77,               // All cameras IN
  "total_hourly_out": 54,              // All cameras OUT
  "total_hourly_net": 23               // All cameras NET
}
```

### Dashboard Widget Setup

Create widgets in ThingsBoard for:

**Bidirectional Metrics:**
- `{camera}_in` - Real-time entries
- `{camera}_out` - Real-time exits
- `{camera}_net` - Net flow
- `{camera}_total_in` - Cumulative entries
- `{camera}_total_out` - Cumulative exits

**Aggregate Metrics:**
- `total_in` - All cameras entries/minute
- `total_out` - All cameras exits/minute
- `total_net` - System-wide net flow

## 🎯 Model Selection Guide

### Model Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|------------|----------|
| **yolo11n.pt** | Fastest | Good | Simple scenes, high FPS needs |
| **yolo11s.pt** | Fast | Better | Balanced performance |
| **yolo11m.pt** | Moderate | High | Standard surveillance |
| **yolo11l.pt** | Slower | Highest | Crowded/complex scenes |
| **yolo11x.pt** | Slowest | Best | Maximum accuracy |

### Recommended Configurations

**Store Entrance (Bidirectional with early detection):**
```
rtsp://url|yolo11x.pt|bidirectional_horizontal|5|0.3
```

**Narrow Hallway (Simple with late detection):**
```
rtsp://url|yolo11n.pt|left_to_right|10|0.8
```

**Crowded Area (Heavy model with center line):**
```
rtsp://url|yolo11l.pt|bidirectional_horizontal|3|0.5
```

**Escalator/Stairs (Direction-specific with custom lines):**
```
# Bottom of escalator - early detection
rtsp://url|yolo11s.pt|bottom_to_top|5|0.2

# Top of escalator - late detection
rtsp://url|yolo11s.pt|top_to_bottom|5|0.8
```

## 🎯 Line Position Guide

### Per-Camera Line Position
Each camera can have its own line position (0.0 to 1.0):

| Position | Value Range | Use Case | Example |
|----------|-------------|----------|---------|
| **Very Early** | 0.1-0.2 | Wide entrances, maximum lead time | Store entrance: 0.15 |
| **Early** | 0.3-0.4 | Standard entrances, good lead time | Office lobby: 0.35 |
| **Center** | 0.5 | Balanced detection, general purpose | Hallway: 0.5 |
| **Standard** | 0.6-0.7 | Default, reduces false positives | Most cameras: 0.7 |
| **Late** | 0.8-0.9 | Narrow passages, exit confirmation | Doorways: 0.85 |

### Line Position Examples by Scenario

**Retail Store Setup:**
```bash
# cameras.txt
# Wide entrance - early detection for greeting customers
rtsp://url|yolo11x.pt|bidirectional_horizontal|5|0.25

# Narrow aisles - late detection to avoid false counts
rtsp://url|yolo11n.pt|left_to_right|10|0.8

# Checkout area - center line for balanced counting
rtsp://url|yolo11s.pt|top_to_bottom|5|0.5
```

**Office Building:**
```bash
# Main lobby - early detection at 30%
rtsp://url|yolo11l.pt|bidirectional_horizontal|5|0.3

# Elevator exit - late detection at 85%
rtsp://url|yolo11x.pt|bidirectional_vertical|5|0.85

# Stairwell - center detection at 50%
rtsp://url|yolo11n.pt|bottom_to_top|3|0.5
```

**Transportation Hub:**
```bash
# Wide platform entrance - very early at 20%
rtsp://url|yolo11x.pt|bidirectional_horizontal|8|0.2

# Turnstile exit - very late at 90%
rtsp://url|yolo11s.pt|top_to_bottom|10|0.9

# Escalator top - standard at 70%
rtsp://url|yolo11n.pt|bidirectional_horizontal|5|0.7
```

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

### MQTT Connection Issues

If running multiple instances causes MQTT disconnections:
- Use the single-instance solution with per-camera models
- Or use different access tokens for each instance

### Stream Connection Issues

1. **Test RTSP URL:**
   ```bash
   ffplay "rtsp://username:password@ip:port/path"
   ```

2. **Check Network:**
   ```bash
   ping camera_ip
   ```

### Performance Optimization

**For Multiple Models:**
- Models are cached in memory after first load
- GPU memory usage increases with model variety
- Monitor with `nvidia-smi -l 1`

**Memory Management:**
```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Limit GPU memory growth (add to script)
import torch
torch.cuda.set_per_process_memory_fraction(0.8)
```

## 📊 Example Deployment Scenarios

### Retail Store Setup
```bash
# cameras.txt with per-camera line positions
# Main entrance - bidirectional with early detection (30%)
rtsp://admin:pass@192.168.1.10:554/entrance|yolo11x.pt|bidirectional_horizontal|5|0.3

# Side entrances - single direction with different line positions
rtsp://admin:pass@192.168.1.11:554/side1|yolo11n.pt|top_to_bottom|8|0.4
rtsp://admin:pass@192.168.1.12:554/side2|yolo11n.pt|bottom_to_top|8|0.6

# Checkout area - late detection to confirm exit (85%)
rtsp://admin:pass@192.168.1.13:554/checkout|yolo11s.pt|left_to_right|5|0.85

# Storage area - low traffic, late detection (90%)
rtsp://admin:pass@192.168.1.14:554/storage|yolo11n.pt|left_to_right|2|0.9
```

### Office Building Setup
```bash
# cameras.txt with strategic line placements
# Lobby - high traffic bidirectional, early detection (25%)
rtsp://admin:pass@10.0.0.10:554/lobby|yolo11l.pt|bidirectional_horizontal|5|0.25

# Elevator areas - medium traffic, center lines (50%)
rtsp://admin:pass@10.0.0.11:554/elevator1|yolo11s.pt|bidirectional_vertical|5|0.5
rtsp://admin:pass@10.0.0.12:554/elevator2|yolo11s.pt|bidirectional_vertical|5|0.5

# Stairwells - directional counting with different line positions
rtsp://admin:pass@10.0.0.13:554/stairs_up|yolo11n.pt|bottom_to_top|3|0.3
rtsp://admin:pass@10.0.0.14:554/stairs_down|yolo11n.pt|top_to_bottom|3|0.7

# Conference room entrances - very early detection (20%)
rtsp://admin:pass@10.0.0.15:554/conf_room1|yolo11x.pt|bidirectional_horizontal|5|0.2
```

### Transportation Hub Setup
```bash
# cameras.txt for transit station
# Main entrance - wide area, very early detection (15%)
rtsp://admin:pass@172.16.0.10:554/main|yolo11l.pt|bidirectional_horizontal|10|0.15

# Turnstiles - late detection after payment (90%)
rtsp://admin:pass@172.16.0.11:554/turnstile1|yolo11s.pt|top_to_bottom|8|0.9
rtsp://admin:pass@172.16.0.12:554/turnstile2|yolo11s.pt|top_to_bottom|8|0.9

# Platform access - bidirectional with center line (50%)
rtsp://admin:pass@172.16.0.13:554/platform_a|yolo11x.pt|bidirectional_vertical|5|0.5

# Emergency exits - early warning detection (20%)
rtsp://admin:pass@172.16.0.14:554/emergency1|yolo11n.pt|bottom_to_top|5|0.2
```

## 🔒 Security Considerations

1. **Secure Credentials:**
   ```bash
   chmod 600 cameras.txt  # Restrict file permissions
   ```

2. **Environment Variables:**
   ```bash
   export TB_HOST="192.168.1.11"
   export TB_TOKEN="your_token"
   ```

3. **Network Security:** 
   - Use VPN for remote cameras
   - Implement firewall rules for RTSP ports
   - Use strong passwords for cameras

## 📈 Performance Metrics

Typical performance on RTX 3070:
- **yolo11n.pt**: ~120 FPS per stream
- **yolo11x.pt**: ~30 FPS per stream
- **yolo11l.pt**: ~15 FPS per stream

With 4 cameras at 5 FPS processing:
- GPU Usage: 40-60%
- GPU Memory: 4-6 GB
- CPU Usage: 20-30%

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for tracking algorithm
- [ThingsBoard](https://thingsboard.io/) for IoT platform

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review example configurations
3. Open an issue with:
   - Camera configuration used
   - Error messages
   - GPU/System specifications

---

**Note:** Always ensure you have permission to monitor areas with cameras and comply with local privacy laws and regulations.
