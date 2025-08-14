"""
Multi-stream people counter with per-camera model selection and bidirectional counting.
Supports different YOLO models and bidirectional counting for each camera.
"""

import threading
import cv2
import time
import os
import argparse
import json
import paho.mqtt.client as mqtt
from datetime import datetime
from pathlib import Path
import traceback
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import torch

class EnhancedPeopleCounter:
    """Enhanced people counter with bidirectional counting support."""
    
    def __init__(self, rtsp_url, model_path='yolo11x.pt', process_fps=5, 
                 direction='top_to_bottom', line_position_ratio=0.7):
        """
        Initialize enhanced counter.
        
        Args:
            rtsp_url: RTSP stream URL
            model_path: Path to YOLO model
            process_fps: Frames to process per second
            direction: Counting direction ('bidirectional_horizontal', 'bidirectional_vertical', or single direction)
            line_position_ratio: Position of line (0.0=top/left, 1.0=bottom/right, 0.7=default)
        """
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.process_fps = process_fps
        self.direction = direction
        self.line_position_ratio = line_position_ratio
        
        # Check if bidirectional counting is enabled
        self.is_bidirectional = direction in ['bidirectional_horizontal', 'bidirectional_vertical']
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"✓ GPU detected for {model_path}: {torch.cuda.get_device_name(0)}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            print(f"⚠ Running {model_path} on CPU (GPU not available)")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        
        # Separate tracking for bidirectional counting
        if self.is_bidirectional:
            # For bidirectional counting, track both directions
            self.crossed_ids_forward = set()  # top_to_bottom or left_to_right
            self.crossed_ids_backward = set()  # bottom_to_top or right_to_left
            
            # Separate counts for each direction
            self.counts = {
                'forward': {
                    'total': 0,
                    'minute': 0,
                    'hour': 0
                },
                'backward': {
                    'total': 0,
                    'minute': 0,
                    'hour': 0
                }
            }
        else:
            # Single direction counting
            self.crossed_ids = set()
            self.people_count = 0
            self.minute_count = 0
            self.hour_count = 0
            self.total_count = 0
        
        # Time tracking
        self.current_hour = datetime.now().hour
        self.program_start_time = time.time()
        
        # FPS tracking
        self.source_fps = 0
        self.actual_process_fps = 0
        self.frame_times = []
        
        # Colors
        self.line_color = (0, 255, 0)
        self.box_color = (255, 0, 0)
        self.text_color = (255, 255, 255)
        self.track_color = (255, 255, 0)
        self.forward_color = (0, 255, 255)  # Cyan for forward
        self.backward_color = (255, 0, 255)  # Magenta for backward
        
        # Video properties
        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.line_position = None
        self.is_horizontal_line = None
        
        # Frame processing control
        self.last_process_time = 0
        self.process_interval = 1.0 / process_fps
        self.last_frame = None
        
    def initialize_video(self):
        """Initialize video capture with error handling."""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                return False
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # Set line position based on direction and ratio
            if self.direction in ['top_to_bottom', 'bottom_to_top', 'bidirectional_horizontal']:
                self.is_horizontal_line = True
                self.line_position = int(self.frame_height * self.line_position_ratio)
            else:
                self.is_horizontal_line = False
                self.line_position = int(self.frame_width * self.line_position_ratio)
            
            return True
        except Exception as e:
            print(f"Error initializing video: {e}")
            return False
    
    def draw_line(self, frame):
        """Draw the counting line on the frame with direction indicators."""
        if self.is_horizontal_line:
            cv2.line(frame, (0, self.line_position), (self.frame_width, self.line_position), 
                     self.line_color, 2)
            
            # Add direction arrows for bidirectional
            if self.is_bidirectional:
                # Arrow pointing down (forward)
                cv2.arrowedLine(frame, (50, self.line_position - 30), 
                              (50, self.line_position - 10), self.forward_color, 2)
                cv2.putText(frame, "IN", (60, self.line_position - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.forward_color, 2)
                
                # Arrow pointing up (backward)
                cv2.arrowedLine(frame, (120, self.line_position + 30), 
                              (120, self.line_position + 10), self.backward_color, 2)
                cv2.putText(frame, "OUT", (130, self.line_position + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.backward_color, 2)
        else:
            cv2.line(frame, (self.line_position, 0), (self.line_position, self.frame_height), 
                     self.line_color, 2)
            
            # Add direction arrows for bidirectional
            if self.is_bidirectional:
                # Arrow pointing right (forward)
                cv2.arrowedLine(frame, (self.line_position - 30, 50), 
                              (self.line_position - 10, 50), self.forward_color, 2)
                cv2.putText(frame, "IN", (self.line_position - 35, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.forward_color, 2)
                
                # Arrow pointing left (backward)
                cv2.arrowedLine(frame, (self.line_position + 30, 120), 
                              (self.line_position + 10, 120), self.backward_color, 2)
                cv2.putText(frame, "OUT", (self.line_position + 15, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.backward_color, 2)
    
    def draw_counter(self, frame, show_minute_count=True):
        """Draw counter information with support for bidirectional counts."""
        overlay = frame.copy()
        
        if self.is_bidirectional:
            box_height = 240 if show_minute_count else 200
        else:
            box_height = 200 if show_minute_count else 180
            
        cv2.rectangle(overlay, (10, 10), (320, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Calculate runtime
        runtime = time.time() - self.program_start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        
        # Display information
        y_offset = 35
        info_lines = [f"Model: {os.path.basename(self.model_path)}"]
        
        if self.is_bidirectional:
            # Show bidirectional counts
            if self.direction == 'bidirectional_horizontal':
                forward_label, backward_label = "Down↓", "Up↑"
            else:
                forward_label, backward_label = "Right→", "Left←"
            
            info_lines.extend([
                f"═══ {forward_label} (IN) ═══",
                f"Total: {self.counts['forward']['total']}",
                f"Hour: {self.counts['forward']['hour']}"
            ])
            if show_minute_count:
                info_lines.append(f"Minute: {self.counts['forward']['minute']}")
            
            info_lines.extend([
                f"═══ {backward_label} (OUT) ═══",
                f"Total: {self.counts['backward']['total']}",
                f"Hour: {self.counts['backward']['hour']}"
            ])
            if show_minute_count:
                info_lines.append(f"Minute: {self.counts['backward']['minute']}")
            
            total_all = self.counts['forward']['total'] + self.counts['backward']['total']
            info_lines.append(f"═══ NET: {self.counts['forward']['total'] - self.counts['backward']['total']} ═══")
        else:
            # Single direction counts
            info_lines.extend([
                f"Total Count: {self.total_count}",
                f"This Hour: {self.hour_count}"
            ])
            if show_minute_count:
                info_lines.append(f"This Minute: {self.minute_count}")
        
        info_lines.extend([
            f"Active Tracks: {len(self.track_history)}",
            f"Runtime: {hours:02d}:{minutes:02d}"
        ])
        
        for line in info_lines:
            # Use different colors for different sections
            if "═══" in line:
                color = (0, 255, 255)  # Cyan for headers
            elif "IN" in line or "Down" in line or "Right" in line:
                color = self.forward_color
            elif "OUT" in line or "Up" in line or "Left" in line:
                color = self.backward_color
            elif "NET" in line:
                color = (0, 255, 0)  # Green for net
            else:
                color = self.text_color
                
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20
    
    def check_line_crossing(self, track_id, current_x, current_y):
        """Check if person crossed the line and in which direction."""
        history = self.track_history[track_id]
        if len(history) < 2:
            return None
        
        prev_positions = history[-5:] if len(history) >= 5 else history[:-1]
        
        if self.is_bidirectional:
            # Check both directions for bidirectional counting
            if self.direction == 'bidirectional_horizontal':
                # Check top to bottom (forward)
                if track_id not in self.crossed_ids_forward:
                    was_above = any(pos[1] < self.line_position for pos in prev_positions)
                    is_below = current_y > self.line_position
                    if was_above and is_below:
                        self.crossed_ids_forward.add(track_id)
                        return 'forward'
                
                # Check bottom to top (backward)
                if track_id not in self.crossed_ids_backward:
                    was_below = any(pos[1] > self.line_position for pos in prev_positions)
                    is_above = current_y < self.line_position
                    if was_below and is_above:
                        self.crossed_ids_backward.add(track_id)
                        return 'backward'
                        
            elif self.direction == 'bidirectional_vertical':
                # Check left to right (forward)
                if track_id not in self.crossed_ids_forward:
                    was_left = any(pos[0] < self.line_position for pos in prev_positions)
                    is_right = current_x > self.line_position
                    if was_left and is_right:
                        self.crossed_ids_forward.add(track_id)
                        return 'forward'
                
                # Check right to left (backward)
                if track_id not in self.crossed_ids_backward:
                    was_right = any(pos[0] > self.line_position for pos in prev_positions)
                    is_left = current_x < self.line_position
                    if was_right and is_left:
                        self.crossed_ids_backward.add(track_id)
                        return 'backward'
        else:
            # Single direction checking
            if track_id in self.crossed_ids:
                return None
            
            if self.direction == 'top_to_bottom':
                was_above = any(pos[1] < self.line_position for pos in prev_positions)
                is_below = current_y > self.line_position
                if was_above and is_below:
                    self.crossed_ids.add(track_id)
                    return 'single'
                    
            elif self.direction == 'bottom_to_top':
                was_below = any(pos[1] > self.line_position for pos in prev_positions)
                is_above = current_y < self.line_position
                if was_below and is_above:
                    self.crossed_ids.add(track_id)
                    return 'single'
                    
            elif self.direction == 'left_to_right':
                was_left = any(pos[0] < self.line_position for pos in prev_positions)
                is_right = current_x > self.line_position
                if was_left and is_right:
                    self.crossed_ids.add(track_id)
                    return 'single'
                    
            elif self.direction == 'right_to_left':
                was_right = any(pos[0] > self.line_position for pos in prev_positions)
                is_left = current_x < self.line_position
                if was_right and is_left:
                    self.crossed_ids.add(track_id)
                    return 'single'
        
        return None
    
    def increment_counts(self, direction):
        """Increment counts based on crossing direction."""
        if self.is_bidirectional:
            if direction == 'forward':
                self.counts['forward']['total'] += 1
                self.counts['forward']['minute'] += 1
                self.counts['forward']['hour'] += 1
            elif direction == 'backward':
                self.counts['backward']['total'] += 1
                self.counts['backward']['minute'] += 1
                self.counts['backward']['hour'] += 1
        else:
            self.people_count += 1
            self.minute_count += 1
            self.hour_count += 1
            self.total_count += 1
    
    def reset_minute_counts(self):
        """Reset minute counters."""
        if self.is_bidirectional:
            self.counts['forward']['minute'] = 0
            self.counts['backward']['minute'] = 0
        else:
            self.minute_count = 0
    
    def reset_hour_counts(self):
        """Reset hour counters."""
        if self.is_bidirectional:
            self.counts['forward']['hour'] = 0
            self.counts['backward']['hour'] = 0
        else:
            self.hour_count = 0
    
    def get_minute_counts(self):
        """Get minute counts for MQTT."""
        if self.is_bidirectional:
            return {
                'in': self.counts['forward']['minute'],
                'out': self.counts['backward']['minute'],
                'net': self.counts['forward']['minute'] - self.counts['backward']['minute']
            }
        else:
            return {'count': self.minute_count}
    
    def get_total_counts(self):
        """Get total counts."""
        if self.is_bidirectional:
            return {
                'in': self.counts['forward']['total'],
                'out': self.counts['backward']['total'],
                'net': self.counts['forward']['total'] - self.counts['backward']['total']
            }
        else:
            return {'total': self.total_count}
    
    def process_frame(self, frame):
        """Process frame with error handling."""
        try:
            current_time = time.time()
            
            # Track frame times for FPS calculation
            self.frame_times.append(current_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # Calculate actual process FPS
            if len(self.frame_times) > 1:
                time_diff = self.frame_times[-1] - self.frame_times[0]
                if time_diff > 0:
                    self.actual_process_fps = (len(self.frame_times) - 1) / time_diff
            
            if current_time - self.last_process_time < self.process_interval:
                return self.last_frame if self.last_frame is not None else frame
            
            self.last_process_time = current_time
            
            # Run YOLO detection and tracking
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml",
                conf=0.3,
                iou=0.5,
                classes=[0],  # Person class
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=False  # Reduce console output
            )
            
            if results[0].boxes is None:
                self.last_frame = frame.copy()
                return frame
            
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is None:
                self.last_frame = frame.copy()
                return frame
            
            track_ids = track_ids.int().cpu().tolist()
            
            # Process each detection
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                self.track_history[track_id].append((x, y))
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Check for line crossing
                crossing_direction = self.check_line_crossing(track_id, x, y)
                if crossing_direction:
                    self.increment_counts(crossing_direction)
                
                # Draw bounding box with appropriate color
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Color based on crossing status
                if self.is_bidirectional:
                    if track_id in self.crossed_ids_forward:
                        color = self.forward_color
                    elif track_id in self.crossed_ids_backward:
                        color = self.backward_color
                    else:
                        color = self.box_color
                else:
                    color = (0, 255, 255) if track_id in self.crossed_ids else self.box_color
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw track
                points = np.array(self.track_history[track_id], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(frame, [points], False, self.track_color, 2)
            
            # Clean up old tracks
            current_ids = set(track_ids)
            to_remove = [tid for tid in self.track_history if tid not in current_ids]
            for tid in to_remove:
                del self.track_history[tid]
            
            self.last_frame = frame.copy()
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame


class MultiStreamThingsBoard:
    """Multi-stream counter with ThingsBoard integration and bidirectional counting."""
    
    def __init__(self, urls_file, thingsboard_host='192.168.1.11', 
                 access_token='mAYztIXMLIris3zAIcsJ', process_fps=5, 
                 direction='top_to_bottom', default_model='yolo11x.pt', 
                 line_position_ratio=0.7, enable_thingsboard=True):
        """
        Initialize multi-stream counter with ThingsBoard.
        
        Args:
            urls_file: Path to file with RTSP URLs and optional model specifications
            thingsboard_host: ThingsBoard server IP
            access_token: ThingsBoard device access token
            process_fps: Frames to process per second
            direction: Counting direction
            default_model: Default YOLO model to use
            line_position_ratio: Line position (0.0-1.0)
            enable_thingsboard: Enable ThingsBoard telemetry
        """
        self.urls_file = urls_file
        self.thingsboard_host = thingsboard_host
        self.access_token = access_token
        self.default_process_fps = process_fps
        self.default_direction = direction
        self.default_model = default_model
        self.line_position_ratio = line_position_ratio
        self.enable_thingsboard = enable_thingsboard
        
        self.threads = []
        self.counters = {}
        self.running = True
        self.loaded_models = {}  # Cache for loaded models
        
        # MQTT setup (only if enabled)
        if self.enable_thingsboard:
            self.setup_mqtt()
        else:
            self.mqtt_client = None
            print("⚠ ThingsBoard telemetry disabled")
        
        # Load URLs
        self.stream_configs = self.load_urls_from_file()
        
        # Minute tracking for MQTT
        self.last_mqtt_minute = datetime.now().minute
        
    def setup_mqtt(self):
        """Setup MQTT client for ThingsBoard."""
        try:
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            self.mqtt_client.username_pw_set(self.access_token)
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            self.mqtt_client.connect(self.thingsboard_host, 1883, 60)
            self.mqtt_client.loop_start()
            print(f"✓ Connected to ThingsBoard at {self.thingsboard_host}")
        except Exception as e:
            print(f"⚠ ThingsBoard connection failed: {e}")
            self.mqtt_client = None
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            print("✓ MQTT connected successfully")
        else:
            print(f"⚠ MQTT connection failed with code {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        if rc != 0:
            print("⚠ MQTT disconnected unexpectedly")
    
    def load_urls_from_file(self):
        """
        Load RTSP URLs from file with optional per-camera model specification.
        
        File format supports:
        - Simple URL: rtsp://192.168.1.100/stream
        - URL with model: rtsp://192.168.1.100/stream|yolo11n.pt
        - URL with model and direction: rtsp://192.168.1.100/stream|yolo11n.pt|bidirectional_horizontal
        - Comments: # This is a comment
        """
        if not os.path.exists(self.urls_file):
            raise FileNotFoundError(f"URLs file not found: {self.urls_file}")
        
        configs = []
        with open(self.urls_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not lines:
            raise ValueError(f"No URLs found in {self.urls_file}")
        
        print(f"✓ Found {len(lines)} camera configurations")
        
        for idx, line in enumerate(lines):
            parts = line.split('|')
            url = parts[0].strip()
            
            # Parse optional model specification
            model = parts[1].strip() if len(parts) > 1 else self.default_model
            
            # Parse optional direction specification
            direction = parts[2].strip() if len(parts) > 2 else self.default_direction
            
            # Parse optional FPS specification
            process_fps = int(parts[3].strip()) if len(parts) > 3 else self.default_process_fps
            
            # Create unique camera name
            try:
                if '@' in url:
                    ip_part = url.split('@')[1].split(':')[0].replace('.', '_')
                    name = f"cam_{ip_part}"
                else:
                    name = f"cam_{idx + 1}"
            except:
                name = f"cam_{idx + 1}"
            
            config = {
                'name': name,
                'url': url,
                'model': model,
                'direction': direction,
                'process_fps': process_fps,
                'window_index': idx
            }
            configs.append(config)
            
            # Show direction type
            if direction in ['bidirectional_horizontal', 'bidirectional_vertical']:
                dir_display = f"{direction} (IN/OUT)"
            else:
                dir_display = direction
            
            print(f"  {name}: model={model}, direction={dir_display}, fps={process_fps}")
        
        return configs
    
    def send_mqtt_counts(self):
        """Send counts to ThingsBoard with support for bidirectional counting."""
        if not self.mqtt_client:
            return
        
        telemetry = {}
        total_minute_in = 0
        total_minute_out = 0
        total_minute_single = 0
        has_data = False
        
        for name, counter in self.counters.items():
            counts = counter.get_minute_counts()
            totals = counter.get_total_counts()
            
            if counter.is_bidirectional:
                # Send bidirectional counts
                if counts['in'] > 0 or counts['out'] > 0:
                    telemetry[f"{name}_in"] = counts['in']
                    telemetry[f"{name}_out"] = counts['out']
                    telemetry[f"{name}_net"] = counts['net']
                    telemetry[f"{name}_total_in"] = totals['in']
                    telemetry[f"{name}_total_out"] = totals['out']
                    telemetry[f"{name}_total_net"] = totals['net']
                    
                    total_minute_in += counts['in']
                    total_minute_out += counts['out']
                    
                    print(f"  {name}: IN={counts['in']}, OUT={counts['out']}, NET={counts['net']} (Total IN={totals['in']}, OUT={totals['out']})")
                    has_data = True
            else:
                # Send single direction counts
                if counts['count'] > 0:
                    telemetry[f"{name}_count"] = counts['count']
                    telemetry[f"{name}_total"] = totals['total']
                    total_minute_single += counts['count']
                    print(f"  {name}: {counts['count']} people (Total: {totals['total']})")
                    has_data = True
            
            # Reset minute counts
            counter.reset_minute_counts()
        
        if has_data:
            # Add aggregate counts
            if total_minute_in > 0 or total_minute_out > 0:
                telemetry['total_in'] = total_minute_in
                telemetry['total_out'] = total_minute_out
                telemetry['total_net'] = total_minute_in - total_minute_out
            
            if total_minute_single > 0:
                telemetry['total_count'] = total_minute_single
            
            try:
                self.mqtt_client.publish('v1/devices/me/telemetry', json.dumps(telemetry), 1)
                print(f"→ Sent to ThingsBoard: IN={total_minute_in}, OUT={total_minute_out}, Single={total_minute_single}")
            except Exception as e:
                print(f"⚠ MQTT publish error: {e}")
    
    def send_hourly_counts(self):
        """Send hourly counts to ThingsBoard."""
        if not self.mqtt_client:
            return
        
        telemetry = {}
        total_hour_in = 0
        total_hour_out = 0
        total_hour_single = 0
        
        print(f"\n[{datetime.now().strftime('%H:00')}] Sending hourly counts to ThingsBoard...")
        
        for name, counter in self.counters.items():
            if counter.is_bidirectional:
                hour_in = counter.counts['forward']['hour']
                hour_out = counter.counts['backward']['hour']
                if hour_in > 0 or hour_out > 0:
                    telemetry[f"{name}_hourly_in"] = hour_in
                    telemetry[f"{name}_hourly_out"] = hour_out
                    telemetry[f"{name}_hourly_net"] = hour_in - hour_out
                    total_hour_in += hour_in
                    total_hour_out += hour_out
                    print(f"  {name}: IN={hour_in}, OUT={hour_out} this hour")
            else:
                if counter.hour_count > 0:
                    telemetry[f"{name}_hourly"] = counter.hour_count
                    total_hour_single += counter.hour_count
                    print(f"  {name}: {counter.hour_count} people this hour")
            
            # Reset hour counts
            counter.reset_hour_counts()
        
        if telemetry:
            # Add aggregate hourly counts
            if total_hour_in > 0 or total_hour_out > 0:
                telemetry['total_hourly_in'] = total_hour_in
                telemetry['total_hourly_out'] = total_hour_out
                telemetry['total_hourly_net'] = total_hour_in - total_hour_out
            
            if total_hour_single > 0:
                telemetry['total_hourly'] = total_hour_single
            
            try:
                self.mqtt_client.publish('v1/devices/me/telemetry', json.dumps(telemetry), 1)
                print(f"→ Sent hourly data: IN={total_hour_in}, OUT={total_hour_out}, Single={total_hour_single}")
            except Exception as e:
                print(f"⚠ MQTT hourly publish error: {e}")
    
    def mqtt_timer(self):
        """Timer thread to send MQTT data every minute."""
        if not self.enable_thingsboard:
            return
            
        last_hour = datetime.now().hour
        
        while self.running:
            current_time = datetime.now()
            current_minute = current_time.minute
            current_hour = current_time.hour
            
            # Send minute counts
            if current_minute != self.last_mqtt_minute:
                print(f"\n[{current_time.strftime('%H:%M')}] Sending counts to ThingsBoard...")
                self.send_mqtt_counts()
                self.last_mqtt_minute = current_minute
            
            # Send hourly counts
            if current_hour != last_hour:
                self.send_hourly_counts()
                last_hour = current_hour
            
            time.sleep(1)
    
    def process_stream_with_window(self, config):
        """Process a single stream with error recovery."""
        name = config['name']
        url = config['url']
        model = config['model']
        direction = config['direction']
        window_index = config.get('window_index', 0)
        
        # Show direction info
        if direction in ['bidirectional_horizontal', 'bidirectional_vertical']:
            print(f"✓ Starting {name} with model {model} (bidirectional counting)")
        else:
            print(f"✓ Starting {name} with model {model} ({direction})")
        
        # Window setup
        window_name = f"{name}"
        window_width, window_height = 800, 600
        windows_per_row = 3
        
        row = window_index // windows_per_row
        col = window_index % windows_per_row
        x_pos = col * (window_width + 10)
        y_pos = row * (window_height + 50)
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.moveWindow(window_name, x_pos, y_pos)
        
        retry_count = 0
        max_retries = 5
        
        while self.running:
            try:
                # Create/recreate counter with specified model and direction
                counter = EnhancedPeopleCounter(
                    rtsp_url=url,
                    model_path=model,
                    process_fps=config['process_fps'],
                    direction=direction,  # Use the specified direction
                    line_position_ratio=self.line_position_ratio
                )
                
                self.counters[name] = counter
                
                # Initialize video
                if not counter.initialize_video():
                    raise Exception("Failed to initialize video")
                
                print(f"✓ {name} connected ({counter.frame_width}x{counter.frame_height})")
                retry_count = 0
                
                # Frame processing loop
                frames_read = 0
                frames_processed = 0
                last_stats_time = time.time()
                source_frame_times = []
                
                while self.running:
                    # Read frame
                    ret, frame = counter.cap.read()
                    frames_read += 1
                    
                    # Track source FPS
                    current_time = time.time()
                    source_frame_times.append(current_time)
                    if len(source_frame_times) > 30:
                        source_frame_times.pop(0)
                    
                    if len(source_frame_times) > 1:
                        time_diff = source_frame_times[-1] - source_frame_times[0]
                        if time_diff > 0:
                            counter.source_fps = (len(source_frame_times) - 1) / time_diff
                    
                    if not ret:
                        raise Exception("Stream disconnected")
                    
                    # Process frame
                    current_time = time.time()
                    annotated_frame = counter.process_frame(frame)
                    
                    if annotated_frame is not None:
                        counter.draw_line(annotated_frame)
                        counter.draw_counter(annotated_frame, show_minute_count=self.enable_thingsboard)
                        
                        # Add camera name
                        cv2.putText(annotated_frame, name, (10, counter.frame_height - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Show frame
                        cv2.imshow(window_name, annotated_frame)
                    
                    # Check for quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print(f"✗ {name} closed by user")
                        self.running = False
                        break
                    elif key == ord('q'):
                        self.running = False
                        break
                    
                    time.sleep(0.001)
                    
            except Exception as e:
                retry_count += 1
                print(f"⚠ {name} error (retry {retry_count}/{max_retries}): {str(e)[:50]}")
                
                if retry_count >= max_retries:
                    print(f"✗ {name} failed after {max_retries} retries")
                    break
                
                # Show error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "RECONNECTING...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, error_frame)
                cv2.waitKey(1)
                
                # Wait before retry
                time.sleep(5)
                
            finally:
                if name in self.counters and hasattr(self.counters[name], 'cap'):
                    if self.counters[name].cap:
                        self.counters[name].cap.release()
        
        cv2.destroyWindow(window_name)
    
    def print_statistics(self):
        """Print periodic statistics with bidirectional support."""
        while self.running:
            time.sleep(60)  # Every minute
            
            if not self.running:
                break
            
            # Group counts by model and type
            model_counts = defaultdict(lambda: {'single': 0, 'in': 0, 'out': 0, 'cameras': []})
            
            for name, counter in self.counters.items():
                model = counter.model_path
                model_counts[model]['cameras'].append(name)
                
                if counter.is_bidirectional:
                    model_counts[model]['in'] += counter.counts['forward']['total']
                    model_counts[model]['out'] += counter.counts['backward']['total']
                else:
                    model_counts[model]['single'] += counter.total_count
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Statistics:")
            
            total_in = sum(data['in'] for data in model_counts.values())
            total_out = sum(data['out'] for data in model_counts.values())
            total_single = sum(data['single'] for data in model_counts.values())
            
            print(f"  Total: IN={total_in}, OUT={total_out}, Single={total_single}")
            
            for model, data in model_counts.items():
                model_name = os.path.basename(model)
                if data['in'] > 0 or data['out'] > 0:
                    print(f"  {model_name}: IN={data['in']}, OUT={data['out']} ({len(data['cameras'])} cameras)")
                if data['single'] > 0:
                    print(f"  {model_name}: {data['single']} ({len(data['cameras'])} cameras)")
    
    def run(self):
        """Start all streams and monitoring."""
        if not self.stream_configs:
            print("No streams to process!")
            return
        
        # Check GPU status
        if torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch CUDA: {torch.cuda.is_available()}")
        else:
            print("⚠ GPU not available, using CPU")
            print("  For GPU support, ensure CUDA PyTorch is installed")
        
        # Count models and directions being used
        models_in_use = set(config['model'] for config in self.stream_configs)
        directions_in_use = set(config['direction'] for config in self.stream_configs)
        bidirectional_count = sum(1 for config in self.stream_configs 
                                 if config['direction'] in ['bidirectional_horizontal', 'bidirectional_vertical'])
        
        print(f"\n{'='*50}")
        print(f"Configuration:")
        print(f"  Cameras: {len(self.stream_configs)}")
        print(f"  Bidirectional cameras: {bidirectional_count}")
        print(f"  Models in use: {', '.join(os.path.basename(m) for m in models_in_use)}")
        print(f"  Line position: {self.line_position_ratio:.0%} from top/left")
        print(f"  Default direction: {self.default_direction}")
        print(f"  Default process FPS: {self.default_process_fps}")
        
        if self.enable_thingsboard:
            print(f"\nThingsBoard:")
            print(f"  Server: {self.thingsboard_host}")
            print(f"  Token: {self.access_token[:10]}...")
        else:
            print(f"\nThingsBoard: Disabled (standalone mode)")
        
        print(f"{'='*50}\n")
        
        # Start stream threads
        for config in self.stream_configs:
            thread = threading.Thread(
                target=self.process_stream_with_window,
                args=(config,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            time.sleep(0.5)
        
        # Start MQTT timer thread
        mqtt_thread = threading.Thread(target=self.mqtt_timer, daemon=True)
        mqtt_thread.start()
        
        # Start statistics thread
        stats_thread = threading.Thread(target=self.print_statistics, daemon=True)
        stats_thread.start()
        
        # Wait for threads
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            print("\n✗ Stopping all streams...")
            self.running = False
        
        # Cleanup
        cv2.destroyAllWindows()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        # Final counts
        print(f"\n{'='*50}")
        print("Final counts by camera:")
        for name, counter in self.counters.items():
            model_name = os.path.basename(counter.model_path)
            if counter.is_bidirectional:
                totals = counter.get_total_counts()
                print(f"  {name} ({model_name}): IN={totals['in']}, OUT={totals['out']}, NET={totals['net']}")
            else:
                print(f"  {name} ({model_name}): {counter.total_count} total")
        print(f"{'='*50}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Multi-stream people counter with bidirectional counting and per-camera models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
URLs file format examples:
  Simple:                 rtsp://192.168.1.100/stream
  With model:             rtsp://192.168.1.100/stream|yolo11n.pt
  With direction:         rtsp://192.168.1.100/stream|yolo11x.pt|left_to_right
  Bidirectional:          rtsp://192.168.1.100/stream|yolo11x.pt|bidirectional_horizontal
  Full options:           rtsp://192.168.1.100/stream|yolo11x.pt|bidirectional_vertical|10
                          (URL | model | direction | FPS)

Direction options:
  Single directions:      top_to_bottom, bottom_to_top, left_to_right, right_to_left
  Bidirectional:          bidirectional_horizontal (counts both up/down)
                          bidirectional_vertical (counts both left/right)
        """
    )
    parser.add_argument('--urls-file', type=str, required=True,
                       help='Path to file with RTSP URLs and optional model specifications')
    parser.add_argument('--thingsboard-host', type=str, default='192.168.1.11',
                       help='ThingsBoard server IP (default: 192.168.1.11)')
    parser.add_argument('--access-token', type=str, default='mAYztIXMLIris3zAIcsJ',
                       help='ThingsBoard device access token')
    parser.add_argument('--no-thingsboard', action='store_true',
                       help='Disable ThingsBoard telemetry (run standalone)')
    parser.add_argument('--process-fps', type=int, default=5,
                       help='Default frames to process per second (default: 5)')
    parser.add_argument('--direction', type=str, default='top_to_bottom',
                       choices=['top_to_bottom', 'bottom_to_top', 'left_to_right', 'right_to_left',
                               'bidirectional_horizontal', 'bidirectional_vertical'],
                       help='Default counting direction (default: top_to_bottom)')
    parser.add_argument('--model', type=str, default='yolo11x.pt',
                       help='Default YOLO model to use (default: yolo11x.pt)')
    parser.add_argument('--line-position', type=float, default=0.7,
                       help='Line position ratio (0.0-1.0, default: 0.7)')
    
    args = parser.parse_args()
    
    # Create and run
    counter = MultiStreamThingsBoard(
        urls_file=args.urls_file,
        thingsboard_host=args.thingsboard_host,
        access_token=args.access_token,
        process_fps=args.process_fps,
        direction=args.direction,
        default_model=args.model,
        line_position_ratio=args.line_position,
        enable_thingsboard=not args.no_thingsboard
    )
    
    counter.run()


if __name__ == "__main__":
    main()
