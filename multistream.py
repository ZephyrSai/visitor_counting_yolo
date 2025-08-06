"""
Multi-stream people counter with ThingsBoard MQTT integration.
Sends counts every minute and handles errors gracefully.
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
    """Enhanced people counter with MQTT and adjustable line position."""
    
    def __init__(self, rtsp_url, model_path='yolo12n.pt', process_fps=5, 
                 direction='top_to_bottom', line_position_ratio=0.75):
        """
        Initialize enhanced counter.
        
        Args:
            rtsp_url: RTSP stream URL
            model_path: Path to YOLO model
            process_fps: Frames to process per second
            direction: Counting direction
            line_position_ratio: Position of line (0.0=top/left, 1.0=bottom/right, 0.75=default)
        """
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.process_fps = process_fps
        self.direction = direction
        self.line_position_ratio = line_position_ratio
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            print("⚠ Running on CPU (GPU not available)")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.crossed_ids = set()
        self.people_count = 0
        self.minute_count = 0  # Count for current minute
        self.hour_count = 0    # Count for current hour
        self.total_count = 0   # Total count since start
        
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
            if self.direction in ['top_to_bottom', 'bottom_to_top']:
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
        """Draw the counting line on the frame."""
        if self.is_horizontal_line:
            cv2.line(frame, (0, self.line_position), (self.frame_width, self.line_position), 
                     self.line_color, 2)
        else:
            cv2.line(frame, (self.line_position, 0), (self.line_position, self.frame_height), 
                     self.line_color, 2)
    
    def draw_counter(self, frame, show_minute_count=True):
        """Draw counter information with FPS stats."""
        overlay = frame.copy()
        box_height = 180 if show_minute_count else 160
        cv2.rectangle(overlay, (10, 10), (300, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Calculate runtime
        runtime = time.time() - self.program_start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        
        # Display information
        y_offset = 35
        info_lines = [
            f"Total Count: {self.total_count}",
            f"This Hour: {self.hour_count}",
        ]
        
        if show_minute_count:
            info_lines.append(f"This Minute: {self.minute_count}")
        
        info_lines.extend([
            f"Active Tracks: {len(self.track_history)}",
            f"Source FPS: {self.source_fps:.1f}",
            f"Process FPS: {self.actual_process_fps:.1f} / {self.process_fps}",
            f"Runtime: {hours:02d}:{minutes:02d}"
        ])
        
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
            y_offset += 20
    
    def check_line_crossing(self, track_id, current_x, current_y):
        """Check if person crossed the line."""
        if track_id in self.crossed_ids:
            return False
        
        history = self.track_history[track_id]
        if len(history) < 2:
            return False
        
        prev_positions = history[-5:] if len(history) >= 5 else history[:-1]
        
        if self.direction == 'top_to_bottom':
            was_above = any(pos[1] < self.line_position for pos in prev_positions)
            is_below = current_y > self.line_position
            if was_above and is_below:
                self.crossed_ids.add(track_id)
                return True
                
        elif self.direction == 'bottom_to_top':
            was_below = any(pos[1] > self.line_position for pos in prev_positions)
            is_above = current_y < self.line_position
            if was_below and is_above:
                self.crossed_ids.add(track_id)
                return True
                
        elif self.direction == 'left_to_right':
            was_left = any(pos[0] < self.line_position for pos in prev_positions)
            is_right = current_x > self.line_position
            if was_left and is_right:
                self.crossed_ids.add(track_id)
                return True
                
        elif self.direction == 'right_to_left':
            was_right = any(pos[0] > self.line_position for pos in prev_positions)
            is_left = current_x < self.line_position
            if was_right and is_left:
                self.crossed_ids.add(track_id)
                return True
        
        return False
    
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
            
            # Run YOLO detection and tracking - FORCE GPU WITH INTEGER 0
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml",
                conf=0.3,
                iou=0.5,
                classes=[0],  # Person class
                device=0 if torch.cuda.is_available() else 'cpu',  # MUST BE INTEGER 0 FOR GPU
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
                if self.check_line_crossing(track_id, x, y):
                    self.people_count += 1
                    self.minute_count += 1
                    self.hour_count += 1
                    self.total_count += 1
                
                # Draw bounding box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
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
    """Multi-stream counter with ThingsBoard integration."""
    
    def __init__(self, urls_file, thingsboard_host='192.168.1.11', 
                 access_token='mAYztIXMLIris3zAIcsJ', process_fps=5, 
                 direction='top_to_bottom', model='yolo12n.pt', 
                 line_position_ratio=0.75, enable_thingsboard=True):
        """
        Initialize multi-stream counter with ThingsBoard.
        
        Args:
            urls_file: Path to file with RTSP URLs
            thingsboard_host: ThingsBoard server IP
            access_token: ThingsBoard device access token
            process_fps: Frames to process per second
            direction: Counting direction
            model: YOLO model to use
            line_position_ratio: Line position (0.0-1.0)
        """
        self.urls_file = urls_file
        self.thingsboard_host = thingsboard_host
        self.access_token = access_token
        self.default_process_fps = process_fps
        self.default_direction = direction
        self.default_model = model
        self.line_position_ratio = line_position_ratio
        self.enable_thingsboard = enable_thingsboard
        
        self.threads = []
        self.counters = {}
        self.running = True
        
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
        """Load RTSP URLs from file."""
        if not os.path.exists(self.urls_file):
            raise FileNotFoundError(f"URLs file not found: {self.urls_file}")
        
        configs = []
        with open(self.urls_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not lines:
            raise ValueError(f"No URLs found in {self.urls_file}")
        
        print(f"✓ Found {len(lines)} cameras")
        
        for idx, url in enumerate(lines):
            try:
                # Create unique camera name
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
                'model': self.default_model,
                'direction': self.default_direction,
                'process_fps': self.default_process_fps,
                'window_index': idx
            }
            configs.append(config)
        
        return configs
    
    def send_mqtt_counts(self):
        """Send counts to ThingsBoard and reset minute counters."""
        if not self.mqtt_client:
            return
        
        telemetry = {}
        total_minute_count = 0
        has_data = False
        
        for name, counter in self.counters.items():
            if counter.minute_count > 0:
                telemetry[f"{name}_count"] = counter.minute_count
                telemetry[f"{name}_total"] = counter.total_count
                total_minute_count += counter.minute_count
                print(f"  {name}: {counter.minute_count} people (Total: {counter.total_count})")
                has_data = True
                counter.minute_count = 0  # Reset minute counter
        
        if has_data:
            telemetry['total_count'] = total_minute_count
            telemetry['all_cameras_total'] = sum(c.total_count for c in self.counters.values())
            try:
                self.mqtt_client.publish('v1/devices/me/telemetry', json.dumps(telemetry), 1)
                print(f"→ Sent to ThingsBoard: {total_minute_count} this minute, {telemetry['all_cameras_total']} total")
            except Exception as e:
                print(f"⚠ MQTT publish error: {e}")
    
    def send_hourly_counts(self):
        """Send hourly counts to ThingsBoard."""
        if not self.mqtt_client:
            return
        
        telemetry = {}
        total_hour_count = 0
        
        print(f"\n[{datetime.now().strftime('%H:00')}] Sending hourly counts to ThingsBoard...")
        
        for name, counter in self.counters.items():
            if counter.hour_count > 0:
                telemetry[f"{name}_hourly"] = counter.hour_count
                total_hour_count += counter.hour_count
                print(f"  {name}: {counter.hour_count} people this hour")
                counter.hour_count = 0  # Reset hour counter
        
        if telemetry:
            telemetry['total_hourly'] = total_hour_count
            try:
                self.mqtt_client.publish('v1/devices/me/telemetry', json.dumps(telemetry), 1)
                print(f"→ Sent hourly data: {total_hour_count} total")
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
        window_index = config.get('window_index', 0)
        
        print(f"✓ Starting {name}")
        
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
                # Create/recreate counter
                counter = EnhancedPeopleCounter(
                    rtsp_url=url,
                    model_path=config['model'],
                    process_fps=config['process_fps'],
                    direction=config['direction'],
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
        """Print periodic statistics."""
        while self.running:
            time.sleep(60)  # Every minute
            
            if not self.running:
                break
            
            total = sum(c.people_count for c in self.counters.values())
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Total: {total} people")
    
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
        
        print(f"\n{'='*50}")
        print(f"Configuration:")
        print(f"  Cameras: {len(self.stream_configs)}")
        print(f"  Line position: {self.line_position_ratio:.0%} from top/left")
        print(f"  Direction: {self.default_direction}")
        print(f"  Process FPS: {self.default_process_fps}")
        print(f"  Model: {self.default_model}")
        
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
        print("Final counts:")
        for name, counter in self.counters.items():
            print(f"  {name}: {counter.people_count} total")
        print(f"{'='*50}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multi-stream people counter with ThingsBoard')
    parser.add_argument('--urls-file', type=str, required=True,
                       help='Path to file with RTSP URLs')
    parser.add_argument('--thingsboard-host', type=str, default='192.168.1.11',
                       help='ThingsBoard server IP (default: 192.168.1.11)')
    parser.add_argument('--access-token', type=str, default='mAYztIXMLIris3zAIcsJ',
                       help='ThingsBoard device access token (default: mAYztIXMLIris3zAIcsJ)')
    parser.add_argument('--no-thingsboard', action='store_true',
                       help='Disable ThingsBoard telemetry (run standalone)')
    parser.add_argument('--process-fps', type=int, default=5,
                       help='Frames to process per second (default: 5)')
    parser.add_argument('--direction', type=str, default='top_to_bottom',
                       choices=['top_to_bottom', 'bottom_to_top', 'left_to_right', 'right_to_left'],
                       help='Counting direction (default: top_to_bottom)')
    parser.add_argument('--model', type=str, default='yolo12n.pt',
                       help='YOLO model to use (default: yolo12n.pt)')
    parser.add_argument('--line-position', type=float, default=0.75,
                       help='Line position ratio (0.0-1.0, default: 0.75)')
    
    args = parser.parse_args()
    
    # Create and run
    counter = MultiStreamThingsBoard(
        urls_file=args.urls_file,
        thingsboard_host=args.thingsboard_host,
        access_token=args.access_token,
        process_fps=args.process_fps,
        direction=args.direction,
        model=args.model,
        line_position_ratio=args.line_position,
        enable_thingsboard=not args.no_thingsboard
    )
    
    counter.run()


if __name__ == "__main__":
    main()
