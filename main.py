import cv2
import mediapipe as mp
import numpy as np
import time
import psutil
import os
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# -----------------------------
# CONFIGURATION
# -----------------------------
EAR_THRESHOLD = 0.23
DROWSINESS_TIME = 2.0
WARNING_TIME = 1.5
MAX_HISTORY = 100

# Color scheme (BGR format)
COLOR_PRIMARY = (0, 255, 200)      # Cyan
COLOR_WARNING = (0, 165, 255)      # Orange
COLOR_DANGER = (0, 0, 255)         # Red
COLOR_SUCCESS = (0, 255, 0)        # Green
COLOR_TEXT = (255, 255, 255)       # White
COLOR_ACCENT = (255, 0, 255)       # Magenta

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Face oval for visualization
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    pts = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


# -----------------------------
# PROCESS-SPECIFIC RESOURCE MONITOR
# -----------------------------
class ProcessMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.cpu_history = deque(maxlen=MAX_HISTORY)
        self.ram_history = deque(maxlen=MAX_HISTORY)
        self.timestamps = deque(maxlen=MAX_HISTORY)
        
    def update(self):
        """Update resource usage for this process only"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        ram_mb = self.process.memory_info().rss / (1024 * 1024)
        
        self.cpu_history.append(cpu_percent)
        self.ram_history.append(ram_mb)
        self.timestamps.append(time.time())
        
        return {
            'cpu': cpu_percent,
            'ram': ram_mb,
            'threads': self.process.num_threads()
        }
    
    def get_stats(self):
        if not self.cpu_history:
            return None
        return {
            'cpu_avg': np.mean(self.cpu_history),
            'cpu_current': self.cpu_history[-1],
            'ram_avg': np.mean(self.ram_history),
            'ram_current': self.ram_history[-1],
            'ram_peak': max(self.ram_history)
        }


# -----------------------------
# ENHANCED VISUALIZATION
# -----------------------------
class DrowsinessDetector:
    def __init__(self):
        self.monitor = ProcessMonitor()
        self.ear_history = deque(maxlen=MAX_HISTORY)
        self.closed_start = None
        self.blink_count = 0
        self.last_blink = time.time()
        self.alert_triggered = False
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
    def draw_eye_landmarks(self, frame, landmarks, eye_indices, color):
        """Draw eye contours with glow effect"""
        h, w = frame.shape[:2]
        pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in eye_indices]
        pts = np.array(pts, np.int32)
        
        # Glow effect
        cv2.polylines(frame, [pts], True, (50, 50, 50), 5)
        cv2.polylines(frame, [pts], True, color, 2)
        
        # Draw points
        for pt in pts:
            cv2.circle(frame, pt, 2, color, -1)
    
    def draw_face_mesh(self, frame, face_landmarks):
        """Draw beautiful face mesh with selective connections"""
        h, w = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        
        # Draw face oval
        oval_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in FACE_OVAL]
        oval_pts = np.array(oval_pts, np.int32)
        cv2.polylines(frame, [oval_pts], True, COLOR_ACCENT, 1, cv2.LINE_AA)
        
        # Draw eyes
        self.draw_eye_landmarks(frame, landmarks, LEFT_EYE, COLOR_PRIMARY)
        self.draw_eye_landmarks(frame, landmarks, RIGHT_EYE, COLOR_PRIMARY)
        
        # Draw iris
        for iris_idx in LEFT_IRIS + RIGHT_IRIS:
            pt = (int(landmarks[iris_idx][0]), int(landmarks[iris_idx][1]))
            cv2.circle(frame, pt, 1, COLOR_SUCCESS, -1)
        
        return landmarks
    
    def draw_monitoring_panel(self, frame):
        """Draw resource monitoring panel"""
        stats = self.monitor.get_stats()
        if not stats:
            return
        
        panel_height = 200
        panel_width = 350
        x_offset = frame.shape[1] - panel_width - 20
        y_offset = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset), 
                     (x_offset + panel_width, y_offset + panel_height),
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     COLOR_PRIMARY, 2)
        
        # Title
        cv2.putText(frame, "SYSTEM MONITOR", (x_offset + 10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PRIMARY, 2)
        
        y = y_offset + 60
        
        # CPU Usage
        cpu_bar_width = int((stats['cpu_current'] / 100) * 300)
        cv2.rectangle(frame, (x_offset + 10, y), 
                     (x_offset + 310, y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (x_offset + 10, y),
                     (x_offset + 10 + cpu_bar_width, y + 20), 
                     COLOR_SUCCESS if stats['cpu_current'] < 50 else COLOR_WARNING, -1)
        cv2.putText(frame, f"CPU: {stats['cpu_current']:.1f}%", 
                   (x_offset + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, COLOR_TEXT, 1)
        
        y += 40
        
        # RAM Usage
        ram_bar_width = int((stats['ram_current'] / (stats['ram_peak'] + 1)) * 300)
        cv2.rectangle(frame, (x_offset + 10, y),
                     (x_offset + 310, y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (x_offset + 10, y),
                     (x_offset + 10 + ram_bar_width, y + 20),
                     COLOR_SUCCESS if stats['ram_current'] < 200 else COLOR_WARNING, -1)
        cv2.putText(frame, f"RAM: {stats['ram_current']:.1f} MB",
                   (x_offset + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, COLOR_TEXT, 1)
        
        y += 50
        
        # FPS
        if self.fps_history:
            fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {fps:.1f}", (x_offset + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
        
        y += 30
        
        # Blink count
        cv2.putText(frame, f"Blinks: {self.blink_count}", (x_offset + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
    
    def draw_ear_graph(self, frame):
        """Draw EAR trend graph"""
        if len(self.ear_history) < 2:
            return
        
        graph_height = 100
        graph_width = 400
        x_offset = 20
        y_offset = frame.shape[0] - graph_height - 20
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset),
                     (x_offset + graph_width, y_offset + graph_height),
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x_offset, y_offset),
                     (x_offset + graph_width, y_offset + graph_height),
                     COLOR_PRIMARY, 2)
        
        # Title
        cv2.putText(frame, "EAR TREND", (x_offset + 10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRIMARY, 1)
        
        # Threshold line
        threshold_y = int(y_offset + graph_height - (EAR_THRESHOLD / 0.5) * (graph_height - 30))
        cv2.line(frame, (x_offset + 10, threshold_y),
                (x_offset + graph_width - 10, threshold_y),
                COLOR_DANGER, 1, cv2.LINE_AA)
        
        # Plot EAR values
        ear_array = list(self.ear_history)
        for i in range(1, len(ear_array)):
            x1 = int(x_offset + 10 + (i - 1) * (graph_width - 20) / MAX_HISTORY)
            x2 = int(x_offset + 10 + i * (graph_width - 20) / MAX_HISTORY)
            y1 = int(y_offset + graph_height - 10 - (ear_array[i-1] / 0.5) * (graph_height - 30))
            y2 = int(y_offset + graph_height - 10 - (ear_array[i] / 0.5) * (graph_height - 30))
            
            color = COLOR_SUCCESS if ear_array[i] > EAR_THRESHOLD else COLOR_DANGER
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    def draw_status_indicator(self, frame, ear_value, elapsed_time):
        """Draw large status indicator"""
        center_x = frame.shape[1] // 2
        center_y = 100
        
        if ear_value < EAR_THRESHOLD:
            if elapsed_time > DROWSINESS_TIME:
                # DANGER
                status_text = "DROWSINESS ALERT!"
                status_color = COLOR_DANGER
                radius = 60
            elif elapsed_time > WARNING_TIME:
                # WARNING
                status_text = "WARNING"
                status_color = COLOR_WARNING
                radius = 50
            else:
                # Eyes closed
                status_text = "EYES CLOSED"
                status_color = COLOR_WARNING
                radius = 40
            
            # Pulsing effect
            pulse = int(10 * np.sin(time.time() * 10))
            radius += pulse
            
            # Draw circle
            cv2.circle(frame, (center_x, center_y), radius, status_color, 3)
            cv2.circle(frame, (center_x, center_y), radius - 5, status_color, -1, cv2.LINE_AA)
            
            # Add glow
            overlay = frame.copy()
            cv2.circle(overlay, (center_x, center_y), radius + 20, status_color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Text
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(frame, status_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
            cv2.putText(frame, status_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_TEXT, 2)
        else:
            # All good
            status_text = "ALERT"
            status_color = COLOR_SUCCESS
            cv2.circle(frame, (center_x, center_y), 30, status_color, 2)
            
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(frame, status_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    def process_frame(self, frame):
        """Main processing function"""
        current_time = time.time()
        
        # Calculate FPS
        fps = 1.0 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        # Update resource monitoring
        self.monitor.update()
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        elapsed_time = 0
        ear_value = 0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh
            landmarks = self.draw_face_mesh(frame, face_landmarks)
            
            # Calculate EAR
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear_value = (left_ear + right_ear) / 2.0
            
            self.ear_history.append(ear_value)
            
            # Detect blinks
            if ear_value < EAR_THRESHOLD:
                if self.closed_start is None:
                    self.closed_start = current_time
                    if current_time - self.last_blink > 0.3:  # Avoid counting same blink
                        self.blink_count += 1
                        self.last_blink = current_time
                elapsed_time = current_time - self.closed_start
            else:
                self.closed_start = None
            
            # Draw status indicator
            self.draw_status_indicator(frame, ear_value, elapsed_time)
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {ear_value:.3f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PRIMARY, 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_DANGER, 2)
        
        # Draw monitoring panel
        self.draw_monitoring_panel(frame)
        
        # Draw EAR graph
        self.draw_ear_graph(frame)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        return frame


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    print("=" * 60)
    print("ADVANCED DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print(f"Process ID: {os.getpid()}")
    print(f"EAR Threshold: {EAR_THRESHOLD}")
    print(f"Drowsiness Time: {DROWSINESS_TIME}s")
    print("=" * 60)
    print("\nPress 'q' to quit")
    print("Press 's' to save screenshot")
    print()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = DrowsinessDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        frame = detector.process_frame(frame)
        
        # Display
        cv2.imshow("Advanced Driver Drowsiness Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    stats = detector.monitor.get_stats()
    if stats:
        print("\n" + "=" * 60)
        print("SESSION STATISTICS")
        print("=" * 60)
        print(f"Average CPU Usage: {stats['cpu_avg']:.2f}%")
        print(f"Average RAM Usage: {stats['ram_avg']:.2f} MB")
        print(f"Peak RAM Usage: {stats['ram_peak']:.2f} MB")
        print(f"Total Blinks: {detector.blink_count}")
        print("=" * 60)


if __name__ == "__main__":
    main()