"""
Application Flask pour la détection de somnolence en temps réel
Fichier: app.py
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import json

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'driver_detector.pt'  # Votre modèle exporté  # Assurez-vous que best.pt est dans le même dossier
model = YOLO(MODEL_PATH)

STATES = {
    'normal': {
        'text': 'DRIVING NORMAL',
        'emoji': '🚗🙂',
        'color': '#00FF00',
        'bg_color': '#10b981'
    },
    'fatigue': {
        'text': 'FATIGUE ALERT',
        'emoji': '😮‍💨⚠️',
        'color': '#FFFF00',
        'bg_color': '#f59e0b'
    },
    'sleeping': {
        'text': 'FALLING ASLEEP ALERT',
        'emoji': '😴🚨',
        'color': '#FF0000',
        'bg_color': '#ef4444'
    },
    'drowsiness': {
        'text': 'DROWSINESS ALERT',
        'emoji': '😪⚠️',
        'color': '#FF8C00',
        'bg_color': '#f97316'
    }
}

# ============================================================================
# CLASSE TRACKER
# ============================================================================

class DrowsinessTracker:
    def __init__(self, fps_window=100, blink_window=60):
        self.fps_history = deque(maxlen=fps_window)
        self.fps_timestamps = deque(maxlen=fps_window)
        self.blink_history = deque(maxlen=blink_window)
        self.blink_timestamps = deque(maxlen=blink_window)
        self.total_blinks = 0
        self.last_eye_state = None
        self.blink_in_progress = False
        self.start_time = time.time()
        self.current_state = None
        
    def update_eyes_state(self, left_open, right_open, left_closed, right_closed):
        current_time = time.time()
        both_open = left_open and right_open
        both_closed = left_closed and right_closed
        
        current_state = None
        if both_open:
            current_state = 'open'
        elif both_closed:
            current_state = 'closed'
        
        if self.last_eye_state == 'open' and current_state == 'closed':
            self.blink_in_progress = True
        elif self.last_eye_state == 'closed' and current_state == 'open' and self.blink_in_progress:
            self.total_blinks += 1
            self.blink_in_progress = False
            self.blink_timestamps.append(current_time)
            
        self.last_eye_state = current_state
        
    def update_fps(self, fps):
        self.fps_history.append(fps)
        self.fps_timestamps.append(time.time())
        
    def get_blink_rate(self, window_seconds=60):
        if len(self.blink_timestamps) < 2:
            return 0.0
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        recent_blinks_times = [t for t in self.blink_timestamps if t >= cutoff_time]
        recent_blinks = len(recent_blinks_times)
        
        if recent_blinks > 0 and len(recent_blinks_times) > 0:
            time_span = current_time - min(recent_blinks_times)
            if time_span > 0:
                blink_rate = (recent_blinks / time_span) * 60
                self.blink_history.append(blink_rate)
                return blink_rate
        
        self.blink_history.append(0.0)
        return 0.0
    
    def get_average_fps(self):
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_session_duration(self):
        return time.time() - self.start_time
    
    def get_stats_dict(self):
        duration = self.get_session_duration()
        return {
            'total_blinks': self.total_blinks,
            'current_fps': self.fps_history[-1] if self.fps_history else 0,
            'average_fps': self.get_average_fps(),
            'blink_rate': self.get_blink_rate(),
            'duration_seconds': int(duration),
            'duration_formatted': f"{int(duration//60):02d}:{int(duration%60):02d}",
            'current_state': self.current_state,
            'fps_history': list(self.fps_history)[-30:],
            'blink_history': list(self.blink_history)[-30:]
        }

# Initialisation globale
tracker = DrowsinessTracker()
camera = None

# ============================================================================
# FONCTIONS DE DÉTECTION
# ============================================================================

def analyze_driver_state(detections, class_names, tracker):
    has_yawn = False
    has_no_yawn = False
    left_eye_open = False
    right_eye_open = False
    left_eye_closed = False
    right_eye_closed = False
    
    for det in detections:
        cls_id = int(det.cls[0])
        class_name = class_names[cls_id]
        conf = float(det.conf[0])
        
        if conf < 0.4:
            continue
        
        if class_name == 'yawn':
            has_yawn = True
        elif class_name == 'no_yawn':
            has_no_yawn = True
        elif class_name == 'open_eyeL':
            left_eye_open = True
        elif class_name == 'open_eyeR':
            right_eye_open = True
        elif class_name == 'close_eyeL':
            left_eye_closed = True
        elif class_name == 'close_eyeR':
            right_eye_closed = True
    
    tracker.update_eyes_state(left_eye_open, right_eye_open, 
                              left_eye_closed, right_eye_closed)
    
    both_eyes_open = left_eye_open and right_eye_open
    both_eyes_closed = left_eye_closed and right_eye_closed
    
    if both_eyes_open and has_no_yawn and not has_yawn:
        return 'normal'
    elif both_eyes_open and has_yawn:
        return 'fatigue'
    elif both_eyes_closed and has_no_yawn and not has_yawn:
        return 'sleeping'
    elif both_eyes_closed and has_yawn:
        return 'drowsiness'
    
    return None

def generate_frames():
    global camera, tracker
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Prédiction
        results = model.predict(source=frame, conf=0.4, iou=0.5, imgsz=640, save=False, verbose=False)
        driver_state = analyze_driver_state(results[0].boxes, model.names, tracker)
        tracker.current_state = driver_state
        
        # Frame annotée
        annotated_frame = results[0].plot()
        
        # Calculer FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            tracker.update_fps(fps)
            fps_start_time = time.time()
            fps_counter = 0
        
        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ============================================================================
# ROUTES FLASK
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    stats = tracker.get_stats_dict()
    if stats['current_state'] and stats['current_state'] in STATES:
        state_info = STATES[stats['current_state']]
        stats['state_text'] = state_info['text']
        stats['state_emoji'] = state_info['emoji']
        stats['state_color'] = state_info['bg_color']
    else:
        stats['state_text'] = 'ANALYZING'
        stats['state_emoji'] = '🔄'
        stats['state_color'] = '#6b7280'
    
    return jsonify(stats)

@app.route('/stop')
def stop():
    global camera
    if camera:
        camera.release()
    return jsonify({'status': 'stopped'})

# ============================================================================
# LANCEMENT DU SERVEUR
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🚀 DROWSINESS DETECTION SYSTEM - WEB APPLICATION")
    print("="*70)
    print("\n✨ Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎯 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)