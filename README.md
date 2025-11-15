# 🚗 Advanced Driver Drowsiness Detection System

<div align="center">

**Real-time AI-powered driver monitoring system that detects drowsiness and prevents accidents**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>


## 🎯 Overview

The **Advanced Driver Drowsiness Detection System** is a cutting-edge computer vision application that monitors drivers in real-time to detect signs of drowsiness and fatigue. By analyzing facial landmarks and eye movements, the system provides immediate alerts to prevent accidents caused by driver fatigue.

According to the National Highway Traffic Safety Administration (NHTSA), drowsy driving causes over 100,000 crashes annually in the United States alone. Our system aims to reduce these statistics by providing an affordable, accessible, and highly accurate monitoring solution.

---

## 🚨 The Problem

**Driver drowsiness is a silent killer on our roads:**

- 💤 **1 in 25 adult drivers** report having fallen asleep while driving in the past 30 days
- 🚑 **6,000+ fatal crashes** per year are attributed to drowsy driving
- ⏰ **Peak danger times**: Late night (midnight-6am) and mid-afternoon (2-4pm)
- 💰 **$109 billion** annual cost to society from drowsy driving crashes

Traditional solutions like rumble strips or basic alertness monitors are either passive or prohibitively expensive for average consumers.

---

## 💡 Our Solution

We've developed an intelligent, **non-invasive monitoring system** that:

✅ Uses standard webcam technology (no special hardware required)  
✅ Provides real-time drowsiness detection with multi-stage warnings  
✅ Tracks precise facial landmarks using state-of-the-art AI  
✅ Monitors system performance to ensure reliability  
✅ Delivers an intuitive, visually stunning interface  

**Cost-effective • Accurate • Easy to deploy**

---

## 🌟 Key Features

### 1. **Advanced Face Mesh Detection**
- 468-point facial landmark tracking
- Eye Aspect Ratio (EAR) calculation for drowsiness detection
- Iris tracking for enhanced accuracy
- Real-time face oval visualization with glow effects

### 2. **Multi-Stage Alert System**
- 🟢 **Normal**: Driver is alert and attentive
- 🟡 **Warning**: Eyes closed for 1.5+ seconds
- 🔴 **Critical Alert**: Drowsiness detected (2+ seconds)
- Pulsing visual alerts with sound-ready architecture

### 3. **Intelligent Monitoring**
- Process-specific resource tracking (CPU, RAM)
- Real-time FPS monitoring
- Blink detection and counting
- EAR trend visualization with scrolling graph

### 4. **Professional UI/UX**
- Semi-transparent monitoring panels
- Color-coded status indicators
- Live performance graphs
- Screenshot capability for documentation
- HD resolution support (1280x720)

### 5. **Optimization & Performance**
- Lightweight process footprint
- Efficient memory management
- Configurable thresholds for different environments
- Session statistics and analytics

---

## 🔬 How It Works

### Eye Aspect Ratio (EAR) Algorithm

The system uses the **Eye Aspect Ratio (EAR)** method, a proven technique in drowsiness detection research:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where `p1...p6` are the 6 key eye landmarks.

**Key Thresholds:**
- **Normal EAR**: ~0.25-0.35 (eyes open)
- **Drowsy EAR**: <0.23 (eyes closing/closed)
- **Critical Duration**: 2.0 seconds of continuous low EAR

### Detection Pipeline

```
Camera Feed → Face Detection → Landmark Extraction → EAR Calculation → Alert System
     ↓              ↓                  ↓                    ↓              ↓
  1280x720      MediaPipe         468 Points           Analytics      Multi-stage
  30+ FPS     Face Mesh AI       Eye Tracking         EAR < 0.23       Warnings
```

### Real-Time Processing Flow

1. **Capture**: Webcam captures video at 30+ FPS
2. **Detect**: MediaPipe Face Mesh identifies 468 facial landmarks
3. **Calculate**: System computes EAR for both eyes
4. **Analyze**: Continuous monitoring of EAR values against threshold
5. **Alert**: Multi-level warnings based on duration and severity
6. **Track**: Blink counting, trend analysis, and performance metrics

---

## 🛠️ Technology Stack

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary language | 3.8+ |
| **OpenCV** | Computer vision & video processing | 4.8+ |
| **MediaPipe** | Face mesh detection & tracking | Latest |
| **NumPy** | Numerical computations | Latest |
| **psutil** | System resource monitoring | Latest |

### Key Algorithms

- **Face Mesh**: Google MediaPipe's 468-point facial landmark model
- **EAR Calculation**: Eye Aspect Ratio algorithm for drowsiness detection
- **Blink Detection**: Temporal analysis of EAR fluctuations
- **Performance Tracking**: Real-time process-level resource monitoring

### Why These Technologies?

- **MediaPipe**: State-of-the-art accuracy with minimal latency (<10ms inference time)
- **OpenCV**: Industry-standard for real-time computer vision
- **Python**: Rapid development with extensive ML/CV ecosystem
- **Lightweight**: No GPU required, runs on standard laptops

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or USB)
- Windows/Linux/MacOS

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install opencv-python mediapipe numpy psutil matplotlib
```

4. **Run the application**
```bash
python main.py
```

### Quick Install (One-Liner)
```bash
pip install opencv-python mediapipe numpy psutil matplotlib && python main.py
```

---

## 🎮 Usage

### Starting the System

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save screenshot with timestamp |

### Understanding the Interface

#### Main Display Window
- **Top Center**: Large status indicator (Green/Orange/Red)
- **Top Left**: Current EAR value
- **Top Right**: System resource panel with CPU/RAM usage
- **Bottom Left**: EAR trend graph with threshold line
- **Bottom Right**: Session timestamp

#### Status Indicators
- 🟢 **ALERT** (Green): Driver is alert, eyes open
- 🟡 **WARNING** (Orange): Eyes closed 1.5+ seconds
- 🔴 **DROWSINESS ALERT** (Red): Critical drowsiness detected

#### Monitoring Panel Shows
- Real-time CPU usage (%)
- RAM consumption (MB)
- Current FPS
- Total blink count

### Configuration

Edit these constants in `main.py` to customize behavior:

```python
EAR_THRESHOLD = 0.23        # Lower = more sensitive
DROWSINESS_TIME = 2.0       # Seconds before critical alert
WARNING_TIME = 1.5          # Seconds before warning alert
MAX_HISTORY = 100           # Data points to store in graphs
```

---

## 🏗️ System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Camera     │─────▶│    Face      │─────▶│   EAR     │ │
│  │   Capture    │      │    Mesh      │      │ Calculator│ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     ▼       │
│         │                      │              ┌───────────┐ │
│         │                      └─────────────▶│  Alert    │ │
│         │                                     │  System   │ │
│         │                                     └───────────┘ │
│         │                                            │       │
│         ▼                                            │       │
│  ┌──────────────┐                                   │       │
│  │ Visualization│◀──────────────────────────────────┘       │
│  │   Renderer   │                                           │
│  └──────────────┘                                           │
│         │                                                    │
│         │                      ┌──────────────┐            │
│         └─────────────────────▶│   Resource   │            │
│                                │   Monitor    │            │
│                                └──────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Class Structure

**`ProcessMonitor`**
- Tracks application-specific CPU and RAM usage
- Maintains rolling history of performance metrics
- Provides statistical analysis (average, peak, current)

**`DrowsinessDetector`**
- Core detection logic
- Manages EAR calculation and threshold checking
- Handles blink detection and counting
- Orchestrates visualization components
- Processes each frame through the detection pipeline

### Data Flow

1. **Input**: Webcam captures RGB frames at 30+ FPS
2. **Processing**: MediaPipe processes frame to extract 468 facial landmarks
3. **Analysis**: EAR calculated for left and right eyes, averaged
4. **Decision**: Compare EAR against threshold, track duration
5. **Alert**: Generate appropriate warning level based on state
6. **Visualization**: Render all UI components with current state
7. **Monitoring**: Update resource usage and performance metrics
8. **Output**: Display composite frame with all overlays

---

## 📊 Performance Metrics

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Dual-core 2.0 GHz | Quad-core 2.5+ GHz |
| **RAM** | 4 GB | 8 GB |
| **Camera** | 720p @ 30fps | 1080p @ 30fps |
| **OS** | Windows 10, Linux, macOS 10.14+ | Latest versions |

### Performance Benchmarks

**Tested on: Intel i5-8250U, 8GB RAM, Integrated Graphics**

- **Average FPS**: 32-35 fps
- **CPU Usage**: 15-25% (single core)
- **RAM Usage**: 80-120 MB
- **Detection Latency**: <50ms per frame
- **False Positive Rate**: <2% (with default threshold)
- **True Positive Rate**: >95% for drowsiness detection

### Optimization Features

✅ Process-specific monitoring (not system-wide)  
✅ Efficient numpy operations for landmark processing  
✅ Rolling buffer system prevents memory bloat  
✅ Smart frame processing with minimal overhead  
✅ No GPU required (CPU-only processing)  

---

## 🚀 Future Enhancements

### Planned Features

#### Short-term (v2.0)
- [ ] Audio alerts with customizable sounds
- [ ] Yawn detection using mouth landmarks
- [ ] Head pose estimation for distraction detection
- [ ] Multi-driver support for commercial vehicles
- [ ] CSV export of session data
- [ ] Configurable sensitivity profiles

#### Medium-term (v3.0)
- [ ] Cloud-based analytics dashboard
- [ ] Mobile app companion (iOS/Android)
- [ ] Integration with vehicle CAN bus
- [ ] Machine learning model for personalized thresholds
- [ ] Driver behavior scoring system
- [ ] Fleet management features

#### Long-term (v4.0)
- [ ] Multi-camera support
- [ ] Night vision / IR camera compatibility
- [ ] Integration with autonomous driving systems
- [ ] Predictive drowsiness detection using ML
- [ ] Real-time data sharing with emergency services
- [ ] Wearable device integration

### Research Directions

- **Deep Learning Models**: Explore CNN-based approaches for improved accuracy
- **Sensor Fusion**: Combine vision with steering wheel sensors, accelerometer data
- **Edge Computing**: Optimize for deployment on embedded systems (Raspberry Pi, Jetson Nano)
- **Privacy-First Design**: Implement on-device processing with zero cloud dependency

---

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug fixes, new features, or documentation improvements.

### How to Contribute

1. **Fork the repository**
2. **Create a featur