# ⬡ ABDS — Abnormal Behavior Detection System

A computer vision system that analyzes surveillance video to detect abnormal crowd behaviors.

---

## Project Structure

```
abnormal-behavior-detection/
├── frontend/
│   └── app.py                    # Gradio UI — video upload & results display
│
├── backend/
│   ├── analyzer.py               # Main orchestrator — runs all detectors
│   ├── detectors/
│   │   ├── loitering_detector.py # Detects same person lingering / circling
│   │   ├── panic_detector.py     # Detects running + crowd panic + dispersal
│   │   └── crowd_detector.py     # Detects density spikes / anomalies
│   ├── utils/
│   │   ├── tracker.py            # Person detection + centroid tracking
│   │   └── video_writer.py       # Annotated video output
│   └── models/                   # (reserved for YOLO/deep learning models)
│
└── requirements.txt
```

---

## Behaviors Detected

### 1. Loitering (⚠ WARNING)
- **What**: Same person walking around in the same area repeatedly for ~10 minutes
- **How**: Centroid tracking stores position history per person ID. If ≥65% of their positions fall within a configurable radius over the threshold time window → flagged as loitering
- **Also detects**: Repeated return pattern — person leaves zone and re-enters 3+ times

### 2. Panic / Running (🚨 CRITICAL)
- **What**: People running away causing crowd panic
- **How**:
  - Per-person velocity computed between frames
  - If ≥50% of visible persons exceed speed threshold → crowd panic event
  - Checks directional consensus (everyone fleeing same direction = higher confidence)
  - Monitors crowd density: sudden 60%+ drop in person count → crowd dispersal

### 3. Crowd Anomaly (⚠ WARNING)
- **What**: Unusual density spikes (crowd forming unexpectedly)
- **How**: Baseline crowd count established from first 30 frames; alerts when recent count exceeds 2.5× baseline

---

## Detection Pipeline

```
Video Frame
    │
    ▼
PersonTracker (MOG2 background subtraction + centroid tracking)
    │ detections: [{id, centroid, bbox}, ...]
    ▼
┌───────────────┬──────────────────┬────────────────┐
│LoiteringDetector│  PanicDetector  │ CrowdDetector  │
│ dwell time +  │ velocity +       │ density delta  │
│ zone history  │ direction cons.  │ baseline ratio │
└───────┬───────┴────────┬─────────┴──────┬─────────┘
        │                │                │
        └────────────────▼────────────────┘
                    Events List
                         │
                    Frame Annotation
                         │
                    Output Video + Report
```

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python frontend/app.py
# Opens at http://localhost:7860
```

---

## Configuration

In the Gradio UI, you can adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Loitering Threshold | 60s (demo) / 600s (production) | Time in same zone before flagging |
| Running Speed Threshold | 8.0 px/frame | Velocity to classify as running |

In `backend/analyzer.py` `_default_config()`:
```python
"loitering_area_radius": 150,     # pixel radius defining "same zone"
"panic_spread_threshold": 0.5,    # 50% of crowd running = panic
"frame_skip": 2,                  # process every 2nd frame for speed
```

---

## Extending with Deep Learning

The `backend/models/` directory is reserved for YOLO-based person detection.
Replace the `PersonTracker.detect_persons()` method with a YOLOv8 detector for
significantly better accuracy in complex scenes:

```python
# Install: pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model(frame, classes=[0])  # class 0 = person
```

---

## Output

- **Annotated video**: Bounding boxes (green=normal, orange=loitering, red=panic), timestamps, and live alert overlays
- **Structured report**: Event-by-event breakdown with timestamps, person IDs, confidence scores, and human-readable descriptions
- **Severity level**: NORMAL / WARNING / CRITICAL
