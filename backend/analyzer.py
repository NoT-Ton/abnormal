"""
Abnormal Behavior Detection - Main Analyzer
Orchestrates all detection modules and returns structured results.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Generator

from detectors.loitering_detector import LoiteringDetector
from detectors.panic_detector import PanicDetector
from detectors.crowd_detector import CrowdDetector
from utils.tracker import PersonTracker
from utils.video_writer import AnnotatedVideoWriter


class AbnormalBehaviorAnalyzer:
    """
    Main analyzer that processes video and detects:
    1. Loitering: same person walking around repeatedly ~10 minutes
    2. Panic/Running: people running away causing crowd panic
    3. Crowd anomaly: sudden crowd dispersal or density spike
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.tracker = PersonTracker(
            conf_threshold=self.config.get("yolo_conf_threshold", 0.45)
        )
        self.loitering_detector = LoiteringDetector(
            threshold_seconds=self.config["loitering_threshold_seconds"],
            area_radius=self.config["loitering_area_radius"],
        )
        self.panic_detector = PanicDetector(
            speed_threshold=self.config["panic_speed_threshold"],
            spread_threshold=self.config["panic_spread_threshold"],
        )
        self.crowd_detector = CrowdDetector()
        self.person_trails: dict = {}
        self.TRAIL_LEN = 40

    def _default_config(self) -> dict:
        return {
            "loitering_threshold_seconds": 60,  # Demo: 60s (production: 600s)
            "loitering_area_radius": 150,        # pixels radius for "same area"
            "panic_speed_threshold": 8.0,        # pixels/frame to consider running
            "panic_spread_threshold": 0.6,       # 60% of people running = panic
            "frame_skip": 2,                     # process every Nth frame
            "max_persons": 50,
        }

    def analyze_video(
        self, video_path: str, output_path: str, progress_callback=None
    ) -> dict:
        """
        Analyze video for abnormal behaviors.
        Returns structured report with all detected events.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps

        writer = AnnotatedVideoWriter(output_path, fps, width, height)

        events = []
        frame_idx = 0
        processed = 0
        skip = self.config["frame_skip"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip != 0:
                writer.write_frame(frame)
                continue

            timestamp = frame_idx / fps

            # Track persons in frame
            detections = self.tracker.detect_and_track(frame)

            # Run behavior detectors
            loitering_events = self.loitering_detector.update(
                detections, timestamp, frame
            )
            panic_events = self.panic_detector.update(
                detections, timestamp, frame
            )
            crowd_events = self.crowd_detector.update(
                detections, timestamp, frame
            )

            all_frame_events = loitering_events + panic_events + crowd_events
            events.extend(all_frame_events)

            # Annotate frame
            annotated = self._annotate_frame(
                frame, detections, all_frame_events, timestamp
            )
            writer.write_frame(annotated)

            processed += 1
            if progress_callback and frame_idx % 30 == 0:
                pct = min(frame_idx / max(total_frames, 1), 0.99)
                progress_callback(pct, f"Analyzing frame {frame_idx}/{total_frames}...")

        cap.release()
        writer.release()

        report = self._build_report(events, duration_sec, fps, total_frames)
        if progress_callback:
            progress_callback(1.0, "Analysis complete!")
        return report


    def analyze_video_streaming(
        self, video_path: str, output_path: str
    ):
        """
        Generator version of analyze_video.
        Yields (annotated_frame_rgb, progress_pct, status_msg) during processing,
        then finally yields (None, 1.0, report_dict) as the last item.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps

        writer = AnnotatedVideoWriter(output_path, fps, width, height)

        events = []
        frame_idx = 0
        skip = self.config["frame_skip"]
        preview_every = max(skip * 5, 10)  # yield preview every N frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp = frame_idx / fps

            if frame_idx % skip != 0:
                writer.write_frame(frame)
                continue

            detections = self.tracker.detect_and_track(frame)

            loitering_events = self.loitering_detector.update(detections, timestamp, frame)
            panic_events = self.panic_detector.update(detections, timestamp, frame)
            crowd_events = self.crowd_detector.update(detections, timestamp, frame)

            all_frame_events = loitering_events + panic_events + crowd_events
            events.extend(all_frame_events)

            annotated = self._annotate_frame(frame, detections, all_frame_events, timestamp)
            writer.write_frame(annotated)

            # Yield live preview frame
            if frame_idx % preview_every == 0:
                pct = min(frame_idx / max(total_frames, 1), 0.99)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield rgb, pct, f"Processing frame {frame_idx}/{total_frames}..."

        cap.release()
        writer.release()

        # Re-encode to H.264 for browser playback (replaces mp4v if needed)
        browser_path = writer.get_browser_compatible_path()

        report = self._build_report(events, duration_sec, fps, total_frames)
        report["_output_video_path"] = browser_path
        yield None, 1.0, report

    def _annotate_frame(self, frame, detections, events, timestamp):
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Decide event state per person
        alert_pids_loiter = {ev.get("person_id") for ev in events if ev["type"] == "loitering"}
        alert_pids_panic  = {ev.get("person_id") for ev in events if ev["type"] == "panic"}
        has_panic = bool([e for e in events if e["type"] == "panic"])

        # ── Draw motion trails ──
        for pid, trail in self.person_trails.items():
            if len(trail) < 2:
                continue
            pts = list(trail)
            # Color trail by state
            if pid in alert_pids_panic or has_panic:
                trail_color = (0, 0, 200)
            elif pid in alert_pids_loiter:
                trail_color = (0, 120, 220)
            else:
                trail_color = (60, 160, 60)

            for i in range(1, len(pts)):
                alpha = i / len(pts)            # fade older points
                thickness = max(1, int(alpha * 3))
                faded = tuple(int(c * alpha) for c in trail_color)
                cv2.line(annotated, pts[i-1], pts[i], faded, thickness, cv2.LINE_AA)

        # ── Draw bounding boxes ──
        for det in detections:
            pid = det.get("id", -1)
            box = det.get("bbox", [])
            if len(box) != 4:
                continue
            x, y, x2, y2 = [int(v) for v in box]

            if pid in alert_pids_panic or has_panic:
                color = (0, 0, 255)      # Red — panic
                label = f"#{pid} RUN"
            elif pid in alert_pids_loiter:
                color = (0, 165, 255)    # Orange — loitering
                label = f"#{pid} LOITER"
            else:
                color = (0, 220, 80)     # Green — normal
                label = f"#{pid}"

            # Filled top bar for label
            cv2.rectangle(annotated, (x, y - 18), (x + len(label)*9, y), color, -1)
            cv2.rectangle(annotated, (x, y), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(annotated, label, (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

            # Update trail
            cx, cy = det.get("centroid", ((x+x2)//2, (y+y2)//2))
            trail = self.person_trails.setdefault(pid, [])
            trail.append((int(cx), int(cy)))
            if len(trail) > self.TRAIL_LEN:
                trail.pop(0)

        # ── HUD: top bar ──
        ts_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
        cv2.rectangle(annotated, (0, 0), (w, 38), (0, 0, 0), -1)
        cv2.putText(annotated, f"ABDS  |  {ts_str}  |  PERSONS: {len(detections)}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 210, 255), 2, cv2.LINE_AA)

        # ── Alert banners ──
        y_offset = 55
        for ev in events:
            if ev["type"] == "loitering":
                banner = f"LOITERING  Person #{ev.get('person_id')}  {ev.get('duration_seconds',0):.0f}s"
                bg = (0, 100, 180)
            elif ev["type"] == "panic":
                banner = f"PANIC / RUNNING  {ev.get('runners_count','?')}/{ev.get('total_persons','?')} persons"
                bg = (0, 0, 180)
            else:
                banner = f"CROWD ANOMALY"
                bg = (0, 100, 180)

            text_w = len(banner) * 11
            cv2.rectangle(annotated, (8, y_offset - 18), (8 + text_w, y_offset + 6), bg, -1)
            cv2.putText(annotated, banner, (12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 30

        return annotated

    def _build_report(self, events, duration_sec, fps, total_frames) -> dict:
        loitering = [e for e in events if e["type"] == "loitering"]
        panic = [e for e in events if e["type"] == "panic"]
        crowd = [e for e in events if e["type"] == "crowd_anomaly"]

        # Deduplicate by person/time window
        unique_loitering = self._dedup_events(loitering, key="person_id", window_sec=30)
        unique_panic = self._dedup_events(panic, key="type", window_sec=10)
        unique_crowd = self._dedup_events(crowd, key="type", window_sec=15)

        severity = "NORMAL"
        if unique_panic:
            severity = "CRITICAL"
        elif unique_loitering:
            severity = "WARNING"

        return {
            "severity": severity,
            "duration_seconds": round(duration_sec, 1),
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "summary": {
                "loitering_incidents": len(unique_loitering),
                "panic_incidents": len(unique_panic),
                "crowd_anomalies": len(unique_crowd),
            },
            "events": {
                "loitering": unique_loitering,
                "panic": unique_panic,
                "crowd_anomaly": unique_crowd,
            },
        }

    def _dedup_events(self, events, key, window_sec=10):
        seen = {}
        result = []
        for ev in events:
            k = ev.get(key, "unknown")
            t = ev.get("timestamp", 0)
            if k not in seen or (t - seen[k]) > window_sec:
                seen[k] = t
                result.append(ev)
        return result