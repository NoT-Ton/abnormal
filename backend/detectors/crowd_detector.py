"""
Crowd Anomaly Detector
Detects unusual crowd density changes and formation patterns.
"""

import numpy as np
from collections import deque


class CrowdDetector:
    """Detects crowd-level anomalies like sudden density spikes or formations."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.count_history = deque(maxlen=window_size * 5)
        self.last_alert_time: float = -999
        self.baseline_count: float = None

    def update(self, detections: list, timestamp: float, frame) -> list:
        events = []
        count = len(detections)
        self.count_history.append((timestamp, count))

        if len(self.count_history) < 20:
            return events

        counts = [c for _, c in self.count_history]

        # Establish baseline from early portion
        if self.baseline_count is None and len(counts) >= 30:
            self.baseline_count = np.median(counts[:30])

        if self.baseline_count is None or self.baseline_count < 2:
            return events

        # Detect spike (sudden crowd increase)
        recent_avg = np.mean(counts[-5:])
        spike_ratio = recent_avg / self.baseline_count

        if spike_ratio > 2.5 and (timestamp - self.last_alert_time) > 8:
            self.last_alert_time = timestamp
            events.append({
                "type": "crowd_anomaly",
                "subtype": "density_spike",
                "timestamp": round(timestamp, 2),
                "baseline_count": int(self.baseline_count),
                "current_count": int(recent_avg),
                "spike_ratio": round(spike_ratio, 2),
                "confidence": round(min((spike_ratio - 1) / 3, 0.95), 2),
                "description": (
                    f"CROWD SPIKE: {int(recent_avg)} persons detected "
                    f"({spike_ratio:.1f}x normal density of {int(self.baseline_count)}). "
                    "Unusual crowd gathering may require attention."
                ),
            })

        return events