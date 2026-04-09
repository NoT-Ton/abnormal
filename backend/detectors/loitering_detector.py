"""
Loitering Detector
Detects when the same person stays in or repeatedly returns to the same area
for an extended period (default: ~10 minutes, configurable).
"""

import numpy as np
from collections import defaultdict
import math


class LoiteringDetector:
    """
    Algorithm:
    1. Track each person's position history over time
    2. Compute their "home zone" as weighted average of positions
    3. If they stay within radius of home zone for threshold_seconds → loitering
    4. Also detects repeated return: person leaves but comes back multiple times
    """

    def __init__(self, threshold_seconds: float = 600, area_radius: float = 150):
        self.threshold_seconds = threshold_seconds
        self.area_radius = area_radius

        # person_id -> list of (timestamp, centroid)
        self.tracks: dict[int, list] = defaultdict(list)
        # person_id -> first_seen timestamp
        self.first_seen: dict[int, float] = {}
        # person_id -> set of zone IDs visited
        self.zone_visits: dict[int, dict] = defaultdict(dict)

    def update(self, detections: list, timestamp: float, frame) -> list:
        events = []
        active_ids = set()

        for det in detections:
            pid = det["id"]
            cx, cy = det["centroid"]
            active_ids.add(pid)

            if pid not in self.first_seen:
                self.first_seen[pid] = timestamp

            self.tracks[pid].append((timestamp, cx, cy))

            # Trim old history beyond 2x threshold
            cutoff = timestamp - self.threshold_seconds * 2
            self.tracks[pid] = [
                t for t in self.tracks[pid] if t[0] >= cutoff
            ]

            # Check loitering
            event = self._check_loitering(pid, timestamp)
            if event:
                events.append(event)

        # Clean up persons not seen recently
        for pid in list(self.tracks.keys()):
            if pid not in active_ids:
                recent = self.tracks[pid]
                if recent and (timestamp - recent[-1][0]) > self.threshold_seconds:
                    del self.tracks[pid]
                    self.first_seen.pop(pid, None)

        return events

    def _check_loitering(self, pid: int, timestamp: float) -> dict | None:
        history = self.tracks[pid]
        if len(history) < 5:
            return None

        # Time in tracking
        first_ts = history[0][0]
        duration = timestamp - first_ts

        if duration < self.threshold_seconds:
            return None

        # Compute positions in last threshold window
        window_start = timestamp - self.threshold_seconds
        window = [(t, x, y) for t, x, y in history if t >= window_start]

        if len(window) < 3:
            return None

        positions = np.array([[x, y] for _, x, y in window])
        center = positions.mean(axis=0)

        # Check if most positions are within radius of center
        dists = np.linalg.norm(positions - center, axis=1)
        within_radius = (dists <= self.area_radius).sum()
        ratio = within_radius / len(positions)

        if ratio >= 0.65:  # 65% of time in same zone
            return {
                "type": "loitering",
                "person_id": pid,
                "timestamp": round(timestamp, 2),
                "duration_seconds": round(duration, 1),
                "location": (int(center[0]), int(center[1])),
                "confidence": round(ratio, 2),
                "description": (
                    f"Person #{pid} has been loitering in the same area "
                    f"for {self._format_duration(duration)}. "
                    f"Stayed within {self.area_radius}px radius {ratio*100:.0f}% of the time."
                ),
            }

        # Check repeated return pattern (leaves & comes back)
        return self._check_repeated_return(pid, timestamp, history)

    def _check_repeated_return(self, pid: int, timestamp: float, history: list) -> dict | None:
        if len(history) < 10:
            return None

        positions = np.array([[x, y] for _, x, y in history])
        center = positions.mean(axis=0)
        dists = np.linalg.norm(positions - center, axis=1)

        # Count zone entry/exits
        in_zone = dists <= self.area_radius
        transitions = np.diff(in_zone.astype(int))
        entries = (transitions == 1).sum()

        if entries >= 3:  # Returned 3+ times
            duration = history[-1][0] - history[0][0]
            return {
                "type": "loitering",
                "person_id": pid,
                "timestamp": round(timestamp, 2),
                "duration_seconds": round(duration, 1),
                "location": (int(center[0]), int(center[1])),
                "confidence": min(0.5 + entries * 0.1, 0.95),
                "description": (
                    f"Person #{pid} repeatedly returned to the same area "
                    f"{entries} times over {self._format_duration(duration)}. "
                    "Suspicious circular movement pattern detected."
                ),
            }

        return None

    def _format_duration(self, seconds: float) -> str:
        if seconds >= 60:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        return f"{int(seconds)}s"
