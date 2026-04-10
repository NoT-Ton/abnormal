"""
Panic / Running Detector
Detects when people run away from an area, causing crowd panic.

Algorithm:
1. Compute per-person velocity (speed + direction) between frames
2. Detect sudden high-speed movement (running threshold)
3. Detect crowd-level panic: majority of visible persons moving fast
4. Detect directional consensus: everyone fleeing same direction
5. Detect sudden density drop: crowd dispersal event
"""

import numpy as np
from collections import defaultdict
import math


class PanicDetector:
    """
    Detects running and crowd panic behaviors.
    """

    def __init__(self, speed_threshold: float = 8.0, spread_threshold: float = 0.5):
        self.speed_threshold = speed_threshold      # px/frame for "running"
        self.spread_threshold = spread_threshold    # ratio of runners for "panic"

        # person_id -> deque of recent positions
        self.prev_positions: dict[int, tuple] = {}
        self.velocities: dict[int, list] = defaultdict(list)

        # For crowd-level analysis
        self.crowd_history: list[dict] = []
        self.panic_start_time: float | None = None
        self.last_panic_alert: float = -999

    def update(self, detections: list, timestamp: float, frame) -> list:
        events = []

        current_positions = {d["id"]: d["centroid"] for d in detections}
        speeds = {}
        directions = {}

        # Compute velocities
        for pid, pos in current_positions.items():
            if pid in self.prev_positions:
                prev = self.prev_positions[pid]
                dx = pos[0] - prev[0]
                dy = pos[1] - prev[1]
                speed = math.sqrt(dx * dx + dy * dy)
                speeds[pid] = speed
                directions[pid] = math.atan2(dy, dx)

                self.velocities[pid].append(speed)
                if len(self.velocities[pid]) > 30:
                    self.velocities[pid].pop(0)

        self.prev_positions = current_positions

        # Individual running events
        for pid, speed in speeds.items():
            avg_speed = np.mean(self.velocities[pid]) if self.velocities[pid] else speed
            if avg_speed > self.speed_threshold:
                pass  # Will handle at crowd level below

        # Crowd panic analysis
        if len(speeds) >= 2:
            runners = [pid for pid, s in speeds.items() if s > self.speed_threshold]
            run_ratio = len(runners) / len(speeds)

            # Check directional consensus (all fleeing same way)
            directional_panic = False
            if len(runners) >= 2:
                dirs = [directions[pid] for pid in runners if pid in directions]
                if dirs:
                    dir_variance = self._circular_variance(dirs)
                    directional_panic = dir_variance < 0.8  # Low variance = same direction

            if run_ratio >= self.spread_threshold:
                if timestamp - self.last_panic_alert > 1:   # 3s cooldown — capture every distinct event
                    self.last_panic_alert = timestamp
                    confidence = min(run_ratio + (0.2 if directional_panic else 0), 1.0)
                    events.append({
                        "type": "panic",
                        "timestamp": round(timestamp, 2),
                        "runners_count": len(runners),
                        "total_persons": len(speeds),
                        "run_ratio": round(run_ratio, 2),
                        "directional_consensus": directional_panic,
                        "confidence": round(confidence, 2),
                        "description": (
                            f"PANIC DETECTED: {len(runners)}/{len(speeds)} persons "
                            f"({run_ratio*100:.0f}%) are running. "
                            + ("Crowd fleeing in same direction — likely panic trigger nearby." 
                               if directional_panic 
                               else "Dispersed running — possible crowd panic.")
                        ),
                    })

        # Crowd density drop (sudden dispersal)
        self.crowd_history.append({
            "timestamp": timestamp,
            "count": len(detections),
        })
        if len(self.crowd_history) > 150:
            self.crowd_history.pop(0)

        dispersal_event = self._check_sudden_dispersal(timestamp)
        if dispersal_event:
            events.append(dispersal_event)

        return events

    def _check_sudden_dispersal(self, timestamp: float) -> dict | None:
        if len(self.crowd_history) < 30:
            return None

        recent = self.crowd_history[-5:]
        past = self.crowd_history[-30:-20]

        avg_recent = np.mean([h["count"] for h in recent])
        avg_past = np.mean([h["count"] for h in past])

        if avg_past < 3:
            return None

        drop_ratio = (avg_past - avg_recent) / avg_past

        if drop_ratio > 0.6:  # 60% crowd drop
            if timestamp - self.last_panic_alert > 1:
                self.last_panic_alert = timestamp
                return {
                    "type": "panic",
                    "subtype": "crowd_dispersal",
                    "timestamp": round(timestamp, 2),
                    "crowd_before": int(avg_past),
                    "crowd_after": int(avg_recent),
                    "drop_ratio": round(drop_ratio, 2),
                    "confidence": round(min(drop_ratio, 0.95), 2),
                    "description": (
                        f"CROWD DISPERSAL: {drop_ratio*100:.0f}% sudden drop in crowd density "
                        f"({int(avg_past)} → {int(avg_recent)} persons). "
                        "Possible panic or emergency event causing crowd to flee."
                    ),
                }
        return None

    def _circular_variance(self, angles: list) -> float:
        """Compute circular variance of angles (0=all same, 1=uniform spread)."""
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        r = math.sqrt(sin_mean**2 + cos_mean**2)
        return 1 - r  # 0 = all same direction, 1 = random