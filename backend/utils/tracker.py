"""
Person Tracker
Uses OpenCV background subtraction + contour detection for person detection,
with centroid-based ID assignment for tracking across frames.
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class PersonTracker:
    """
    Lightweight person tracker using:
    - MOG2 background subtraction for detection
    - Centroid tracking with ID persistence
    """

    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid
        self.bboxes = OrderedDict()        # id -> bbox
        self.disappeared = OrderedDict()   # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect_persons(self, frame):
        """Detect person-like blobs using background subtraction."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (gray pixels = 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        h, w = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area (roughly person-sized blob)
            if area < 600 or area > (h * w * 0.3):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(bw, 1)

            # Person-like aspect ratio (taller than wide)
            if aspect < 0.8 or aspect > 6.0:
                continue

            cx = x + bw // 2
            cy = y + bh // 2
            detections.append({
                "centroid": (cx, cy),
                "bbox": (x, y, x + bw, y + bh),
                "area": area,
            })

        return detections

    def update(self, detections):
        """Update tracker with new detections, return objects with IDs."""
        if not detections:
            for pid in list(self.disappeared):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    del self.objects[pid]
                    del self.bboxes[pid]
                    del self.disappeared[pid]
            return self._to_list()

        input_centroids = np.array([d["centroid"] for d in detections])
        input_bboxes = [d["bbox"] for d in detections]

        if not self.objects:
            for i, c in enumerate(input_centroids):
                self._register(c, input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                pid = object_ids[row]
                self.objects[pid] = input_centroids[col]
                self.bboxes[pid] = input_bboxes[col]
                self.disappeared[pid] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                pid = object_ids[row]
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    del self.objects[pid]
                    del self.bboxes[pid]
                    del self.disappeared[pid]

            for col in unused_cols:
                self._register(input_centroids[col], input_bboxes[col])

        return self._to_list()

    def _register(self, centroid, bbox):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _to_list(self):
        result = []
        for pid, centroid in self.objects.items():
            bbox = self.bboxes.get(pid, (0, 0, 0, 0))
            result.append({
                "id": pid,
                "centroid": centroid,
                "bbox": bbox,
            })
        return result

    def detect_and_track(self, frame):
        """Full pipeline: detect then track."""
        raw_detections = self.detect_persons(frame)
        return self.update(raw_detections)
