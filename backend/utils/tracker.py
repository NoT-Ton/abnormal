"""
Person Tracker — YOLOv8 Edition
Replaces MOG2 blob detection with YOLOv8 person-class detection.
Fixes: body-part fragmentation, glass reflections, false positives.
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class PersonTracker:
    """
    Person tracker using:
    - YOLOv8n for detection  (class 0 = person only, confidence >= 0.45)
    - Centroid tracking with IoU-assisted ID assignment
    - NMS to suppress overlapping boxes on same person
    """

    def __init__(self, max_disappeared=8, max_distance=120, conf_threshold=0.45):
        self.next_id = 0
        self.objects = OrderedDict()      # id -> centroid (x, y)
        self.bboxes = OrderedDict()       # id -> (x1,y1,x2,y2)
        self.disappeared = OrderedDict()  # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.conf_threshold = conf_threshold

        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            # yolov8n.pt is ~6MB, downloads once automatically
            self.model = YOLO("yolov8m.pt")
            self.model.to("cpu")  # force CPU inference
            print("[Tracker] YOLOv8n loaded OK (CPU mode)")
        except Exception as e:
            print(f"[Tracker] YOLOv8 unavailable: {e} -- falling back to HOG detector")
            self.model = None
            self._init_hog_fallback()

    def _init_hog_fallback(self):
        """HOG person detector as fallback (much better than MOG2)."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_persons(self, frame):
        """Detect persons using YOLO (preferred) or HOG (fallback)."""
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame):
        """Run YOLOv8, keep only class=0 (person) above confidence threshold."""
        results = self.model(
            frame,
            classes=[0],           # person only -- ignores everything else
            conf=self.conf_threshold,
            iou=0.45,              # NMS threshold -- merges overlapping boxes
            device="cpu",          # explicit CPU — no GPU required
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                # Sanity: minimum person size (avoids tiny distant specks)
                w = x2 - x1
                h = y2 - y1
                if w < 15 or h < 20:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append({
                    "centroid": (cx, cy),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })

        return detections

    def _detect_hog(self, frame):
        """HOG fallback -- better than MOG2 for static cameras."""
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        rects, weights = self.hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )

        detections = []
        for i, (rx, ry, rw, rh) in enumerate(rects):
            x1 = int(rx / scale)
            y1 = int(ry / scale)
            x2 = int((rx + rw) / scale)
            y2 = int((ry + rh) / scale)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = float(weights[i]) if i < len(weights) else 0.5
            detections.append({
                "centroid": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            })

        detections = self._apply_nms(detections, iou_threshold=0.4)
        return detections

    def _apply_nms(self, detections, iou_threshold=0.4):
        """Non-maximum suppression to merge overlapping detections."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections], dtype=float)
        scores = np.array([d["confidence"] for d in detections])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_threshold]

        return [detections[i] for i in keep]

    # ── Centroid Tracking ────────────────────────────────────────────────

    def update(self, detections):
        """Assign persistent IDs to detections using centroid matching."""
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
                self._register(tuple(c), input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            D = dist.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                pid = object_ids[row]
                self.objects[pid] = tuple(input_centroids[col])
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
                self._register(tuple(input_centroids[col]), input_bboxes[col])

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
        """Full pipeline: YOLO detect -> centroid track."""
        raw_detections = self.detect_persons(frame)
        return self.update(raw_detections)