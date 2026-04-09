"""
Annotated Video Writer
Handles writing annotated frames to output video file.
"""
 
import cv2
 
 
class AnnotatedVideoWriter:
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.frame_count = 0
 
    def write_frame(self, frame):
        self.writer.write(frame)
        self.frame_count += 1
 
    def release(self):
        self.writer.release()
 