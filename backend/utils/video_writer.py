"""
Annotated Video Writer
Writes frames with a browser-compatible H.264 codec.
Falls back to ffmpeg re-encode if OpenCV can't write H.264 directly.
"""

import cv2
import os
import subprocess
import tempfile


class AnnotatedVideoWriter:
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_count = 0

        # Try H.264 first (browser-compatible), fall back to mp4v then re-encode
        self.writer, self.raw_path = self._open_writer()

    def _open_writer(self):
        """
        Try codecs in order of browser compatibility:
          1. avc1  — H.264, natively plays in Chrome/Firefox/Edge
          2. H264  — alias on some OpenCV builds
          3. mp4v  — fallback; needs ffmpeg re-encode after writing
        """
        codecs = ["avc1", "H264", "mp4v"]
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
            if writer.isOpened():
                print(f"[VideoWriter] Using codec: {codec}")
                return writer, self.output_path

        raise RuntimeError("Could not open VideoWriter with any codec.")

    def write_frame(self, frame):
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        self.writer.release()

    def get_browser_compatible_path(self) -> str:
        """
        Return a path to a browser-playable H.264 MP4.
        If the writer used mp4v, re-encode with ffmpeg.
        Otherwise return the original path.
        """
        # Check if file is valid
        if not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0:
            return self.output_path

        # Try to detect if we need to re-encode (mp4v files need it)
        # We always attempt ffmpeg re-encode for guaranteed browser compat
        reencoded = self.output_path.replace(".mp4", "_h264.mp4")
        success = _ffmpeg_reencode(self.output_path, reencoded)
        if success:
            return reencoded
        # ffmpeg not available — return original (download works, playback may not)
        return self.output_path


def _ffmpeg_reencode(src: str, dst: str) -> bool:
    """Re-encode src to dst using H.264/AAC via ffmpeg. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",   # required for browser compat
                "-an",                    # no audio
                "-movflags", "+faststart", # allow streaming/seeking
                dst,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return result.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False