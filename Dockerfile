FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────
# ffmpeg      : H.264 re-encoding for browser-compatible video
# libgl1      : OpenCV GUI (needed even in headless mode)
# libglib2.0  : OpenCV dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download YOLOv8n weights at build time ────────────────
# Avoids a slow first-run download inside the container
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Copy application source ───────────────────────────────────
COPY backend/  ./backend/
COPY frontend/ ./frontend/

# ── Runtime environment ───────────────────────────────────────
# ABDS_TMP: writable folder for input/output video files
# Matches the path used in frontend/app.py
ENV ABDS_TMP=/tmp/abds_tmp
RUN mkdir -p /tmp/abds_tmp

# Gradio listens on all interfaces inside the container
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ── Expose port ───────────────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

# ── Entrypoint ────────────────────────────────────────────────
CMD ["python", "frontend/app.py"]