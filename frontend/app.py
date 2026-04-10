"""
Abnormal Behavior Detection - Gradio Frontend
Sleek surveillance-style UI with video upload and real-time analysis results.
"""

import gradio as gr
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analyzer import AbnormalBehaviorAnalyzer

# ──────────────────────────────────────────────
# CSS — Dark surveillance aesthetic
# ──────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg-dark:    #080c10;
    --bg-panel:   #0d1117;
    --bg-card:    #111820;
    --border:     #1e3a4a;
    --accent:     #00d4ff;
    --accent2:    #ff4444;
    --warning:    #ff8c00;
    --success:    #00ff88;
    --text:       #c8e0ee;
    --muted:      #4a6070;
    --mono:       'Share Tech Mono', monospace;
    --display:    'Rajdhani', sans-serif;
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: var(--display) !important;
    color: var(--text) !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    background: linear-gradient(180deg, rgba(0,212,255,0.04) 0%, transparent 100%);
}
.app-header h1 {
    font-family: var(--display);
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
    margin: 0;
    text-transform: uppercase;
}
.app-header p {
    font-family: var(--mono);
    color: var(--muted);
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    margin-top: 0.4rem;
    text-transform: uppercase;
}

/* Panels */
.panel-box {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed var(--border) !important;
    border-radius: 4px !important;
    background: var(--bg-card) !important;
    transition: border-color 0.2s;
    min-height: 200px;
}
.upload-zone:hover { border-color: var(--accent) !important; }

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #004455, #006677) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.7rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.15) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #006677, #0099aa) !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Section labels */
.section-label {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.5rem !important;
}

/* Severity badge */
.severity-critical { color: var(--accent2) !important; text-shadow: 0 0 10px rgba(255,68,68,0.5); }
.severity-warning  { color: var(--warning) !important; text-shadow: 0 0 10px rgba(255,140,0,0.5); }
.severity-normal   { color: var(--success) !important; text-shadow: 0 0 10px rgba(0,255,136,0.4); }

/* Report output */
#report-output textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    border-radius: 2px !important;
}

/* Progress */
.progress-bar-wrap { border-radius: 2px !important; }
.progress-bar { background: var(--accent) !important; }

/* Video components */
video { border: 1px solid var(--border) !important; border-radius: 2px !important; }

/* Scan line overlay effect */
# .scan-overlay {
#     position: fixed;
#     top: 0; left: 0; right: 0; bottom: 0;
#     background: repeating-linear-gradient(
#         0deg,
#         transparent,
#         transparent 2px,
#         rgba(0,0,0,0.03) 2px,
#         rgba(0,0,0,0.03) 4px
#     );
#     pointer-events: none;
#     z-index: 9999;
# }

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: pulse 2s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Metrics row */
.metric-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: 1rem !important;
    text-align: center !important;
}

label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; }
"""


# ──────────────────────────────────────────────
# Analysis function
# ──────────────────────────────────────────────

def format_report(report: dict) -> str:
    """Convert report dict to readable text."""
    lines = []
    sev = report.get("severity", "UNKNOWN")
    
    lines.append("=" * 56)
    lines.append(f"  BEHAVIORAL ANALYSIS REPORT")
    lines.append("=" * 56)
    lines.append(f"  THREAT LEVEL  : {sev}")
    lines.append(f"  VIDEO DURATION: {report.get('duration_seconds', 0):.1f}s")
    lines.append(f"  TOTAL FRAMES  : {report.get('total_frames', 0)} @ {report.get('fps', 0):.1f}fps")
    lines.append("")

    summary = report.get("summary", {})
    lines.append("── SUMMARY ──────────────────────────────────────────")
    lines.append(f"  Loitering Incidents  : {summary.get('loitering_incidents', 0)}")
    lines.append(f"  Panic/Run Incidents  : {summary.get('panic_incidents', 0)}")
    lines.append(f"  Crowd Anomalies      : {summary.get('crowd_anomalies', 0)}")
    lines.append("")

    events = report.get("events", {})

    loitering = events.get("loitering", [])
    if loitering:
        lines.append("── ⚠  LOITERING EVENTS ─────────────────────────────")
        for i, ev in enumerate(loitering, 1):
            ts = ev.get("timestamp", 0)
            mm = int(ts // 60); ss = int(ts % 60)
            lines.append(f"  [{i}] @ {mm:02d}:{ss:02d} | Person #{ev.get('person_id', '?')}")
            lines.append(f"      Duration : {ev.get('duration_seconds', 0):.0f}s")
            lines.append(f"      Confidence: {ev.get('confidence', 0)*100:.0f}%")
            lines.append(f"      {ev.get('description', '')}")
            lines.append("")

    panic = events.get("panic", [])
    if panic:
        lines.append("── 🚨 PANIC / RUNNING EVENTS ───────────────────────")
        for i, ev in enumerate(panic, 1):
            ts = ev.get("timestamp", 0)
            mm = int(ts // 60); ss = int(ts % 60)
            lines.append(f"  [{i}] @ {mm:02d}:{ss:02d}")
            lines.append(f"      Confidence : {ev.get('confidence', 0)*100:.0f}%")
            lines.append(f"      {ev.get('description', '')}")
            lines.append("")

    crowd = events.get("crowd_anomaly", [])
    if crowd:
        lines.append("── ⚠  CROWD ANOMALIES ──────────────────────────────")
        for i, ev in enumerate(crowd, 1):
            ts = ev.get("timestamp", 0)
            mm = int(ts // 60); ss = int(ts % 60)
            lines.append(f"  [{i}] @ {mm:02d}:{ss:02d}")
            lines.append(f"      {ev.get('description', '')}")
            lines.append("")

    if not loitering and not panic and not crowd:
        lines.append("── ✓  NO ABNORMAL BEHAVIORS DETECTED ───────────────")
        lines.append("  Video appears normal. No suspicious activity found.")
        lines.append("")

    lines.append("=" * 56)
    lines.append("  END OF REPORT")
    lines.append("=" * 56)
    return "\n".join(lines)


def run_analysis(video_file, loitering_threshold, speed_threshold, conf_threshold):
    """
    Streaming generator.
    Outputs order matches .click() outputs list:
      live_frame (Image), video_output (Video),
      severity, loitering, panic, crowd, report
    """
    if video_file is None:
        raise gr.Error("Please upload a video file first.")

    config = {
        "loitering_threshold_seconds": int(loitering_threshold),
        "loitering_area_radius": 150,
        "panic_speed_threshold": float(speed_threshold),
        "panic_spread_threshold": 0.5,
        "frame_skip": 2,
        "max_persons": 50,
        "yolo_conf_threshold": float(conf_threshold),
    }

    analyzer = AbnormalBehaviorAnalyzer(config=config)

    # Use a dedicated output dir to avoid Windows file-lock issues on Gradio's temp dir.
    # tempfile.mktemp can conflict on Windows — use a named temp file in our own folder.
    out_dir = os.path.join(tempfile.gettempdir(), "abds_output")
    os.makedirs(out_dir, exist_ok=True)
    output_video = os.path.join(out_dir, f"annotated_{os.getpid()}.mp4")

    # Copy input away from Gradio's locked temp path (Windows PermissionError fix)
    input_copy = os.path.join(out_dir, f"input_{os.getpid()}.mp4")
    shutil.copy2(video_file, input_copy)

    for result in analyzer.analyze_video_streaming(input_copy, output_video):
        frame_rgb, pct, payload = result

        if frame_rgb is not None:
            # ── Still processing: show live frame ──
            yield (
                gr.update(value=frame_rgb, visible=True),    # live_frame: show
                gr.update(value=None),                        # video_output: clear while processing
                "",                                           # severity
                "",                                           # loitering
                "",                                           # panic
                "",                                           # crowd
                f"⏳ Analyzing... {int(pct*100)}%",           # report placeholder
            )
        else:
            # ── Done: hide live frame, load final video ──
            report = payload
            report_text = format_report(report)
            severity = report.get("severity", "NORMAL")
            summary = report.get("summary", {})

            severity_display = (
                f"🔴 {severity}" if severity == "CRITICAL" else
                f"🟠 {severity}" if severity == "WARNING" else
                f"🟢 {severity}"
            )

            # Copy the output video into a stable location Gradio can serve
            final_video = None
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                import uuid
                serve_dir = os.path.join(tempfile.gettempdir(), "abds_serve")
                os.makedirs(serve_dir, exist_ok=True)
                final_video = os.path.join(serve_dir, f"result_{uuid.uuid4().hex}.mp4")
                shutil.copy2(output_video, final_video)

            yield (
                gr.update(value=None, visible=False),         # live_frame: hide
                gr.update(value=final_video, visible=True),   # video_output: show with stable path
                severity_display,
                str(summary.get("loitering_incidents", 0)),
                str(summary.get("panic_incidents", 0)),
                str(summary.get("crowd_anomalies", 0)),
                report_text,
            )


# ──────────────────────────────────────────────
# Build UI
# ──────────────────────────────────────────────

with gr.Blocks(title="ABDS — Abnormal Behavior Detection") as demo:

    gr.HTML("""
    <div class="scan-overlay"></div>
    <div class="app-header">
        <h1>⬡ ABDS</h1>
        <p>Abnormal Behavior Detection System · Surveillance AI · v2.0</p>
    </div>
    """)

    with gr.Row():
        # ── LEFT COLUMN: Upload + Config ──
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">▸ Input · Video Feed</div>')
            video_input = gr.Video(
                label="Upload Surveillance Video",
                elem_classes=["upload-zone"],
            )

            gr.HTML('<div class="section-label" style="margin-top:1.2rem">▸ Detection Parameters</div>')

            with gr.Row():
                loitering_slider = gr.Slider(
                    minimum=30,
                    maximum=600,
                    value=60,
                    step=30,
                    label="Loitering Threshold (seconds)",
                    info="Time in same zone to flag as loitering",
                )

            with gr.Row():
                speed_slider = gr.Slider(
                    minimum=3.0,
                    maximum=20.0,
                    value=8.0,
                    step=0.5,
                    label="Running Speed Threshold (px/frame)",
                    info="Movement speed to classify as running",
                )

            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.25,
                    maximum=0.85,
                    value=0.45,
                    step=0.05,
                    label="YOLO Detection Confidence",
                    info="Higher = fewer but more certain detections (raise if too many false boxes)",
                )

            analyze_btn = gr.Button(
                "◉  INITIATE ANALYSIS",
                variant="primary",
                size="lg",
            )

            gr.HTML("""
            <div style="margin-top:1rem; padding:0.8rem; background:#0d1117; border:1px solid #1e3a4a; border-radius:2px; font-family:monospace; font-size:0.7rem; color:#4a6070; line-height:1.8;">
                <span class="status-dot"></span>SYSTEM ONLINE<br>
                ▸ Loitering: centroid tracking + zone dwell<br>
                ▸ Panic: velocity + directional consensus<br>
                ▸ Crowd: density delta anomaly detection
            </div>
            """)

        # ── RIGHT COLUMN: Results ──
        with gr.Column(scale=2):

            gr.HTML('''
            <div class="section-label">▸ Annotated Output Feed</div>
            ''')

            # Live tracking preview (shown while processing)
            live_frame = gr.Image(
                label="",
                show_label=True,
                interactive=False,
                height=360,
                elem_id="live-feed",
                visible=True,
            )

            # Final annotated video — hidden until processing done
            video_output = gr.Video(
                label="▶ Annotated Output — downloadable",
                show_label=True,
                interactive=False,
                height=360,
                visible=False,
                value=None,
                include_audio=False,
            )

            gr.HTML('<div class="section-label" style="margin-top:1rem">▸ Threat Assessment</div>')
            with gr.Row():
                severity_out = gr.Textbox(
                    label="THREAT LEVEL",
                    interactive=False,
                    elem_classes=["metric-card"],
                )
                loitering_out = gr.Textbox(
                    label="LOITERING EVENTS",
                    interactive=False,
                    elem_classes=["metric-card"],
                )
                panic_out = gr.Textbox(
                    label="PANIC EVENTS",
                    interactive=False,
                    elem_classes=["metric-card"],
                )
                crowd_out = gr.Textbox(
                    label="CROWD ANOMALIES",
                    interactive=False,
                    elem_classes=["metric-card"],
                )

            gr.HTML('<div class="section-label" style="margin-top:1rem">▸ Analysis Report</div>')
            report_out = gr.Textbox(
                label="",
                lines=12,
                max_lines=20,
                interactive=False,
                elem_id="report-output",
                placeholder="Report will appear here after analysis...",
            )

    # Wire up
    analyze_btn.click(
        fn=run_analysis,
        inputs=[video_input, loitering_slider, speed_slider, conf_slider],
        outputs=[live_frame, video_output, severity_out, loitering_out, panic_out, crowd_out, report_out],
    )

    gr.HTML("""
    <div style="text-align:center; padding:1.5rem; font-family:monospace; font-size:0.65rem; color:#2a4050; letter-spacing:0.15em;">
        ABDS · ABNORMAL BEHAVIOR DETECTION SYSTEM · FOR SECURITY RESEARCH USE
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )