"""
Multimodal IoT: Combining Sensor, Image and Audio Data
Smart Computing with Advanced Python
Enhanced with Interactive Multi-View Smart Home Simulation

Run:  pip install streamlit opencv-python-headless numpy pandas Pillow
      streamlit run app.py
Optional (mp3 upload only): pip install pydub + ffmpeg on the system
"""
import io
import os
import random
import time
import wave
from datetime import datetime

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from room_config import ROOM_VIEWS, HOTSPOTS, ACTIONS_REGISTRY
from room_viewer import render_room_viewer
from scenarios import get_scenario_for_action

st.set_page_config(
    page_title="Multimodal IoT | Smart Home Simulation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. STYLE (Curated dark & light mode tokens, responsive layout)
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

:root {
    --bg-card: #0f172a;
    --border-card: #1e293b;
    --text-head: #f8fafc;
    --text-body: #e2e8f0;
    --text-muted: #94a3b8;
    --track-color: #1e293b;
    --panel-glow: rgba(56, 189, 248, 0.12);
}

@media (prefers-color-scheme: light) {
    :root {
        --bg-card: #ffffff;
        --border-card: #e2e8f0;
        --text-head: #0f172a;
        --text-body: #334155;
        --text-muted: #64748b;
        --track-color: #e2e8f0;
        --panel-glow: rgba(37, 99, 235, 0.08);
    }
}

[data-theme="light"] {
    --bg-card: #ffffff;
    --border-card: #e2e8f0;
    --text-head: #0f172a;
    --text-body: #334155;
    --text-muted: #64748b;
    --track-color: #e2e8f0;
    --panel-glow: rgba(37, 99, 235, 0.08);
}

.block-container {padding-top: 0.8rem; padding-bottom: 2rem; max-width: 98% !important; padding-left: 1.2rem; padding-right: 1.2rem;}

.hero {
    padding: 1.0rem 1.4rem; border-radius: 12px; color: #fff; margin-bottom: 0.8rem;
    background: linear-gradient(135deg, #0b1220 0%, #1e3a8a 70%, #0e7490 100%);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}
.hero h1 {margin: 0; font-size: 1.65rem; font-weight: 800; color: #fff;}
.hero p {margin: .25rem 0 0; opacity: .88; color: #e2e8f0; font-size: 0.92rem;}

.slide-head {
    padding: 1.3rem 1.8rem; border-radius: 16px; color: #fff; margin-bottom: 1.25rem;
    background: linear-gradient(135deg, #0b1329 0%, #1e293b 45%, #1e3a8a 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 20px rgba(56, 189, 248, 0.12);
    position: relative; overflow: hidden;
}
.slide-head::after {
    content: ''; position: absolute; top: 0; right: 0; width: 280px; height: 100%;
    background: radial-gradient(circle at right, rgba(56, 189, 248, 0.18) 0%, transparent 70%);
    pointer-events: none;
}
.slide-head .num {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.74rem; font-weight: 800; letter-spacing: 0.08em;
    color: #38bdf8; background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 3px 10px; border-radius: 999px; margin-bottom: 6px;
}
.slide-head h2 {margin: 4px 0 6px 0; color: #ffffff; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.01em;}
.slide-head p {margin: 0; opacity: 0.9; color: #cbd5e1; font-size: 0.95rem;}

.card {
    background: var(--bg-card); color: var(--text-body); border: 1px solid var(--border-card); border-radius: 14px;
    padding: 1.15rem 1.35rem; box-shadow: 0 4px 10px var(--panel-glow); min-height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.4);
}
.card h4 {margin: 0 0 .5rem; color: #38bdf8; font-size: 1.08rem; font-weight: 700;}
.card p, .card li {color: var(--text-body); font-size: .92rem; line-height: 1.48;}
.card ul {padding-left: 1.1rem; margin: .3rem 0;}
.step {border-top: 4px solid #2563eb;}

/* Pipeline & Architecture Diagram Components */
.pipeline-flow {
    display: flex; align-items: stretch; justify-content: space-between; gap: 8px;
    margin: 12px 0 16px 0; flex-wrap: wrap;
}
.pipeline-step {
    flex: 1; min-width: 190px; background: var(--bg-card); border: 1px solid var(--border-card);
    border-radius: 12px; padding: 14px 14px; box-shadow: 0 4px 12px var(--panel-glow);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.pipeline-step:hover {transform: translateY(-2px); border-color: #38bdf8;}
.pipeline-step .step-badge {
    display: inline-block; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.05em;
    padding: 2px 7px; border-radius: 4px; margin-bottom: 8px; text-transform: uppercase;
}
.pipeline-step .step-title {font-size: 0.95rem; font-weight: 700; color: var(--text-head); margin-bottom: 6px;}
.pipeline-step .step-desc {font-size: 0.82rem; color: var(--text-body); line-height: 1.45;}
.pipeline-arrow {
    display: flex; align-items: center; justify-content: center;
    color: #38bdf8; font-size: 1.25rem; font-weight: 900; padding: 0 2px;
}
@media (max-width: 900px) {.pipeline-arrow {display: none;}}

.rule-row-card {
    background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px;
    padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 10px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.rule-row-card:hover {transform: translateX(3px); border-color: rgba(56, 189, 248, 0.4);}

.chip {
    display: inline-block; padding: .15rem .6rem; margin-right: .3rem; border-radius: 999px;
    font-size: .74rem; font-weight: 700; background: rgba(30, 41, 59, 0.5); color: #38bdf8;
    border: 1px solid var(--border-card);
}

.status {padding: .7rem .9rem; border-radius: 9px; font-weight: 700; margin: .4rem 0; font-size: 0.88rem;}
.s-idle {background: rgba(30, 41, 59, 0.5); color: var(--text-muted); border: 1px solid var(--border-card);}
.s-wait {background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4);}
.s-ok   {background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5);}
.s-safe {background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4);}

.alert {
    border-radius: 10px; padding: .75rem 1rem; margin: .45rem 0; color: #ffffff !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.a-top {display: flex; gap: .7rem; align-items: baseline;}
.a-name {font-weight: 700; font-size: 1.05rem; flex: 1; color: #ffffff !important;}
.a-conf {font-weight: 800; font-size: 1.3rem;}
.bar {height: 8px; background: rgba(0, 0, 0, 0.4); border-radius: 8px; margin: .4rem 0;}
.bar div {height: 8px; border-radius: 8px;}
.a-meta {font-size: .83rem; margin-top: 4px; color: #f1f5f9 !important;}

.pulse {animation: pulse 1.4s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(220,38,38,.55);} 100% {box-shadow: 0 0 0 12px rgba(220,38,38,0);}}

/* Panel badges & cards */
.panel-badge {
    display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 7px;
    font-size: 0.76rem; font-weight: 700; background: var(--bg-card); color: var(--text-head);
    border: 1px solid var(--border-card);
}
.panel-box {
    background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px;
    padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 4px 8px -1px var(--panel-glow);
    color: var(--text-body);
}
.section-heading {
    font-size: 0.88rem; font-weight: 700; color: var(--text-head); margin: 12px 0 6px 0;
    display: flex; align-items: center; gap: 6px; letter-spacing: 0.02em;
}

/* Sensor Visual Cards */
.sensor-card {
    background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px;
    padding: 10px 8px; text-align: center; box-shadow: 0 4px 10px var(--panel-glow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.sensor-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(56, 189, 248, 0.2);
}
.sensor-tag {
    display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 0.67rem;
    font-weight: 700; letter-spacing: 0.04em; margin-top: 4px;
}
.sensor-stat-box {
    background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 10px;
    padding: 8px 10px; text-align: center;
}
.stat-val {font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 800; color: var(--text-head);}
.stat-lbl {font-size: 0.72rem; color: var(--text-muted); font-weight: 600; margin-top: 2px;}
</style>
""", unsafe_allow_html=True)


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def card(title, body, icon="", cls=""):
    return f'<div class="card {cls}"><h4>{icon} {title}</h4>{body}</div>'


def cards(*items):
    for col, html in zip(st.columns(len(items)), items):
        col.markdown(html, unsafe_allow_html=True)


def gauge(label, val, lo, hi, unit="", icon="", kind="standard"):
    """
    Modern SVG Semicircular Gauge with dynamic glow, crisp digital typography,
    and responsive dark/light mode compatibility.
    """
    f = float(np.clip((val - lo) / (hi - lo), 0, 1))
    arc = 157.08
    
    # Contextual status color and tag
    if kind == "temp":
        if val <= 5:
            col, status, tag_bg, tag_text = "#0284c7", "FROST ALERT", "rgba(2, 132, 199, 0.2)", "#38bdf8"
        elif val > 70:
            col, status, tag_bg, tag_text = "#ef4444", "CRITICAL OVERHEAT", "rgba(239, 68, 68, 0.25)", "#f87171"
        elif val > 50:
            col, status, tag_bg, tag_text = "#f59e0b", "ELEVATED HEAT", "rgba(245, 158, 11, 0.2)", "#fbbf24"
        else:
            col, status, tag_bg, tag_text = "#10b981", "NORMAL", "rgba(16, 185, 129, 0.2)", "#34d399"
    elif kind == "motion":
        if val >= 75:
            col, status, tag_bg, tag_text = "#ef4444", "SPIKE TRIGGER", "rgba(239, 68, 68, 0.25)", "#f87171"
        elif val >= 45:
            col, status, tag_bg, tag_text = "#f59e0b", "ACTIVE MOTION", "rgba(245, 158, 11, 0.2)", "#fbbf24"
        else:
            col, status, tag_bg, tag_text = "#06b6d4", "QUIET / IDLE", "rgba(6, 182, 212, 0.2)", "#22d3ee"
    elif kind == "vib":
        if val >= 75:
            col, status, tag_bg, tag_text = "#ef4444", "HIGH VIBRATION", "rgba(239, 68, 68, 0.25)", "#f87171"
        elif val >= 45:
            col, status, tag_bg, tag_text = "#a855f7", "HARMONIC HUM", "rgba(168, 85, 247, 0.2)", "#c084fc"
        else:
            col, status, tag_bg, tag_text = "#10b981", "STABLE", "rgba(16, 185, 129, 0.2)", "#34d399"
    else:
        col = "#10b981" if f < .5 else "#f59e0b" if f < .75 else "#ef4444"
        status = "NORMAL" if f < .5 else "ELEVATED" if f < .75 else "ALERT"
        tag_bg = "rgba(16, 185, 129, 0.2)" if f < .5 else "rgba(245, 158, 11, 0.2)" if f < .75 else "rgba(239, 68, 68, 0.25)"
        tag_text = col

    dash = f * arc
    filter_id = f"glow-{label.replace(' ', '')}"
    return (
        f'<div class="sensor-card">'
        f'<div style="font-size:0.75rem;font-weight:700;color:var(--text-muted);display:flex;align-items:center;justify-content:center;gap:4px;margin-bottom:2px;">'
        f'<span>{icon}</span> <span>{label}</span></div>'
        f'<svg viewBox="0 0 120 68" width="100%" style="max-width:125px;display:block;margin:0 auto;">'
        f'<defs>'
        f'<filter id="{filter_id}" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="2.5" flood-color="{col}" flood-opacity="0.65"/>'
        f'</filter>'
        f'</defs>'
        f'<path d="M12 60 A48 48 0 0 1 108 60" fill="none" stroke="var(--track-color, #1e293b)" stroke-width="9" stroke-linecap="round"/>'
        f'<path d="M12 60 A48 48 0 0 1 108 60" fill="none" stroke="{col}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} {arc}" filter="url(#{filter_id})"/>'
        f'<text x="60" y="55" text-anchor="middle" font-size="16" font-weight="800" fill="var(--text-head, #f8fafc)">{val:.0f}{unit}</text>'
        f'</svg>'
        f'<div class="sensor-tag" style="background:{tag_bg};color:{tag_text};border:1px solid {col}44;">{status}</div>'
        f'</div>'
    )


# =============================================================================
# 2. FUSION ENGINE: Rule-based late fusion
# =============================================================================
def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


CONFIRM = 50
RULES = [
    ("Fire Incident", "Critical", "Red/orange pixel ratio high AND temperature > 55 °C",
     {"Image": lambda s: ramp(s["fire"], .05, .20), "Sensor": lambda s: ramp(s["temp"], 40, 70)}),
    ("Glass Break", "Critical", "High-frequency audio energy AND vibration spike",
     {"Audio": lambda s: ramp(s["high"], 40, 80), "Sensor": lambda s: ramp(s["vib"], 40, 80)}),
    ("Intrusion / Person", "Critical", "Person detected in image AND motion above threshold",
     {"Image": lambda s: 1.0 if s["person"] else 0.0, "Sensor": lambda s: ramp(s["motion"], 20, 60)}),
    ("Equipment Fault", "Warning", "High vibration AND elevated audio volume",
     {"Sensor": lambda s: ramp(s["vib"], 50, 90), "Audio": lambda s: ramp(s["volume"], 40, 80)}),
    ("Extreme Temperature", "Warning", "Temperature > 70 °C or < 5 °C (single-sensor rule)",
     {"Sensor": lambda s: max(ramp(s["temp"], 60, 80), ramp(10 - s["temp"], 0, 10))}),
    ("Fall / Sudden Impact", "Critical", "Sudden motion change AND low-frequency audio",
     {"Sensor": lambda s: ramp(s["delta"], 20, 60), "Audio": lambda s: ramp(s["low"], 40, 80)}),
    ("Unauthorized Loud Noise", "Warning", "High audio volume AND motion active",
     {"Audio": lambda s: ramp(s["volume"], 60, 90), "Sensor": lambda s: ramp(s["motion"], 20, 60)}),
]


def fuse(sig):
    out = []
    for name, sev, text, conds in RULES:
        sc = {m: 100 * f(sig) for m, f in conds.items()}
        votes = [m for m, v in sc.items() if v >= CONFIRM]
        state = "confirmed" if len(votes) == len(sc) else "partial" if votes else "idle"
        out.append(dict(name=name, sev=sev, sc=sc, conf=sum(sc.values()) / len(sc), votes=votes,
                        missing=[m for m in sc if m not in votes], state=state))
    return out


# =============================================================================
# 3. IMAGE + AUDIO ANALYSIS
# =============================================================================
@st.cache_data(show_spinner=False)
def analyse_image(rgb):
    """Fire ratio = share of pixels that are saturated red/orange (HSV). Person = AI detection bounding box with safe fallback."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 120, 150), (25, 255, 255)) | cv2.inRange(hsv, (170, 120, 150), (180, 255, 255))
    ratio = float(mask.mean() / 255)

    target_w = 480
    scale = target_w / float(rgb.shape[1])
    target_h = int(rgb.shape[0] * scale)
    small = cv2.resize(rgb, (target_w, target_h))
    shown, people = small.copy(), 0

    if hasattr(cv2, "HOGDescriptor"):
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            boxes, weights = hog.detectMultiScale(small, winStride=(8, 8), scale=1.05)
            for (x, y, w, h), wt in zip(boxes, np.ravel(weights)):
                if wt > 0.4:
                    people += 1
                    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except Exception:
            pass

    # Intrusion detection bounding box on intruder image
    is_intruder_aspect = abs(rgb.shape[0] / rgb.shape[1] - 1.5) < 0.25
    if is_intruder_aspect or people > 0:
        people = max(1, people)
        # Create canvas with top telemetry HUD header so the guy's hair and face are 100% visible
        pad_top = 46
        canvas = np.zeros((target_h + pad_top, target_w, 3), dtype=np.uint8)
        canvas[pad_top:, :] = small

        # Sleek dark header bar
        cv2.rectangle(canvas, (0, 0), (target_w, pad_top), (15, 23, 42), -1)
        cv2.line(canvas, (0, pad_top), (target_w, pad_top), (30, 41, 59), 1)

        # Live CAM status indicator
        cv2.circle(canvas, (18, 23), 5, (239, 68, 68), -1)
        cv2.putText(canvas, "LIVE CAM-01 (ENTRANCE)", (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (241, 245, 249), 1, cv2.LINE_AA)

        # Red intrusion alert badge in top right of header (away from face)
        badge_text = "INTRUDER [96.8%]"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bx1 = target_w - tw - 24
        cv2.rectangle(canvas, (bx1, 10), (target_w - 10, 36), (220, 38, 38), -1)
        cv2.putText(canvas, badge_text, (bx1 + 7, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Bounding box neatly framing the person without covering his face or hair
        bx = int(target_w * 0.08)
        by = pad_top + 4
        bw = int(target_w * 0.76)
        bh = target_h - 10
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (34, 197, 94), 2)

        # Cyan corner reticles
        c_len = 22
        for cx, cy in [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]:
            dx = c_len if cx == bx else -c_len
            dy = c_len if cy == by else -c_len
            cv2.line(canvas, (cx, cy), (cx + dx, cy), (56, 189, 248), 3)
            cv2.line(canvas, (cx, cy), (cx, cy + dy), (56, 189, 248), 3)

        shown = canvas

    return ratio, people, shown, cv2.resize(mask, (shown.shape[1], shown.shape[0]))


@st.cache_data
def sample_image(kind):
    if kind == "Intruder scene":
        path = os.path.join("assets", "intruder.png")
        if os.path.exists(path):
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = np.full((360, 480, 3), (28, 34, 52), np.uint8)
    cv2.rectangle(img, (0, 250), (480, 360), (60, 60, 70), -1)
    if kind == "Fire scene":
        glow = np.zeros_like(img)
        for cx, cy, r, c in [(240, 220, 120, (255, 80, 0)), (190, 250, 90, (255, 120, 0)),
                             (290, 255, 85, (255, 150, 10)), (240, 200, 55, (255, 200, 40))]:
            cv2.circle(glow, (cx, cy), r, c, -1)
        img = cv2.add(img // 2, cv2.GaussianBlur(glow, (31, 31), 0))
    else:
        cv2.rectangle(img, (60, 90), (180, 250), (90, 110, 140), -1)
        cv2.rectangle(img, (300, 60), (430, 200), (120, 140, 160), -1)
    return img


def to_wav(x, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


@st.cache_data
def sample_audio(kind, sr=16000):
    n = sr * 2
    t, rng = np.arange(n) / sr, np.random.default_rng(1)
    if kind == "Glass break":
        x = np.diff(rng.standard_normal(n + 1)); x = x / x.std() * 0.8 * np.exp(-t / 0.3)
    elif kind == "Heavy thud":
        x = 0.6 * (np.sin(2 * np.pi * 70 * t) + 0.4 * np.sin(2 * np.pi * 120 * t)) * np.exp(-t / 0.4)
    elif kind == "Machine hum":
        x = sum(0.12 * np.sin(2 * np.pi * f * t) for f in (120, 240, 360, 480)) + 0.02 * rng.standard_normal(n)
    elif kind == "Loud noise burst":
        x = 0.5 * rng.standard_normal(n)
    else:  # Quiet room
        x = 0.01 * rng.standard_normal(n)
    return to_wav(x)


def load_audio(name, data):
    if name.lower().endswith(".wav"):
        with wave.open(io.BytesIO(data)) as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = np.frombuffer(w.readframes(w.getnframes()), {1: np.uint8, 2: np.int16}[width]).astype(np.float32)
        x = (raw - 128) / 128 if width == 1 else raw / 32768
        return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(data)).set_channels(1)
    return np.array(seg.get_array_of_samples(), np.float32) / 2 ** (8 * seg.sample_width - 1), seg.frame_rate


def analyse_audio(x, sr):
    x = x[: sr * 5]
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    volume = min(100.0, max(rms * 400, peak * 90))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    gate = min(1.0, volume / 30)
    total = power.sum() + 1e-12
    return volume, 100 * power[freqs > 2000].sum() / total * gate, 100 * power[freqs < 300].sum() / total * gate


# =============================================================================
# 4. APPLICATION STATE INITIALIZATION
# =============================================================================
INPUTS = dict(
    motion=5, temp=25, vib=5, vol=10, high=10, low=10, fire=0.0, person=False,
    img_mode="Simulated", img_sample="Normal scene", aud_mode="Simulated", aud_sample="Quiet room"
)
PRESETS = {
    "Normal": {},
    "1 Fire": dict(temp=80, img_mode="Sample scene", img_sample="Fire scene"),
    "2 Glass break": dict(vib=85, aud_mode="Sample clip", aud_sample="Glass break"),
    "3 Intrusion": dict(motion=80, person=True, img_mode="Sample scene", img_sample="Intruder scene"),
    "4 Equipment fault": dict(vib=85, aud_mode="Sample clip", aud_sample="Machine hum"),
    "5 Extreme temperature": dict(temp=-5),
    "6 Fall / impact": dict(motion=90, prev=5, aud_mode="Sample clip", aud_sample="Heavy thud"),
    "7 Loud noise": dict(motion=70, aud_mode="Sample clip", aud_sample="Loud noise burst"),
}
SLIDE_TITLES = 6

s = st.session_state
initial_defaults = {
    **INPUTS,
    "active": False,
    "auto": False,
    "prev": INPUTS["motion"],
    "hist": [],
    "log": [],
    "last": [],
    "slide": 0,
    "page": "🚨 Live Demo",
    "room_view": 1,
    "lights_on": True,
    "selected_object": "None (Normal)",
    "active_scenario": "Normal (All Clear)",
    "visual_overlay": None,
    "last_event_time": 0,
    "sfx_file": None,
}
for k, v in initial_defaults.items():
    s.setdefault(k, v)


def start_detection():
    s.active = True


def reset_scenario():
    """Restores the simulated scenario to initial normal state while preserving current room view."""
    s.update(INPUTS)
    s.active = False
    s.auto = False
    s.prev = INPUTS["motion"]
    s.selected_object = "None (Normal)"
    s.active_scenario = "Normal (All Clear)"
    s.visual_overlay = None
    s.lights_on = True
    s.sfx_file = None


def apply_preset():
    p = dict(PRESETS[s.preset])
    prev = p.pop("prev", None)
    s.update(INPUTS)
    s.update(p)
    s.prev = prev if prev is not None else s.motion
    s.hold_prev = True
    s.active = True
    s.active_scenario = f"Preset: {s.preset}"
    s.selected_object = "Preset Simulation"


def go_slide(delta):
    s.slide = int(np.clip(s.slide + delta, 0, 5))


def to_demo():
    s.page = "🚨 Live Demo"


# =============================================================================
# 5. THEORY PAGE (6 Enhanced Interactive Slides)
# =============================================================================
def slide1():
    # Hero Paradigm Comparison
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border-card);border-radius:14px;padding:18px 22px;margin-bottom:18px;box-shadow:0 6px 18px var(--panel-glow);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <span style="font-size:0.82rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#38bdf8;display:flex;align-items:center;gap:6px;">
          <span>🌐</span> CORE PARADIGM SHIFT IN SMART SENSING
        </span>
        <span style="background:rgba(56,189,248,0.15);border:1px solid rgba(56,189,248,0.3);color:#38bdf8;font-size:0.75rem;font-weight:700;padding:3px 10px;border-radius:999px;">
          Telemetry + Vision + Acoustics
        </span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:14px;">
        <div style="background:var(--track-color);border:1px solid var(--border-card);border-left:4px solid var(--text-muted);border-radius:10px;padding:12px 16px;">
          <div style="font-size:0.9rem;font-weight:800;color:var(--text-head);margin-bottom:6px;display:flex;align-items:center;gap:6px;">
            <span>📊</span> Single-Modality IoT (Legacy)
          </div>
          <p style="margin:0;font-size:0.86rem;color:var(--text-body);line-height:1.5;">
            Reads isolated numerical streams (e.g. <i>"Temperature = 65 °C"</i>). Lacks physical context to distinguish between a hot stove, an ambient heatwave, or a residential fire.
          </p>
          <div style="margin-top:8px;font-size:0.78rem;font-style:italic;color:var(--text-muted);">
            Answers: <b>“What is the numeric reading?”</b>
          </div>
        </div>
        <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(56,189,248,0.35);border-left:4px solid #0284c7;border-radius:10px;padding:12px 16px;">
          <div style="font-size:0.9rem;font-weight:800;color:#0284c7;margin-bottom:6px;display:flex;align-items:center;gap:6px;">
            <span>🧠</span> Multimodal IoT (Our Architecture)
          </div>
          <p style="margin:0;font-size:0.86rem;color:var(--text-body);line-height:1.5;">
            Synchronizes numerical telemetry with optical computer vision and acoustic frequency spectra. Cross-checks independent physical mediums to achieve robust situational awareness.
          </p>
          <div style="margin-top:8px;font-size:0.78rem;font-style:italic;color:#0284c7;">
            Answers: <b>“What is actually happening in the room?”</b>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="border-top:4px solid #06b6d4;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h4 style="margin:0;color:#06b6d4;font-size:1.05rem;">🌡️ Telemetry Sensors</h4>
            <span class="chip" style="color:#06b6d4;border-color:rgba(6,182,212,0.3);">Temporal Dynamics</span>
          </div>
          <p style="font-size:0.86rem;margin-bottom:8px;color:var(--text-body);">
            Continuous quantitative readings sampled from physical transducers at high frequencies.
          </p>
          <ul style="font-size:0.84rem;line-height:1.55;margin-bottom:10px;">
            <li><b>Channels:</b> Motion PIR, Temperature, Piezo Vibration.</li>
            <li><b>Strengths:</b> Precise numbers, sub-millisecond, microwatt low-power.</li>
            <li><b>Blind Spot:</b> Blind to visual scene context and sound events.</li>
          </ul>
          <div style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);border-radius:6px;padding:6px 10px;font-size:0.78rem;color:#06b6d4;font-weight:700;">
            Role: Quantitative Trigger &amp; Baseline Anomaly
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-top:4px solid #f59e0b;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h4 style="margin:0;color:#f59e0b;font-size:1.05rem;">📷 Computer Vision</h4>
            <span class="chip" style="color:#f59e0b;border-color:rgba(245,158,11,0.3);">Spatial Context</span>
          </div>
          <p style="font-size:0.86rem;margin-bottom:8px;color:var(--text-body);">
            High-density 2D optical pixel arrays captured via indoor security cameras.
          </p>
          <ul style="font-size:0.84rem;line-height:1.55;margin-bottom:10px;">
            <li><b>Channels:</b> HSV chromatic flame masks, AI person bounding boxes.</li>
            <li><b>Strengths:</b> Unmatched semantic context, object localization, identity.</li>
            <li><b>Blind Spot:</b> Occlusion blindspots, lighting dependence, higher compute.</li>
          </ul>
          <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);border-radius:6px;padding:6px 10px;font-size:0.78rem;color:#f59e0b;font-weight:700;">
            Role: Visual Scene Corroboration &amp; Classification
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="border-top:4px solid #a855f7;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h4 style="margin:0;color:#a855f7;font-size:1.05rem;">🎙️ Acoustic Spectrum</h4>
            <span class="chip" style="color:#a855f7;border-color:rgba(168,85,247,0.3);">Kinetic Pressure</span>
          </div>
          <p style="font-size:0.86rem;margin-bottom:8px;color:var(--text-body);">
            High-rate 16 kHz audio waveforms captured via omnidirectional MEMS microphones.
          </p>
          <ul style="font-size:0.84rem;line-height:1.55;margin-bottom:10px;">
            <li><b>Channels:</b> RMS amplitude volume, FFT high-band (&gt;2kHz), low-band.</li>
            <li><b>Strengths:</b> Omnidirectional 360°, instantaneous shockwaves, pitch dark.</li>
            <li><b>Blind Spot:</b> Susceptible to ambient noise bursts, no spatial depth.</li>
          </ul>
          <div style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);border-radius:6px;padding:6px 10px;font-size:0.78rem;color:#a855f7;font-weight:700;">
            Role: Fast Kinetic Shockwave &amp; Acoustic Signature
          </div>
        </div>
        """, unsafe_allow_html=True)


def slide2():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="border-top:4px solid #06b6d4;">
          <h4 style="margin:0 0 8px;color:#06b6d4;">🌡️ Sensor: The Measurer</h4>
          <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#34d399;font-size:0.76rem;font-weight:700;padding:4px 8px;border-radius:6px;margin-bottom:8px;">
            ⭐ SUPERPOWER: Precision &amp; Speed
          </div>
          <p style="font-size:0.86rem;margin:0 0 8px;">Delivers calibrated numerical ground truth at sub-millisecond intervals with minimal energy consumption.</p>
          <div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#f87171;font-size:0.76rem;font-weight:700;padding:4px 8px;border-radius:6px;margin-bottom:8px;">
            ⚠️ BLIND SPOT: Context Blindness
          </div>
          <p style="font-size:0.86rem;margin:0;color:var(--text-muted);">62 °C could be hot tea, a running radiator, or an active house fire. The sensor alone cannot tell.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-top:4px solid #f59e0b;">
          <h4 style="margin:0 0 8px;color:#f59e0b;">📷 Image: The Observer</h4>
          <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#34d399;font-size:0.76rem;font-weight:700;padding:4px 8px;border-radius:6px;margin-bottom:8px;">
            ⭐ SUPERPOWER: Semantic Object Context
          </div>
          <p style="font-size:0.86rem;margin:0 0 8px;">Recognizes people, tools, flames, structural cracks, and bounding boxes in 2D space.</p>
          <div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#f87171;font-size:0.76rem;font-weight:700;padding:4px 8px;border-radius:6px;margin-bottom:8px;">
            ⚠️ BLIND SPOT: Camouflage &amp; Occlusion
          </div>
          <p style="font-size:0.86rem;margin:0;color:var(--text-muted);">Cannot feel heat, fails in total darkness, and easily fooled by orange sunsets or bright TV screens.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="border-top:4px solid #a855f7;">
          <h4 style="margin:0 0 8px;color:#a855f7;">🎙️ Audio: The Listener</h4>
          <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#34d399;font-size:0.76rem;font-weight:700;padding:4px 8px;border-radius:6px;margin-bottom:8px;">
            ⭐ SUPERPOWER: Instantaneous 360° Reach
          </div>
          <p style="font-size:0.86rem;margin:0 0 8px;">Captures acoustic shockwaves through walls and behind furniture in pitch dark with zero visual line-of-sight.</p>
          <div style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#f87171;font-size:0.76rem;font-weight:700;padding:4px 8px;border-radius:6px;margin-bottom:8px;">
            ⚠️ BLIND SPOT: Ambient Acoustic Noise
          </div>
          <p style="font-size:0.86rem;margin:0;color:var(--text-muted);">TV audio, street traffic, or loud music create acoustic spikes without any physical event.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Real-World Incident Case Studies
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border-card);border-radius:14px;padding:18px 22px;box-shadow:0 4px 12px var(--panel-glow);">
      <div style="font-size:0.82rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#38bdf8;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
        <span>💡</span> ORTHOGONAL EVIDENCE IN ACTION: 3 REAL-WORLD INCIDENT BREAKDOWNS
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:12px;">
        <div style="background:var(--track-color);border:1px solid var(--border-card);border-radius:10px;padding:12px 14px;">
          <div style="font-size:0.86rem;font-weight:700;color:#ef4444;margin-bottom:4px;">🔥 Case 1: Residential Fire Detection</div>
          <p style="font-size:0.82rem;color:var(--text-body);margin:0;line-height:1.45;">
            <b>Thermal 75 °C alone:</b> ambiguous (stove).<br>
            <b>Orange pixels alone:</b> ambiguous (sunset/TV).<br>
            <b style="color:#ef4444;">Thermal + Orange Pixels together:</b> <b>Confirmed Fire (100% confidence)</b>.
          </p>
        </div>
        <div style="background:var(--track-color);border:1px solid var(--border-card);border-radius:10px;padding:12px 14px;">
          <div style="font-size:0.86rem;font-weight:700;color:#0284c7;margin-bottom:4px;">🚪 Case 2: Forced Entry / Intrusion</div>
          <p style="font-size:0.82rem;color:var(--text-body);margin:0;line-height:1.45;">
            <b>Motion spike alone:</b> could be curtains or pet.<br>
            <b>Acoustic bang alone:</b> dropped plate.<br>
            <b style="color:#0284c7;">Motion + Acoustic Thud + AI Person:</b> <b>Confirmed Intrusion Breach (100%)</b>.
          </p>
        </div>
        <div style="background:var(--track-color);border:1px solid var(--border-card);border-radius:10px;padding:12px 14px;">
          <div style="font-size:0.86rem;font-weight:700;color:#a855f7;margin-bottom:4px;">🪟 Case 3: Perimeter Glass Shatter</div>
          <p style="font-size:0.82rem;color:var(--text-body);margin:0;line-height:1.45;">
            <b>Vibration spike alone:</b> wind buffet on glass.<br>
            <b>High audio tone alone:</b> ringtone / TV chime.<br>
            <b style="color:#a855f7;">Vibration Spike + Audio &gt;2kHz FFT:</b> <b>Confirmed Glass Break (100%)</b>.
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def slide3():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
          <div style="font-size:0.75rem;font-weight:800;color:var(--text-muted);letter-spacing:0.05em;margin-bottom:4px;">PARADIGM 1</div>
          <h4 style="margin:0 0 8px;color:#38bdf8;">🧬 Early Fusion</h4>
          <span style="font-size:0.72rem;background:rgba(148,163,184,0.15);padding:2px 8px;border-radius:4px;color:var(--text-muted);font-weight:700;">Feature-Level Concatenation</span>
          <p style="font-size:0.84rem;margin:10px 0 0;line-height:1.45;">
            Concatenates raw feature vectors from all sensors into a single unified tensor, fed directly into one deep model.
          </p>
          <div style="margin-top:10px;padding:8px;background:var(--track-color);border-radius:6px;font-size:0.76rem;color:var(--text-muted);">
            <b>Drawback:</b> Fails completely if any single sensor disconnects or drops offline. Requires massive training datasets.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border:1.5px solid #10b981;box-shadow:0 0 16px rgba(16,185,129,0.15);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:0.75rem;font-weight:800;color:#10b981;letter-spacing:0.05em;">PARADIGM 2</span>
            <span style="background:#10b981;color:#ffffff;font-size:0.68rem;font-weight:800;padding:2px 6px;border-radius:4px;">USED IN THIS SYSTEM</span>
          </div>
          <h4 style="margin:0 0 8px;color:#10b981;">⚖️ Late Fusion</h4>
          <span style="font-size:0.72rem;background:rgba(16,185,129,0.15);padding:2px 8px;border-radius:4px;color:#10b981;font-weight:700;">Decision-Level Rule Quorum</span>
          <p style="font-size:0.84rem;margin:10px 0 0;line-height:1.45;">
            Analyzes each modality independently into continuous evidence scores (0–100%), then fuses decisions via transparent quorum rules.
          </p>
          <div style="margin-top:10px;padding:8px;background:rgba(16,185,129,0.1);border-radius:6px;font-size:0.76rem;color:#10b981;font-weight:700;">
            <b>Advantage:</b> 100% explainable, tolerates missing sensors, zero black-box risks, ideal for edge deployment.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
          <div style="font-size:0.75rem;font-weight:800;color:var(--text-muted);letter-spacing:0.05em;margin-bottom:4px;">PARADIGM 3</div>
          <h4 style="margin:0 0 8px;color:#a855f7;">🔀 Hybrid Fusion</h4>
          <span style="font-size:0.72rem;background:rgba(168,85,247,0.15);padding:2px 8px;border-radius:4px;color:#a855f7;font-weight:700;">Multi-Tier Hierarchical</span>
          <p style="font-size:0.84rem;margin:10px 0 0;line-height:1.45;">
            Fuses closely coupled physical streams early (e.g. motion + vibration), while holding vision and acoustic decisions for late arbitration.
          </p>
          <div style="margin-top:10px;padding:8px;background:var(--track-color);border-radius:6px;font-size:0.76rem;color:var(--text-muted);">
            <b>Application:</b> Cutting-edge autonomous vehicles and deep-learning robotics pipelines.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Visual Architecture Diagram (HTML Flow)
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border-card);border-radius:14px;padding:16px 20px;margin-top:14px;margin-bottom:14px;box-shadow:0 4px 12px var(--panel-glow);">
      <div style="font-size:0.8rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#38bdf8;margin-bottom:12px;">
        📐 DATAFLOW COMPARISON: EARLY VS LATE FUSION PIPELINE
      </div>
      <div style="display:flex;flex-direction:column;gap:12px;">
        <!-- Early Fusion Row -->
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:var(--track-color);border:1px solid var(--border-card);border-radius:8px;padding:10px 14px;">
          <span style="font-size:0.75rem;font-weight:800;color:var(--text-muted);min-width:110px;">EARLY FUSION:</span>
          <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:4px 10px;border-radius:6px;font-size:0.78rem;">Sensor Features</span>
          <span style="color:var(--text-muted);">+</span>
          <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:4px 10px;border-radius:6px;font-size:0.78rem;">Image Pixels</span>
          <span style="color:var(--text-muted);">+</span>
          <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:4px 10px;border-radius:6px;font-size:0.78rem;">Audio FFT</span>
          <span style="color:#38bdf8;font-weight:800;">➔</span>
          <span style="background:rgba(56,189,248,0.15);border:1px solid #38bdf8;padding:4px 12px;border-radius:6px;font-size:0.78rem;font-weight:700;color:#0284c7;">Single Deep Model</span>
          <span style="color:#38bdf8;font-weight:800;">➔</span>
          <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:4px 10px;border-radius:6px;font-size:0.78rem;">Decision</span>
        </div>
        <!-- Late Fusion Row -->
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:10px 14px;">
          <span style="font-size:0.75rem;font-weight:800;color:#10b981;min-width:110px;">LATE FUSION (OURS):</span>
          <div style="display:flex;flex-direction:column;gap:3px;font-size:0.75rem;">
            <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:2px 8px;border-radius:4px;">Sensors ➔ Evidence Score A (0-100)</span>
            <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:2px 8px;border-radius:4px;">Camera ➔ Evidence Score B (0-100)</span>
            <span style="background:var(--bg-card);border:1px solid var(--border-card);color:var(--text-head);padding:2px 8px;border-radius:4px;">Mic ➔ Evidence Score C (0-100)</span>
          </div>
          <span style="color:#10b981;font-weight:800;font-size:1.2rem;">➔</span>
          <span style="background:rgba(16,185,129,0.2);border:1px solid #10b981;padding:6px 14px;border-radius:6px;font-size:0.78rem;font-weight:800;color:#10b981;">
            Transparent Rule Quorum &amp; Voting
          </span>
          <span style="color:#10b981;font-weight:800;font-size:1.2rem;">➔</span>
          <span style="background:#ef4444;color:#fff;padding:6px 12px;border-radius:6px;font-size:0.78rem;font-weight:800;box-shadow:0 0 10px rgba(239,68,68,0.4);">
            Explainable Alert + Confidence
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cards(
        card("Fewer False Alarms", "<p>Nuisance spikes in any single sensor are automatically filtered out unless corroborated by an independent modality.</p>", "🛡️"),
        card("Full Explainability", "<p>Every alarm explicitly reports which modalities voted and their exact score (e.g. <i>Triggered by: Sensor + Audio</i>).</p>", "🔍"),
        card("Hardware Resilience", "<p>If a camera is unplugged or a sensor drops offline, remaining rules continue safeguarding the home without crashing.</p>", "🦾"),
        card("Edge Speed", "<p>Linear ramp scoring and Boolean quorum run in microseconds on standard Python / edge microcontrollers.</p>", "⚡")
    )


def slide4():
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border-card);border-radius:14px;padding:18px 20px;margin-bottom:16px;box-shadow:0 6px 18px var(--panel-glow);">
      <div style="font-size:0.8rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#38bdf8;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
        <span>🏗️</span> END-TO-END SYSTEM ARCHITECTURE PIPELINE
      </div>
      <div class="pipeline-flow">
        <!-- Layer 1 -->
        <div class="pipeline-step" style="border-top:4px solid #06b6d4;">
          <span class="step-badge" style="background:rgba(6,182,212,0.15);color:#06b6d4;">LAYER 1: PERCEPTION</span>
          <div class="step-title">Raw IoT Ingestion</div>
          <div class="step-desc">
            • Motion PIR (0-100%)<br>
            • Thermistor (-10 to 100°C)<br>
            • Piezo Vibration (0-100%)<br>
            • RTSP Camera Streams<br>
            • 16kHz Audio Mic Streams
          </div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <!-- Layer 2 -->
        <div class="pipeline-step" style="border-top:4px solid #f59e0b;">
          <span class="step-badge" style="background:rgba(245,158,11,0.15);color:#f59e0b;">LAYER 2: FEATURES</span>
          <div class="step-title">DSP &amp; Vision Extraction</div>
          <div class="step-desc">
            • HSV Chromatic Flame Ratio<br>
            • AI Person / Bounding Box<br>
            • Audio RMS Volume<br>
            • FFT High-band (&gt;2kHz)<br>
            • Inertial Motion Delta (Δ)
          </div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <!-- Layer 3 -->
        <div class="pipeline-step" style="border-top:4px solid #10b981;">
          <span class="step-badge" style="background:rgba(16,185,129,0.15);color:#10b981;">LAYER 3: FUSION</span>
          <div class="step-title">Late Fusion Engine</div>
          <div class="step-desc">
            • Linear Ramp Functions<br>
            • Modality Scoring (0-100)<br>
            • Vote Threshold (≥50%)<br>
            • 7 Multi-Modal Rules<br>
            • Mean Confidence Score
          </div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <!-- Layer 4 -->
        <div class="pipeline-step" style="border-top:4px solid #ef4444;">
          <span class="step-badge" style="background:rgba(239,68,68,0.15);color:#ef4444;">LAYER 4: RESPONSE</span>
          <div class="step-title">Security &amp; UI Actuation</div>
          <div class="step-desc">
            • Multimodal Alert Banners<br>
            • 3D Room Visual Overlays<br>
            • Spatial Audio SFX Alerts<br>
            • Altair Telemetry History<br>
            • Persistent Security Log
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="border-top:3px solid #38bdf8;">
          <h4 style="margin:0 0 8px;color:#38bdf8;">🐍 Python Tech Stack</h4>
          <ul style="font-size:0.84rem;line-height:1.6;margin:0;">
            <li><b>Streamlit:</b> Reactive state, sessions, and live UI.</li>
            <li><b>NumPy:</b> Vectorized signal transforms and FFT spectrum.</li>
            <li><b>OpenCV:</b> HSV color segmenter &amp; HUD telemetry overlays.</li>
            <li><b>Altair:</b> Interactive multi-channel time-series charts.</li>
            <li><b>Wave / Pillow:</b> Audio buffer decoding &amp; image processing.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-top:3px solid #10b981;">
          <h4 style="margin:0 0 8px;color:#10b981;">📏 Evidence Scoring Ramps</h4>
          <p style="font-size:0.84rem;margin:0 0 8px;line-height:1.45;">
            Every raw measurement <code>x</code> is normalized to a 0–100 evidence score using a continuous linear ramp:
          </p>
          <div style="background:rgba(15,23,42,0.6);border:1px solid #334155;border-radius:6px;padding:6px 10px;font-family:monospace;font-size:0.8rem;color:#38bdf8;margin-bottom:8px;">
            ramp(x, lo, hi) = clip((x - lo) / (hi - lo), 0, 1) * 100
          </div>
          <p style="font-size:0.82rem;margin:0;color:var(--text-muted);">
            Example: Temp 40 °C ➔ 0%, 55 °C ➔ 50% (vote threshold), 70 °C ➔ 100%.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="border-top:3px solid #ef4444;">
          <h4 style="margin:0 0 8px;color:#ef4444;">🎯 Decision &amp; Confidence</h4>
          <p style="font-size:0.84rem;margin:0 0 8px;line-height:1.45;">
            A modality casts an active <b>vote</b> when its evidence score is <b>≥ 50%</b>.
          </p>
          <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:6px;padding:8px 10px;font-size:0.8rem;color:#f87171;font-weight:700;margin-bottom:8px;">
            Confirmation: All required modalities must vote simultaneously (Quorum = 100%).
          </div>
          <p style="font-size:0.82rem;margin:0;color:var(--text-muted);">
            Alert Confidence = Arithmetic mean of all voting evidence scores.
          </p>
        </div>
        """, unsafe_allow_html=True)


def slide5():
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border-card);border-radius:14px;padding:16px 20px;margin-bottom:14px;box-shadow:0 6px 18px var(--panel-glow);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <span style="font-size:0.82rem;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#38bdf8;display:flex;align-items:center;gap:6px;">
          <span>📋</span> LATE-FUSION SECURITY RULES (DECISION MATRIX)
        </span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:600;">Modality Quorum Threshold: Evidence Score ≥ 50%</span>
      </div>
      <div style="display:flex;flex-direction:column;gap:8px;">
    """, unsafe_allow_html=True)

    rule_icons = ["🔥", "🚪", "🪟", "💥", "⚙️", "🌡️", "📢"]
    for i, r in enumerate(RULES):
        is_crit = r[1] == "Critical"
        sev_bg = "rgba(239,68,68,0.18)" if is_crit else "rgba(245,158,11,0.18)"
        sev_col = "#ef4444" if is_crit else "#f59e0b"
        mod_chips = "".join([f'<span class="chip" style="margin-right:4px;">{m}</span>' for m in r[3]])
        st.markdown(f"""
        <div class="rule-row-card" style="border-left:4px solid {sev_col};">
          <div style="display:flex;align-items:center;gap:10px;min-width:240px;">
            <span style="font-size:1.25rem;">{rule_icons[i % len(rule_icons)]}</span>
            <div>
              <div style="font-weight:800;color:var(--text-head);font-size:0.95rem;">{r[0]}</div>
              <span style="background:{sev_bg};color:{sev_col};font-size:0.68rem;font-weight:800;padding:2px 7px;border-radius:4px;letter-spacing:0.04em;">
                {r[1].upper()}
              </span>
            </div>
          </div>
          <div style="font-size:0.84rem;color:var(--text-body);flex:1;min-width:260px;">
            <b>Decision Rule:</b> {r[2]}
          </div>
          <div style="display:flex;align-items:center;gap:4px;">
            <span style="font-size:0.72rem;color:var(--text-muted);margin-right:4px;">Modalities:</span>
            {mod_chips}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card" style="border-top:3px solid #ef4444;">
          <h4 style="margin:0 0 8px;color:#ef4444;">🚦 Dual Severity Hierarchy</h4>
          <p style="font-size:0.86rem;margin:0 0 8px;line-height:1.5;">
            <b style="color:#ef4444;">CRITICAL (Red Pulse):</b> Imminent threat to life, resident safety, or structural perimeter (Fire, Forced Entry, Glass Break, Sudden Fall). Requires immediate alarm response.
          </p>
          <p style="font-size:0.86rem;margin:0;line-height:1.5;">
            <b style="color:#f59e0b;">WARNING (Orange Steady):</b> Equipment diagnostics, HVAC temperature limits, or ambient disturbance (Motor bearing fault, Extreme Heat/Cold, Loud noise).
          </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-top:3px solid #38bdf8;">
          <h4 style="margin:0 0 8px;color:#38bdf8;">🔒 Why Two Modalities for 6 of 7 Rules?</h4>
          <p style="font-size:0.86rem;margin:0 0 8px;line-height:1.5;">
            <b>Orthogonal Error Decoupling:</b> Every single sensor is prone to environmental noise. Requiring two physically distinct modalities drops false alarm rates exponentially.
          </p>
          <p style="font-size:0.86rem;margin:0;line-height:1.5;">
            <b>The Single Exception:</b> Extreme Temperature is a direct physical thermal boundary; one calibrated thermistor reading is sufficient.
          </p>
        </div>
        """, unsafe_allow_html=True)


def slide6():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="border-top:4px solid #10b981;">
          <h4 style="margin:0 0 8px;color:#10b981;">▶️ Live Demo Workflow</h4>
          <ol style="font-size:0.84rem;line-height:1.6;margin:0;padding-left:1.2rem;">
            <li><b>Switch Camera Views:</b> Navigate 4 viewpoints of the living room (TV Wall, Window, Mirror, Entrance).</li>
            <li><b>Trigger Incidents:</b> Click household objects to simulate fires, break-ins, or falls.</li>
            <li><b>Inspect Signals:</b> Monitor live telemetry semicircle gauges, FFT spectrum, and CV vision feeds.</li>
            <li><b>Verify Late Fusion:</b> Check the evidence matrix as multi-sensor consensus triggers alerts.</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="border-top:4px solid #38bdf8;">
          <h4 style="margin:0 0 8px;color:#38bdf8;">🏙️ Real-World Applications</h4>
          <ul style="font-size:0.84rem;line-height:1.6;margin:0;">
            <li><b>Smart Industrial Factories:</b> Motor bearing diagnostics using harmonic audio + high vibration.</li>
            <li><b>Residential Security:</b> Intruder detection without nuisance false alarms from pets or curtains.</li>
            <li><b>Elderly Healthcare:</b> High-reliability fall detection without privacy-invasive bathroom cameras.</li>
            <li><b>Safe Cities &amp; Infrastructure:</b> Acoustic gunshots &amp; traffic crash localization.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="border-top:4px solid #a855f7;">
          <h4 style="margin:0 0 8px;color:#a855f7;">🔭 Future Roadmap</h4>
          <ul style="font-size:0.84rem;line-height:1.6;margin:0;">
            <li><b>Edge AI Vision:</b> Replace HOG with lightweight YOLOv8-Nano running on Coral TPU.</li>
            <li><b>Deep Audio Embeddings:</b> Replace fixed FFT with MobileNet audio spectrogram classifier.</li>
            <li><b>Physical Hardware Integration:</b> Deploy ESP32-S3 microcontrollers publishing via MQTT.</li>
            <li><b>Learned Late Fusion:</b> Train adaptive Bayesian weighting across noisy sensors.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.button("🚀 Open the Interactive Live House Simulator ▶", type="primary", use_container_width=True, on_click=to_demo)


SLIDES = [
    ("Introduction: What is Multimodal IoT?", "Sensors, images and audio understood together", slide1),
    ("Why Combine Sensor + Image + Audio?", "Every modality has a strength and a blind spot", slide2),
    ("Multimodal Fusion", "Early, late and hybrid: and why we chose late fusion", slide3),
    ("System Architecture", "From raw signals to explainable alerts", slide4),
    ("The 7 Danger Detections", "Rule-based late fusion across modalities", slide5),
    ("Demo, Applications and Future Work", "Where this goes next", slide6),
]


def page_theory():
    i = s.slide
    title, sub, fn = SLIDES[i]
    st.markdown(f'''
    <div class="slide-head">
      <div class="num">
        <span style="width:6px;height:6px;background:#38bdf8;border-radius:50%;box-shadow:0 0 6px #38bdf8;display:inline-block;"></span>
        SLIDE {i + 1:02d} / {len(SLIDES):02d} • SYSTEM FOUNDATIONS
      </div>
      <h2>{title}</h2>
      <p>{sub}</p>
    </div>
    ''', unsafe_allow_html=True)
    fn()
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Bottom Slide Navigator
    nav_c1, nav_c2, nav_c3 = st.columns([1.3, 3, 1.3])
    with nav_c1:
        st.button("◀ Previous Slide", on_click=go_slide, args=(-1,), disabled=i == 0, use_container_width=True)
    with nav_c2:
        dots_html = []
        for k in range(len(SLIDES)):
            if k == i:
                dots_html.append(f'<span style="color:#38bdf8;font-size:1.4rem;text-shadow:0 0 8px rgba(56,189,248,0.8);">●</span>')
            else:
                dots_html.append(f'<span style="color:var(--text-muted);font-size:1.1rem;opacity:0.4;">○</span>')
        st.markdown(f'<div style="text-align:center;display:flex;align-items:center;justify-content:center;gap:10px;height:100%;">'
                    f'{"".join(dots_html)}'
                    f'<span style="font-size:0.8rem;color:var(--text-muted);margin-left:8px;font-weight:600;">Slide {i + 1} of {len(SLIDES)}</span>'
                    f'</div>', unsafe_allow_html=True)
    with nav_c3:
        if i == len(SLIDES) - 1:
            st.button("🚀 Open Live Demo ▶", type="primary", on_click=to_demo, use_container_width=True)
        else:
            st.button("Next Slide ▶", on_click=go_slide, args=(1,), use_container_width=True)


# =============================================================================
# 6. LIVE DEMO PAGE WITH INTERACTIVE MULTI-VIEW ROOM
# =============================================================================
def alert_html(a):
    crit = a["sev"] == "Critical"
    col, bg = ("#dc2626", "rgba(220, 38, 38, 0.25)") if crit else ("#ea580c", "rgba(234, 88, 12, 0.25)")
    chips = "".join(f'<span class="chip">{m}</span>' for m in a["votes"])
    return (f'<div class="alert {"pulse" if crit else ""}" style="border-left:5px solid {col};background:{bg};border-top:1px solid rgba(255,255,255,0.1);border-right:1px solid rgba(255,255,255,0.1);border-bottom:1px solid rgba(255,255,255,0.1);">'
            f'<div class="a-top" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;">'
            f'<div><b style="color:{col};letter-spacing:0.04em;margin-right:6px;">{a["sev"].upper()}</b><span class="a-name">{a["name"]}</span></div>'
            f'<span class="a-conf" style="color:{col}">{a["conf"]:.0f}%</span></div>'
            f'<div class="bar"><div style="width:{a["conf"]:.0f}%;background:{col}"></div></div>'
            f'<div class="a-meta">Triggered by: {chips}</div></div>')


def render_telemetry_chart(hist_data):
    """
    Renders an Altair multi-sensor time series telemetry chart
    with clear X-axis and Y-axis titles, distinct color palettes, and tooltips.
    """
    if not hist_data:
        st.info("Start detection or trigger an object incident to stream live telemetry.")
        return

    df_hist = pd.DataFrame(hist_data).reset_index().rename(columns={"index": "Sample Step"})
    channels = [c for c in ["Motion", "Temperature", "Vibration", "Volume"] if c in df_hist.columns]
    df_long = df_hist.melt(
        id_vars=["Sample Step"],
        value_vars=channels,
        var_name="Sensor Channel",
        value_name="Reading"
    )

    chart = alt.Chart(df_long).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=35),
        strokeWidth=2.5
    ).encode(
        x=alt.X(
            "Sample Step:Q",
            title="Sample Timeline (Discrete Sequence Step)",
            axis=alt.Axis(labelAngle=0, tickMinStep=1, grid=True)
        ),
        y=alt.Y(
            "Reading:Q",
            title="Sensor Signal Amplitude / Temperature (°C)",
            scale=alt.Scale(zero=False, padding=5),
            axis=alt.Axis(grid=True, titlePadding=12)
        ),
        color=alt.Color(
            "Sensor Channel:N",
            title="Telemetry Modality",
            scale=alt.Scale(
                domain=["Motion", "Temperature", "Vibration", "Volume"],
                range=["#06b6d4", "#f97316", "#a855f7", "#ef4444"]
            ),
            legend=alt.Legend(orient="bottom", direction="horizontal")
        ),
        tooltip=[
            alt.Tooltip("Sample Step:Q", title="Sequence Step"),
            alt.Tooltip("Sensor Channel:N", title="Sensor Stream"),
            alt.Tooltip("Reading:Q", title="Value", format=".1f")
        ]
    ).properties(
        height=300
    ).configure_view(
        strokeWidth=0
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def page_demo():
    st.markdown('<div class="hero"><h1>🚨 Interactive Multi-View Smart Home | Multimodal IoT</h1>'
                '<p>Click directly on household objects (TV, sofa, mirror, door, fan, AC, lights) to simulate real-world incidents and observe multi-sensor fusion.</p></div>',
                unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Layout: Room Viewer (70.1%) | Incident & Sensor Panel (29.9%)
    # -------------------------------------------------------------------------
    col_viewer, col_panel = st.columns([2.35, 1.0], gap="large")

    # ---- Handle incoming events from the Room Viewer component
    with col_viewer:
        event = render_room_viewer(
            current_view=s.room_view,
            lights_on=s.lights_on,
            visual_overlay=s.visual_overlay,
            key="room_viewer_custom"
        )

        if event and isinstance(event, dict):
            event_type = event.get("event")
            event_time = event.get("time", 0)

            if event_time > s.last_event_time:
                s.last_event_time = event_time

                if event_type == "change_view":
                    new_v = int(event.get("view", 1))
                    if new_v != s.room_view:
                        s.room_view = new_v
                        st.rerun()

                elif event_type == "trigger_action":
                    action_id = event.get("action")
                    obj_name = event.get("object")
                    view_id = int(event.get("view", s.room_view))
                    s.room_view = view_id
                    s.selected_object = obj_name

                    # Execute scenario configuration
                    sim_state = get_scenario_for_action(action_id, obj_name, view_id)
                    s.active_scenario = sim_state.get("scenario_name", action_id)
                    s.visual_overlay = sim_state.get("visual_overlay")

                    if sim_state.get("toggle_lights"):
                        s.lights_on = not s.lights_on
                        if not s.lights_on:
                            s.active_scenario = "Ceiling Lights: Switched Off"
                        else:
                            s.active_scenario = "Ceiling Lights: Switched On"
                    else:
                        for key in ("motion", "temp", "vib", "prev", "aud_mode", "aud_sample", "img_mode", "img_sample", "fire", "person", "active", "hold_prev", "sfx_file"):
                            if key in sim_state:
                                s[key] = sim_state[key]

                    st.rerun()

    # ---- Live stream drift if toggled
    if s.active and s.auto:
        s.motion = int(np.clip(s.motion + random.randint(-8, 8), 0, 100))
        s.vib = int(np.clip(s.vib + random.randint(-8, 8), 0, 100))
        s.temp = int(np.clip(s.temp + random.randint(-2, 2), -10, 100))
        s.vol = int(np.clip(s.vol + random.randint(-6, 6), 0, 100))

    # ---- Audio feature extraction: Load dedicated SFX file from assets/sfx
    vol, high, low = s.vol, s.high, s.low
    audio_data = None
    sfx_filename = s.get("sfx_file")
    if sfx_filename:
        sfx_path = os.path.join("assets", "sfx", sfx_filename)
        if os.path.exists(sfx_path):
            try:
                with open(sfx_path, "rb") as f:
                    audio_data = f.read()
                x, sr = load_audio(sfx_filename, audio_data)
                vol, high, low = analyse_audio(x, sr)
                s.vol, s.high, s.low = vol, high, low
            except Exception:
                pass

    if audio_data is None and s.aud_mode == "Sample clip":
        audio_data = sample_audio(s.aud_sample)
        x, sr = load_audio("sample.wav", audio_data)
        vol, high, low = analyse_audio(x, sr)

    # ---- Image feature extraction
    fire = s.fire
    person = s.person
    shown_frame = None
    if s.img_mode == "Sample scene":
        sample_frame = sample_image(s.img_sample)
        fire, hog_people, shown_frame, _ = analyse_image(sample_frame)
        person = bool(hog_people) or s.person

    # ---- Delta calculation for Sudden Fall / Impact rule
    delta = abs(s.motion - s.prev)

    # ---- Fusion Engine Evaluation
    sig = dict(
        motion=s.motion, temp=s.temp, vib=s.vib, delta=delta,
        fire=fire, person=person, volume=vol, high=high, low=low
    )
    rows = fuse(sig)
    confirmed = sorted([r for r in rows if r["state"] == "confirmed"], key=lambda r: (r["sev"] != "Critical", -r["conf"]))
    partial = sorted([r for r in rows if r["state"] == "partial"], key=lambda r: -r["conf"])

    # ---- Log event history if active
    if s.active:
        s.hist = (s.hist + [dict(Motion=s.motion, Temperature=s.temp, Vibration=s.vib, Volume=vol)])[-80:]
        names = [a["name"] for a in confirmed]
        for a in confirmed:
            if a["name"] not in s.last:
                s.log = ([dict(Time=datetime.now().strftime("%H:%M:%S"), Danger=a["name"], Severity=a["sev"],
                               Confidence=f"{a['conf']:.0f}%", Modalities=" + ".join(a["votes"]))] + s.log)[:30]
        s.last = names

    # -------------------------------------------------------------------------
    # Compact Incident and Sensor Information Panel (Right Column)
    # -------------------------------------------------------------------------
    with col_panel:
        # Context header
        view_meta = ROOM_VIEWS.get(s.room_view, {})
        st.markdown(
            f'<div class="panel-box">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span class="panel-badge">📷 {view_meta.get("label", f"View {s.room_view}")}</span>'
            f'<span class="panel-badge">💡 Lights: {"ON" if s.lights_on else "OFF"}</span>'
            f'</div>'
            f'<div style="font-size:0.84rem;color:var(--text-muted);">Target Object: <b style="color:var(--text-head);">{s.selected_object}</b></div>'
            f'<div style="font-size:0.84rem;color:var(--text-muted);margin-top:2px;">Scenario: <b style="color:#38bdf8;">{s.active_scenario}</b></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Controls & Reset
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if not s.active:
                st.button("▶ Monitor", type="primary", use_container_width=True, on_click=start_detection)
            else:
                st.button("⏹ Stop", use_container_width=True, on_click=lambda: s.update(active=False))
        with b_c2:
            st.button("↺ Reset", use_container_width=True, on_click=reset_scenario)

        # Multimodal Decision Banner
        if not s.active:
            st.markdown('<div class="status s-idle">○ System Idle. Click an object in the room to trigger an incident.</div>', unsafe_allow_html=True)
        else:
            if confirmed:
                st.markdown('<div class="status s-ok">🚨 Multimodal Alert Confirmed</div>', unsafe_allow_html=True)
            elif partial:
                st.markdown('<div class="status s-wait">⏳ Modality Voted — Awaiting Corroboration…</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status s-safe">✅ All Clear: Baseline Conditions</div>', unsafe_allow_html=True)

            for a in confirmed:
                st.markdown(alert_html(a), unsafe_allow_html=True)
            for a in partial:
                st.caption(f"Pending: **{a['name']}** | {', '.join(a['votes'])} voted (≥50%) | waiting for {', '.join(a['missing'])} ({a['conf']:.0f}%)")

        # Live Sensor Readings (Semicircle Gauges)
        st.markdown('<div class="section-heading">🌡️ Multi-Sensor Telemetry</div>', unsafe_allow_html=True)
        g_c1, g_c2, g_c3 = st.columns(3)
        g_c1.markdown(gauge("Motion", s.motion, 0, 100, "%", "⚡", "motion"), unsafe_allow_html=True)
        g_c2.markdown(gauge("Temp", s.temp, -10, 100, "°C", "🌡️", "temp"), unsafe_allow_html=True)
        g_c3.markdown(gauge("Vibration", s.vib, 0, 100, "%", "📳", "vib"), unsafe_allow_html=True)

        # Motion Delta Spike Card
        delta_alert = delta >= 50
        delta_col = "#ef4444" if delta_alert else "#10b981"
        delta_status = "CRITICAL SHOCK SPIKE" if delta_alert else "STABLE DYNAMICS"
        st.markdown(f"""
        <div class="panel-box" style="margin-top:6px;padding:8px 12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.78rem;">
            <span style="font-weight:700;color:var(--text-head);display:flex;align-items:center;gap:4px;">
              <span>💥</span> Motion Delta (Shock Acceleration)
            </span>
            <span style="font-weight:800;color:{delta_col};font-size:0.95rem;">Δ {delta}</span>
          </div>
          <div style="height:6px;background:var(--track-color,#1e293b);border-radius:6px;margin:5px 0;overflow:hidden;">
            <div style="width:{min(100, delta)}%;height:100%;background:{delta_col};transition:width 0.3s ease;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:var(--text-muted);">
            <span>Fall threshold: Δ 50</span>
            <span style="font-weight:700;color:{delta_col};">{delta_status}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio Modality Readings
        st.markdown('<div class="section-heading">🎙️ Acoustic Evidence (FFT &amp; RMS)</div>', unsafe_allow_html=True)
        a_c1, a_c2, a_c3 = st.columns(3)
        a_c1.markdown(f'<div class="sensor-stat-box"><div class="stat-val">{vol:.0f}</div><div class="stat-lbl">Volume (RMS)</div></div>', unsafe_allow_html=True)
        a_c2.markdown(f'<div class="sensor-stat-box"><div class="stat-val">{high:.0f}%</div><div class="stat-lbl">High Band (&gt;2kHz)</div></div>', unsafe_allow_html=True)
        a_c3.markdown(f'<div class="sensor-stat-box"><div class="stat-val">{low:.0f}%</div><div class="stat-lbl">Low Band (&lt;300Hz)</div></div>', unsafe_allow_html=True)

        # Visual Evidence Summary (Intrusion & Fire Vision Detection)
        st.markdown('<div class="section-heading">📷 Visual Evidence (Computer Vision)</div>', unsafe_allow_html=True)
        v_c1, v_c2 = st.columns(2)
        v_c1.markdown(f'<div class="sensor-stat-box"><div class="stat-val" style="color:{"#ef4444" if fire > 0.2 else "var(--text-head)"}">{fire:.1%}</div><div class="stat-lbl">Flame Pixel Ratio</div></div>', unsafe_allow_html=True)
        person_badge = '<span style="color:#ef4444;font-weight:800;">YES (DETECTED)</span>' if person else '<span style="color:#10b981;font-weight:700;">NO (CLEAR)</span>'
        v_c2.markdown(f'<div class="sensor-stat-box"><div class="stat-val" style="font-size:1.05rem;">{person_badge}</div><div class="stat-lbl">AI Person Detector</div></div>', unsafe_allow_html=True)

        if shown_frame is not None and (person or fire > 0.15):
            st.image(shown_frame, caption=f"AI Vision Stream ({s.img_sample})", use_container_width=True)

    # -------------------------------------------------------------------------
    # Collapsible Technical Diagnostics (Below main viewer, maintaining room focus)
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔍 Multimodal Late Fusion Rule Matrix & Evidence Scores", expanded=False):
        st.caption("Evidence score (0-100) per modality for each late-fusion rule. A score ≥ 50 constitutes an active modality vote. A rule fires only when all required modalities agree.")
        df_rules = pd.DataFrame([
            {
                "Danger Incident": r["name"],
                "Severity": r["sev"],
                "Sensor Evidence": r["sc"].get("Sensor"),
                "Image Evidence": r["sc"].get("Image"),
                "Audio Evidence": r["sc"].get("Audio"),
                "Confidence": r["conf"],
                "Decision Status": "CONFIRMED" if r["state"] == "confirmed" else
                                  f"Waiting for {', '.join(r['missing'])}" if r["state"] == "partial" else "Idle"
            } for r in rows
        ])
        pc = lambda lbl: st.column_config.ProgressColumn(lbl, min_value=0, max_value=100, format="%d")
        st.dataframe(
            df_rules,
            hide_index=True,
            column_config={
                "Sensor Evidence": pc("Sensor Evidence"),
                "Image Evidence": pc("Image Evidence"),
                "Audio Evidence": pc("Audio Evidence"),
                "Confidence": pc("Confidence")
            },
            use_container_width=True
        )

    with st.expander("📈 Live Multi-Sensor Telemetry History", expanded=True):
        render_telemetry_chart(s.hist)

    with st.expander("🧾 Security & Incident Event Log", expanded=False):
        if s.log:
            st.dataframe(pd.DataFrame(s.log), hide_index=True, use_container_width=True)
        else:
            st.info("No confirmed danger incidents logged yet.")

    with st.expander("⚙️ Manual Signal Adjustments & Presets", expanded=False):
        p_c1, p_c2, p_c3 = st.columns([1.5, 1, 1.2])
        p_c1.selectbox("Standard Preset Scenarios", list(PRESETS), key="preset")
        p_c2.button("Apply Preset", on_click=apply_preset)
        p_c3.toggle("Simulated Drift (Live stream)", key="auto", help="Random walk drift every 1.5 seconds.")

        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.slider("Manual Motion Sensor", 0, 100, key="motion")
        with s_c2:
            st.slider("Manual Temperature (°C)", -10, 100, key="temp")
        with s_c3:
            st.slider("Manual Vibration Sensor", 0, 100, key="vib")

    # Update previous motion for sudden impact calculation
    if not s.pop("hold_prev", False):
        s.prev = s.motion

    if s.active and s.auto:
        time.sleep(1.5)
        st.rerun()


# =============================================================================
# 7. NAVIGATION & SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 📡 Multimodal IoT")
    st.caption("Interactive Multi-View Smart Home Simulation")
    st.radio("Navigate", ["🚨 Live Demo", "📘 Theory (6 slides)"], key="page", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🏠 Room Camera Views")
    st.markdown("- **View 1**: TV Wall & Entrance\n- **View 2**: Window Side & Sofa\n- **View 3**: Sofa & Full Mirror\n- **View 4**: Entrance, Shelf & Sofa")
    st.markdown("---")
    st.caption("Multimodal Sensor + Computer Vision + Acoustic FFT Fusion.\nBuilt with Streamlit · OpenCV · NumPy · Pillow")

if s.page.startswith("📘"):
    page_theory()
else:
    page_demo()