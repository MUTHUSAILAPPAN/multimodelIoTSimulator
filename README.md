# 🏠 Multimodal IoT — Smart Home Security & Simulation

An interactive, explainable **Multimodal IoT (Internet of Things)** simulation platform that fuses **sensor telemetry**, **computer vision**, and **acoustic frequency analysis** to detect residential emergencies with high confidence and minimal false alarms.

---

## 🌟 Highlights

- **Interactive Multi-View House Simulator**: A high-performance bi-directional custom Streamlit component rendering 4 camera viewpoints with normalized interactive hotspots on household items.
- **Explainable Late Fusion Architecture**: Modality evidence scoring ($0\text{–}100$) using linear ramps; rules require quorum consensus across independent physical channels before confirming alerts.
- **Real-Time Visual & Acoustic Effects**:
  - Live AI Computer Vision bounding boxes and HSV flame ratio overlays.
  - Native spatial SFX playback with synchronized FFT spectrum and RMS/peak audio analysis.
  - Realistic intruder perspective placement and lights-off nocturnal mode.
- **Interactive Multi-Sensor Telemetry**: Real-time Altair charts tracking Motion, Temperature, Vibration, and Volume with discrete sequence history.
- **Comprehensive Theory Slide Deck**: 6 structured interactive slides detailing the foundations, comparative trade-offs, and architecture of multimodal IoT.

---

## 🏗️ System Architecture

The pipeline processes raw physical phenomena through a 4-layer flow:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         LAYER 1: PERCEPTION                            │
│  [Sensors (Temp/Motion/Vib)]   [Camera / Vision]   [Microphone / Audio]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      LAYER 2: FEATURE EXTRACTION                       │
│  • Motion Delta & Temp (°C)   • HSV Flame Ratio    • Volume RMS & Peak │
│  • Structural Vibration (%)   • HOG / AI Intruder  • FFT Band Ratio    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    LAYER 3: MULTIMODAL LATE FUSION                     │
│  • Linear Evidence Ramps (0 to 100 per modality)                       │
│  • Modality Vote Threshold: Evidence ≥ 50%                             │
│  • Rule Confirmation: All required modalities must vote                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       LAYER 4: EXPLAINABLE RESPONSE                    │
│  • Confirmed Critical/Warning Alerts with Triggering Modalities        │
│  • Live Telemetry Time-Series Chart & Security Incident Log            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚨 The 7 Late-Fusion Threat Rules

| Threat Incident | Severity | Required Modalities | Physical Corroboration Logic |
| :--- | :---: | :--- | :--- |
| **Fire Incident** | `CRITICAL` | `Image` + `Sensor` | Flame pixel ratio $> 5\%$ AND Temperature $> 55^\circ\text{C}$ |
| **Glass Break** | `CRITICAL` | `Audio` + `Sensor` | Acoustic high-frequency burst ($> 2\text{ kHz}$) AND Vibration spike |
| **Intrusion / Person** | `CRITICAL` | `Image` + `Sensor` | Vision bounding box detection AND Active motion acceleration |
| **Equipment Fault** | `WARNING` | `Sensor` + `Audio` | Continuous vibration spike ($> 50\%$) AND Elevated audio hum |
| **Extreme Temperature** | `WARNING` | `Sensor` | Ambient HVAC temperature $> 70^\circ\text{C}$ or $< 5^\circ\text{C}$ |
| **Fall / Sudden Impact** | `CRITICAL` | `Sensor` + `Audio` | Motion acceleration spike ($\Delta \ge 50$) AND Low-frequency thump ($< 300\text{ Hz}$) |
| **Unauthorized Loud Noise** | `WARNING` | `Audio` + `Sensor` | Acoustic volume burst ($> 60\text{ dB}$) AND Active occupant motion |

---

## 📷 Room Camera Views & Interactive Hotspots

| View | Perspective | Monitored Objects & Hotspots |
| :---: | :--- | :--- |
| **View 1** | **TV Wall & Entrance** | Smart TV, Sofa, Glass Coffee Table, Ceiling Fan, AC, Ceiling Lights, Entrance Door, CCTV Dome, Window |
| **View 2** | **Window Side & Balcony** | Balcony Window, Sofa, Glass Table, Wall Shelf, AC, Entrance Door, CCTV Camera, Ceiling Fan, Lights |
| **View 3** | **Sofa & Full Mirror** | Full-Length Mirror, Living Room Sofa, AC, TV Wall, Entrance Door, Ceiling Fan, Ceiling Lights |
| **View 4** | **Entrance & Shelf** | Main Entrance Door, Decorative Shelf, Smart TV, Sofa, Glass Table, Ceiling Lights *(supports Day & Lights-Off view)* |

---

## 📁 Repository Structure

```
multimodel iot/
├── app.py                      # Main Streamlit application, UI layout, fusion engine & slides
├── room_config.py              # Camera viewpoints, normalized hotspot coordinates & action registry
├── scenarios.py                # Action-to-telemetry simulation mapping engine
├── room_viewer.py              # Streamlit bi-directional custom component wrapper
├── room_component/             # Frontend custom component (iframe)
│   ├── index.html              # Interactive room canvas, hotspots, overlays, and audio triggers
│   ├── assets/                 # Camera view photography and intruder cutout textures
│   └── sfx/                    # Native sound effects triggered on user actions
├── assets/                     # Backend assets for Python analysis
│   ├── Room View 1–4.png       # High-resolution photographic room captures
│   ├── Room View 4 lights off.png
│   ├── intruder.png            # AI vision intruder asset
│   └── sfx/                    # Reference audio WAV files analyzed by NumPy FFT engine
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ (Python 3.10–3.12 recommended)
- Git

### 2. Installation

Clone or navigate to the repository folder:

```bash
cd "d:\AppProject\multimodel iot"
```

Install the required Python libraries:

```bash
pip install streamlit opencv-python-headless numpy pandas Pillow altair
```

*(Optional for uploading third-party MP3 files)*:
```bash
pip install pydub
```

### 3. Running the Application

Launch the Streamlit web server:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🎮 How to Use the Simulation

1. **Navigate Perspectives**: Use the `◀ Previous View` / `Next View ▶` buttons or click the camera tabs (View 1 to 4) to inspect different room angles.
2. **Interact with Objects**: Click directly on any glowing hotspot (e.g., Television, Door, Coffee Table, AC, Fan, Lights) to open its action dialog.
3. **Simulate an Incident**:
   - Click **Simulate Fire Incident** on the TV to observe the flame mask and thermal spike.
   - Click **Simulate Forced Entry** or **Inspect Intruder (CCTV)** on the door to see the live intruder overlay and AI vision bounding box.
   - Click **Simulate Sudden Impact / Fall** on the coffee table or sofa to observe the motion delta shockwave and acoustic thump.
4. **Inspect Fusion Evidence**: Expand the *Multimodal Late Fusion Rule Matrix* below the room view to verify evidence scores across Sensor, Image, and Audio modalities.
5. **Explore Theory**: Switch to the **📘 Theory (6 slides)** page in the left sidebar to review multimodal IoT system design and architectural principles.

---

## 🛡️ License

Developed for Multimodal IoT and Smart Computing research and demonstration. All room imagery and simulation audio assets are packaged for educational and demonstrative use.
