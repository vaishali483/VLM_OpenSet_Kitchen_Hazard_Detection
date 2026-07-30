# Open-Set Hazard Detection in the Kitchen: A VLM-Grounded LLM Agent vs. Rule-Based Systems

## Overview

This repository contains the implementation developed for my Master's dissertation on **open-set kitchen hazard detection**.

The project investigates whether a Vision-Language-Model-grounded reasoning system can identify kitchen hazards that are not limited to a predefined rule list. The system combines object detections from **YOLO11s** with visual reasoning from **Google Gemini** and compares the resulting open-set hazard detector with a conventional rule-based baseline.

A companion **Unity AR mobile application** captures kitchen images, sends them to a FastAPI backend, and displays concise safety warnings containing:

- hazard name and status
- severity
- explanation
- recommended action
- a validated visual marker when localisation is reliable

The project is focused on **hazard detection**. Action recognition and short-term cooking-session context are used only as supporting evidence for classifying hazards as present or possible.

---

## Project Scope

The current system supports two complementary workflows.

#### 1. Offline dissertation evaluation

A curated subset of EPIC-KITCHENS clips is processed through an evaluation pipeline that compares:

- an open-set VLM-grounded hazard-analysis agent; and
- a fixed rule-based kitchen-safety baseline.

#### 2. Live Unity AR prototype

The Unity application periodically captures a camera frame and sends it to the backend for:

1. YOLO11s object detection
2. Gemini scene understanding and open-set hazard reasoning
3. present/possible hazard classification
4. conservative hazard localisation
5. display of hazard cards and visual markers in Unity

The live application analyses sampled frames from a cooking session or video. It does not record or analyse a continuous video.

---

## Research Objectives

The project aims to:

- detect open-set kitchen hazards beyond a fixed predefined checklist
- use YOLO detections as supporting visual evidence
- use a VLM to reason about object relationships and scene context
- compare the VLM-grounded approach with a rule-based baseline
- classify detected risks as "present" or "possible"
- avoid displaying misleading bounding boxes when localisation is uncertain
- provide concise warnings suitable for a mobile AR interface
- maintain lightweight textual context during one cooking session
- produce a deduplicated session-end hazard summary

---

## System Architecture

### Offline Evaluation Pipeline

```text
EPIC-KITCHENS video clips
          │
          ▼
01_extract_frames
          │
          ▼
02_detect_objects
          │
          ▼
03_build_scene_json
          │
          ▼
04_select_keyframes
          │
          ▼
05_scene_understanding
          │
          ▼
06_hazard_analysis
          │
          ├──────────────► Open-set VLM hazard results
          │
          ▼
07_ground_truth_templates
          │
          ▼
08_rule_based_baseline
          │
          ▼
09_match_templates
          │
          ▼
10_evaluate

```

### Live AR Pipeline

```text
Unity AR camera frame
          │
          ▼
FastAPI image endpoint
          │
          ▼
YOLO11s object detection
          │
          ▼
Single structured Gemini request
          │
          ├── Scene understanding
          ├── Current action observation
          ├── Open-set hazard analysis
          ├── Present/possible classification
          └── Localisation proposal
          │
          ▼
Backend validation
          │
          ├── Exact YOLO-confirmed box
          ├── VLM approximate region
          └── Warning only when location is uncertain
          │
          ▼
Unity mobile interface
          ├── Analysed-frame preview
          ├── Hazard cards
          ├── Severity and action
          └── Validated visual marker
```

---

## Cooking Session Mode

The live application supports one sampled-frame cooking session with:

- **Start Session**
- **Pause/Resume Session**
- **Stop Session**

During a session:

1. Unity captures one frame periodically.
2. The current frame is sent to the backend.
3. YOLO11s and Gemini analyse the current scene.
4. The backend stores the sampled frame and its structured result.
5. A compact text summary of recent observations is used as optional context for later frames.
6. The backend updates a deduplicated session hazard history.

When the session is stopped, Unity displays a names-only summary grouped into:

```text
PRESENT HAZARDS
• Knife near counter edge
• Plastic packaging near active flame

POSSIBLE HAZARDS
• Towel near unused stovetop
• Surface may retain heat
```

The final list is deduplicated and ordered by the highest severity observed.

---

## Hazard Status Definitions

### Present Hazard

A hazard is classified as `present` when the current visual evidence supports an immediate hazardous condition.

Examples:

- a knife visibly close to the counter edge
- plastic packaging beside a visible active flame
- a hand visibly close to a sharp blade
- a visible liquid spill on the floor

### Possible Hazard

A hazard is classified as `possible` when:

- a visible arrangement may become dangerous after a clear trigger
- an earlier observation suggests a residual risk that cannot currently be confirmed
- appliance state, heat, flame, or another physical condition is uncertain

Examples:

- a plastic bag lying on an unused or uncertain stovetop;
- a stovetop that was previously used but whose current temperature cannot be measured.

The system does not claim that a user intends to activate an appliance, and it does not claim that a surface is hot without sufficient evidence.

---

## Hazard Localisation

Hazard recognition and hazard localisation are treated as separate tasks.

### Exact Bounding Box

A solid bounding box is shown only when:

- YOLO detected the correct object
- Gemini selected the matching detection ID and class
- detector and VLM confidence pass validation
- the box is not too broad or too small
- the hazard text refers to the same object class

### Approximate Region

An approximate marker may be shown when:

- the hazardous area is clearly visible
- no YOLO box tightly encloses the exact hazard
- Gemini provides a sufficiently confident normalised region
- the region passes backend size and validity checks

### Warning Only

No box is shown when localisation is uncertain.

This is intentional. A missing box is safer than a misleading box around the wrong object or a broad person, table, or oven region.

---

## Current Evaluation Snapshot

The current offline evaluation uses 11 clips from participant P07.

### Open-Set VLM Agent

| Metric | Result |
|---|---:|
| True Positives | 11 |
| False Positives | 1 |
| False Negatives | 2 |
| Precision | 0.9167 |
| Recall | 0.8462 |
| F1 Score | 0.8800 |
| Novel Hazard Detection Rate | 0.8462 |
| Clip-Level False Positive Rate | 0.0000 |

### Rule-Based Baseline

| Metric | Result |
|---|---:|
| True Positives | 0 |
| False Positives | 6 |
| False Negatives | 13 |

These values represent the current experimental snapshot.

---

## Dataset

The offline evaluation uses a curated subset of the **EPIC-KITCHENS-100** egocentric cooking dataset.

A broader project subset of approximately 210 cooking clips was prepared to represent activities such as: chopping, cutting and slicing, peeling, stirring, pouring, opening cupboards and refrigerators, switching appliances on or off, handling knives and utensils, placing or moving kitchen objects.

The dataset is not included in this repository because of licensing and storage restrictions.

---

## Project Structure

```text
KitchenSafetyAssistant/
│
├── backend/
│   └── app/
│       ├── core/
│       │   ├── prompt_loader.py
│       │   └── settings.py
│       │
│       ├── routes/
│       │   ├── live_analysis.py
│       │   ├── sessions.py
│       │   └── test_analysis.py
│       │
│       ├── schemas/
│       │   ├── hazard.py
│       │   ├── live_analysis.py
│       │   └── session.py
│       │
│       ├── services/
│       │   ├── live_analysis_service.py
│       │   ├── session_service.py
│       │   └── yolo_service.py
│       │
│       └── main.py
│
├── pipeline/
│   ├── 01_extract_frames.py
│   ├── 02_detect_objects.py
│   ├── 03_build_scene_json.py
│   ├── 04_select_keyframes.py
│   ├── 05_scene_understanding.py
│   ├── 06_hazard_analysis.py
│   ├── 07_ground_truth_templates.py
│   ├── 08_rule_based_baseline.py
│   ├── 09_match_templates.py
│   └── 10_evaluate.py
│
├── prompts/
│   └── live_hazard_analysis.txt
│
├── models/
│   └── yolo11s.pt
│
├── data/
│   └── sessions/
│
├── dataset/
│   ├── annotations/
│   ├── demo/
│   └── ...
│
├── scripts/
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

The Unity client is maintained as a separate Unity project:

```text
KitchenSafetyAR/
└── Assets/
    └── _KitchenSafety/
        └── Scripts/
            ├── Capture/
            ├── Models/
            ├── Networking/
            ├── Scanning/
            ├── Sessions/
            └── UI/
```

---

## Backend API

The FastAPI backend exposes the following endpoints.



### One-Off Live Analysis

```http
POST /api/v1/live/analyse-image
```


### Cooking Sessions

```http
POST /api/v1/sessions/start
POST /api/v1/sessions/{session_id}/pause
POST /api/v1/sessions/{session_id}/resume
POST /api/v1/sessions/{session_id}/analyse-image
POST /api/v1/sessions/{session_id}/stop
GET  /api/v1/sessions/{session_id}
```

---

## Unity AR Client

The Unity application was developed using Unity 6.3 LTS and AR Foundation.

### Main Interface Components

- analysed-frame preview
- one-off analyse control
- cooking-session controls (Start, Pasue/Resume, Stop session)
- scrollable hazard cards
- validated exact or approximate visual markers
- full-screen session-end summary


### Android Backend Address

When testing on an Android phone, use the development computer's local network IP, for example:

```text
http://192.168.x.x:8000
```

The computer and Android device must be on the same network, and the firewall must allow the selected port.

---

## Session Data and Privacy

The application does not save or process a continuous cooking video.

For each session, it may store:

- sampled JPEG images
- capture IDs
- UTC timestamps
- recognised actions
- scene summaries and observations
- hazards
- severity
- confidence
- present/possible status
- first and last observation times
- occurrence counts

---

## Technologies Used

### Backend and Evaluation

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- Ultralytics YOLO11s
- Google Gemini Flash models
- Google Gen AI Python SDK
- OpenCV
- Pillow
- Pandas
- NumPy
- FFmpeg

### Mobile AR Client

- Unity 6.3 LTS
- C#
- AR Foundation
- TextMeshPro
- Unity UI
- UnityWebRequest

### Development

- Visual Studio Code
- Swagger / OpenAPI
- Android development tools
- Git and GitHub

---

## Known Limitations

- Gemini response latency varies and may occasionally be high.
- Gemini may temporarily return `503 UNAVAILABLE` during periods of high demand.
- YOLO11s uses generic object classes and may miss kitchen-specific objects or object parts.
- Small hazards such as flames, hands, spills, and pan handles may not receive exact detector boxes.
- Approximate VLM localisation is intentionally labelled as approximate.
- Some correctly detected hazards are displayed without boxes when localisation is uncertain.
- Sampled-frame analysis can miss brief events occurring between captures.
- The VLM cannot directly measure heat, temperature, gas, or appliance electrical state.
- Session context is observational and must not be treated as proof of user intention.

---

## Author

**Vaishali Veera Saravana Perumal**


MSc Artificial Intelligence and Machine Learning

University of Birmingham
