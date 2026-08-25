# Open-Set Hazard Detection in the Kitchen: A VLM-Grounded LLM Agent vs. Rule-Based Systems

## Overview

This repository contains the implementation developed for my Master's dissertation on **open-set kitchen hazard detection**.

The project investigates whether a Vision-Language-Model-grounded reasoning system can identify kitchen hazards that are not limited to a predefined rule list. The system combines object detections from **YOLO11s** with visual reasoning from **Google Gemini** and compares the resulting open-set hazard detector with a conventional fixed rule-based baseline.

A companion **Unity AR mobile application** captures live kitchen images, sends them to a FastAPI backend, and displays concise safety warnings containing:

- hazard name and status
- severity
- explanation
- recommended action
- a validated visual marker when localisation is reliable

The project is focused on **hazard detection**. Action recognition and short-term cooking-session context are used only as supporting evidence for understanding the current scene and distinguishing between present and possible hazards.

---

## Project Scope

The current system supports two complementary workflows.

### 1. Offline Evaluation Pipeline

A curated subset of EPIC-KITCHENS clips is processed through an offline evaluation pipeline that compares:

- an **open-set VLM-grounded hazard-analysis agent**
- a **fixed closed-set rule-based kitchen safety baseline**

The final evaluation workflow uses manually written ground-truth hazards and manually reviewed semantic matches between predictions and ground truth.

### 2. Live Unity AR Prototype

The Unity application periodically captures a camera frame and sends it to the FastAPI backend for:

1. YOLO11s object detection
2. Gemini scene understanding and open-set hazard reasoning
3. present/possible hazard classification
4. conservative hazard localisation
5. display of hazard cards and visual markers in Unity
6. optional short-term cooking-session context and session-level hazard tracking

The live application analyses sampled frames rather than recording or processing continuous video.

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
07_create_ground_truth_templates
          │
          ▼
Manual ground-truth annotation
          │
          ▼
08_rule_based_baseline
          │
          ├──────────────► Fixed rule-based hazard results
          │
          ▼
09_create_match_templates
          │
          ▼
Manual prediction ↔ ground-truth matching
          │
          ▼
10_evaluate
          │
          ▼
Evaluation summaries and comparison metrics
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
          ├── Exact YOLO-grounded box
          ├── VLM approximate region
          └── Warning only when location is uncertain
          │
          ▼
Unity mobile interface
          ├── Analysed-frame preview
          ├── Hazard cards
          ├── Severity and recommended action
          ├── Validated visual marker
          └── Session controls and summary
```

---

## Cooking Session Mode

The live application supports sampled-frame cooking sessions with:

- **Start Session**
- **Pause Session / Resume Session**
- **Stop Session**

During a session:

1. Unity periodically captures one camera frame.
2. The current frame is sent to the backend.
3. YOLO11s and Gemini analyse the current scene.
4. The backend stores the sampled frame and its structured analysis result locally.
5. A compact textual summary of recent successful analyses is used as supporting context for later frames.
6. Previous sampled images are not sent back to Gemini.
7. The backend maintains a deduplicated session-level hazard history.

The current image remains the primary source of evidence. Session context is supporting evidence only and does not override contradictory evidence in the current image.

When the session is stopped, Unity displays a names-only summary grouped into:

```text
PRESENT HAZARDS
• Knife near counter edge
• Plastic packaging near active flame

POSSIBLE HAZARDS
• Towel near unused stovetop
• Surface may retain heat
```

The final hazard list is deduplicated and ordered according to the highest severity observed during the session.

---
## Hazard Status and Localisation
### Hazard Status Definitions

#### Present Hazard

A hazard is classified as `present` when the current visual evidence supports a currently hazardous condition.

Examples include:

- a knife visibly close to the edge of a work surface
- plastic packaging beside a visible active flame
- a hand visibly close to a sharp blade
- a visible liquid spill in a walking area

#### Possible Hazard

A hazard is classified as `possible` when:

- a visible arrangement could become hazardous after a clear trigger
- an earlier observation leaves a residual risk that cannot currently be confirmed
- appliance state, heat, flame or another physical condition cannot be directly established from the available evidence

Examples include:

- a plastic bag lying on an unused or uncertain stovetop
- a previously used cooking surface whose current temperature cannot be measured

The system does not claim that a user intends to activate an appliance, and it does not claim that a surface is hot without sufficient evidence.

### Hazard Localisation

Hazard recognition and hazard localisation are treated as separate tasks.

#### Exact Bounding Box

A solid bounding box is shown only when:

- YOLO detected the relevant object
- Gemini selected the matching YOLO detection
- the selected detection ID and object class are valid
- detector and VLM localisation confidence pass validation
- the box is not excessively broad or too small
- the hazard description refers to the selected object class

#### Approximate Region

An approximate visual region may be shown when:

- the hazardous area is visibly identifiable
- no YOLO detection tightly represents the exact hazard
- Gemini provides a sufficiently confident normalised region
- the proposed region passes backend size and validity checks

Approximate regions are visually distinguished from exact detector bounding boxes.

#### Warning Only

No visual marker is shown when localisation is not sufficiently reliable.

This is intentional. A missing box is considered safer than displaying a misleading box around the wrong object or a broad person, table or appliance region.

---

## Evaluation

The final evaluation was conducted on a manually reviewed **32-clip test subset**.

The test set contains:

- 32 clips
- 28 clips containing at least one ground-truth hazard
- 4 safe clips
- 42 manually annotated ground-truth hazards
- 40 hazards labelled as novel relative to the fixed rule-based baseline
- 2 hazards represented by the predefined baseline rule set

For each clip, VLM and rule-based predictions were manually matched to the ground truth using one-to-one semantic matching. A prediction was considered a match only when it described the same underlying hazardous condition as the corresponding ground-truth hazard.

The frozen rule-based evaluation ruleset is stored at:

```text
rules/kitchen_rules_evaluation.json
```

### Final Test-Set Results

| Metric                         | VLM Open-Set Agent | Rule-Based Baseline |
| ------------------------------ | -----------------: | ------------------: |
| True Positives                 |             **34** |                   1 |
| False Positives                |             **12** |                  36 |
| False Negatives                |              **8** |                  41 |
| Precision                      |         **0.7391** |              0.0270 |
| Recall                         |         **0.8095** |              0.0238 |
| F1 Score                       |         **0.7727** |              0.0253 |
| Hazardous Clip Detection Rate  | **1.0000 (28/28)** |       0.0357 (1/28) |
| Clip-Level False Positive Rate |       0.5000 (2/4) |    **0.2500 (1/4)** |
| Mean False Warnings per Clip   |         **0.3750** |              1.1250 |
| Novel Hazard Detection Rate    | **0.8250 (33/40)** |       0.0000 (0/40) |
| Known Hazard Recall            |       0.5000 (1/2) |        0.5000 (1/2) |
| High/Critical Hazard Recall    | **0.8462 (11/13)** |       0.0000 (0/13) |
| Medium/Low Hazard Recall       | **0.7931 (23/29)** |       0.0345 (1/29) |


### Evaluation Summary

The VLM-grounded system detected at least one correct hazard in all 28 hazardous clips and identified 33 of the 40 hazards labelled novel relative to the frozen baseline. The baseline detected none of the novel hazards. Although the baseline achieved a lower false-positive rate on the four safe clips, this occurred alongside substantially lower overall hazard recall.

The complete evaluation outputs are available under:

```text
evaluation/results/test/
├── evaluation_summary.json
├── per_clip_summary.csv
└── system_comparison.csv
```

---

## Rule-Based Baseline

The final rule-based baseline is intentionally restricted to deterministic rules that can be evaluated using the available YOLO-derived scene representation.

The baseline uses:

- detected object classes
- YOLO confidence scores
- bounding boxes
- image dimensions
- object proximity
- object overlap
- detected work-surface relationships

It does not use Gemini or any other VLM reasoning.

The frozen evaluation ruleset contains 15 predefined kitchen-safety rules and is stored at:

```text
rules/kitchen_rules_evaluation.json
```

The ruleset is frozen before final evaluation to avoid changing the baseline after observing test-set results.

---

## Dataset

The offline pipeline uses a curated subset of the **EPIC-KITCHENS-100** egocentric cooking dataset.

A project subset of cooking clips was prepared to represent activities such as chopping, cutting and slicing, peeling, stirring, pouring, opening cupboards and refrigerators, switching appliances on or off, handling knives and utensils, and placing or moving kitchen objects.

A separate 32-clip subset is used for the final manual evaluation.

The original EPIC-KITCHENS videos are not included in this repository because of licensing and storage restrictions.

---

## Project Structure

```text
KitchenSafetyAssistant/
│
├── backend/
│   ├── requirements.txt
│   │
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── prompt_loader.py
│       │   └── settings.py
│       │
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── live_analysis.py
│       │   ├── sessions.py
│       │   └── test_analysis.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── hazard.py
│       │   ├── live_analysis.py
│       │   └── session.py
│       │
│       └── services/
│           ├── __init__.py
│           ├── live_analysis_service.py
│           ├── session_service.py
│           └── yolo_service.py
│
├── data/
│   └── sessions/                         # local, not committed
│
├── pipeline/
│   ├── 01_extract_frames.py
│   ├── 02_detect_objects.py
│   ├── 03_build_scene_json.py
│   ├── 04_select_keyframes.py
│   ├── 05_scene_understanding.py
│   ├── 06_hazard_analysis.py
│   ├── 07_create_ground_truth_templates.py
│   ├── 08_rule_based_baseline.py
│   ├── 09_create_match_templates.py
│   ├── 10_evaluate.py
│   └── config.py
│
├── prompts/                              # local prompt files
│
├── dataset/
│   ├── frames/                           # not committed
│   ├── detections/                       # not committed
│   ├── annotated_frames/                 # not committed
│   ├── scene_json/                       # not committed
│   ├── keyframes/
│   ├── scene_understanding/
│   ├── hazard_analysis/
│   ├── ground_truth/
│   └── baseline_hazards/
│
├── evaluation/
│   ├── matching/
│   │   └── test/
│   │
│   └── results/
│       └── test/
│
├── models/                               # local, not committed
│   └── yolo11s.pt
│
├── rules/
│   └── kitchen_rules_evaluation.json
│
├── .env                                  # local, not committed
├── .gitignore
└── README.md
```

---

## Backend API

The FastAPI backend provides a one-off live-analysis endpoint and a set of cooking-session endpoints.

### One-Off Live Analysis

```http
POST /api/v1/live/analyse-image
```

This endpoint performs a single independent image analysis without creating or updating a cooking session.

### Cooking Sessions

```http
POST /api/v1/sessions/start
POST /api/v1/sessions/{session_id}/pause
POST /api/v1/sessions/{session_id}/resume
POST /api/v1/sessions/{session_id}/analyse-image
POST /api/v1/sessions/{session_id}/stop
GET  /api/v1/sessions/{session_id}
```

Session analysis uses the same core live-analysis pipeline while additionally maintaining short-term textual context and session-level hazard aggregation.

---

## Unity AR Client

The Unity application was developed using **Unity 6.3 LTS**, AR Foundation and C#.

The Unity project is maintained separately from the Python backend.

### Main Interface Components

The current application provides:

- one-off **Analyse Now** control
- analysed-frame preview
- scrollable multi-hazard results
- present/possible hazard status
- severity and recommended action
- exact or approximate hazard localisation
- **Session controls** (Start, Pause/Resume, Stop)
- full-screen session-end hazard summary

**Analysed Frame Preview:** The visual hazard markers are displayed against the last analysed frame rather than being left over the continuously changing camera feed. This keeps each bounding box or approximate region spatially associated with the image that actually produced the corresponding hazard result.

**Periodic Analysis**: During an active cooking session, frames are sampled periodically. Only one backend analysis request is allowed to run at a time, preventing overlapping requests when Gemini response latency is high. Warnings remain visible while the next sampled frame is being processed instead of immediately disappearing.

#### Android Backend Address

When testing on a physical Android device, the development computer's local network IP must be used rather than `127.0.0.1`.

For example:

```text
http://192.168.x.x:8000
```

The computer and Android device must be connected to the same network, and the required backend port must be allowed through the computer's firewall.

---

## Unity AR Prototype Results

The Android Unity prototype was tested in live kitchen scenarios to verify the complete camera-to-backend-to-AR workflow.

During an active cooking session, the application periodically captures a frame, sends it to the FastAPI backend, receives the structured hazard-analysis result, and displays the warning together with the analysed-frame preview.

The examples below demonstrate exact YOLO-grounded localisation, approximate VLM localisation, present/possible hazard classification, live session controls, and the final cooking-session summary.

### Exact Object Localisation

When the hazardous object can be reliably associated with a YOLO detection, the application displays an exact bounding box on the analysed frame.

<table>
<tr>
<td width="50%" align="center">

<img src="docs/images/unity/exact_knife_localisation.jpeg" width="300"/>

**Exact knife localisation**

A detected knife is associated with the hazard and displayed using the validated YOLO bounding box.

</td>
<td width="50%" align="center">

<img src="docs/images/unity/exact_knife_session.jpeg" width="300"/>

**Exact localisation during an active session**

The hazard card displays the hazard status, severity, explanation and recommended action while the associated knife is highlighted in the analysed-frame preview.

</td>
</tr>
</table>

### Present and Possible Hazard Classification

The interface distinguishes between hazards that are currently supported by the visual evidence and arrangements that represent a possible risk.

<table>
<tr>
<td width="50%" align="center">

<img src="docs/images/unity/possible_pan_handle.jpeg" width="300"/>

**Possible hazard**

An outward-facing pan handle is classified as a possible hazard and includes its trigger condition in the warning card.

</td>
<td width="50%" align="center">

<img src="docs/images/unity/approximate_hand_hazard.jpeg" width="300"/>

**Present hazard with approximate localisation**

A bare-hand interaction with hot food is classified as present. Because the exact hazardous region is not represented by a reliable YOLO object box, an approximate VLM-generated region is displayed.

</td>
</tr>
</table>

### Approximate VLM Localisation

The system can provide an approximate visual region when a hazard is clearly visible but cannot be reliably represented by an exact YOLO bounding box.

<p align="center">
<img src="docs/images/unity/approximate_plastic_bag.jpeg" width="300"/>
</p>

<p align="center">
<b>Approximate plastic-bag hazard region</b>
</p>

The approximate marker is visually labelled as `APPROX. AREA` so that it is not presented as an exact detector bounding box.

### Cooking Session Summary

When the user stops a cooking session, the application aggregates the hazards observed across the analysed frames, removes repeated entries, and displays separate present and possible hazard lists.

<p align="center">
<img src="docs/images/unity/session_summary.jpeg" width="300"/>
</p>

<p align="center">
<b>Cooking-session summary after 13 analysed frames</b>
</p>

The summary provides a concise record of hazards observed throughout the session rather than displaying every individual frame-level warning.

--- 
## Session Data and Privacy

The live application analyses sampled frames rather than recording continuous video. During a cooking session, sampled JPEG frames and structured session metadata are stored locally under:

```text
data/sessions/<session_id>/
├── session.json
└── frames/
```

Only the current sampled image is sent to Gemini for each analysis. Previous session images are not resent; recent observations may instead be supplied as compact textual context.

---

## Technologies Used

### Backend and Evaluation

- Python 3.12
- FastAPI
- Ultralytics YOLO11s
- Google Gemini Flash models
- OpenCV

### Mobile AR Client

- Unity 6.3 LTS
- AR Foundation

---

## Known Limitations

- Gemini API latency is variable, so the prototype uses sampled-frame near-real-time analysis rather than continuous video processing.
- YOLO11s uses generic COCO classes and may miss kitchen-specific objects, small hazards or object parts.
- Some hazards cannot be reliably localised and are therefore shown using an approximate region or warning-only output.
- Vision alone cannot directly verify physical conditions such as heat, temperature, gas concentration or appliance electrical state.
- The final evaluation is limited to 32 clips, including only four safe clips and two known hazards.
- Ground-truth annotation and semantic prediction matching involve human judgement.

---

## Author

**Vaishali Veera Saravana Perumal**

MSc Artificial Intelligence and Machine Learning

University of Birmingham