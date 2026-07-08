# Smart Kitchen Safety Assistant for Elderly Users
### A Vision-Language-Model-Grounded LLM Agent for AR-Guided Cooking Assistance and Hazard Detection

## Overview

This repository contains the implementation developed as part of my Master's dissertation.

The project proposes an intelligent kitchen safety assistant that combines computer vision, Vision-Language Models (VLMs), and Large Language Models (LLMs) to assist elderly users while cooking. The system continuously observes first-person cooking activities, understands the current cooking context, detects potential hazards, maintains memory across cooking steps, and provides contextual safety guidance suitable for future Augmented Reality (AR) interfaces.

Unlike traditional rule-based kitchen monitoring systems, this project leverages semantic scene understanding and reasoning to interpret ongoing activities rather than relying solely on predefined rules.

---

## Research Objectives

The proposed system aims to:

- Understand egocentric cooking activities.
- Detect kitchen objects using computer vision.
- Generate semantic scene understanding using a Vision-Language Model.
- Maintain temporal memory across cooking steps.
- Detect potential kitchen hazards.
- Identify repeated or forgotten actions.
- Assist users in resuming cooking after interruptions.
- Generate contextual cooking guidance suitable for AR-based visualization.

---

## Proposed Pipeline

```text
Cooking Video
      │
      ▼
Frame Extraction
      │
      ▼
YOLO Object Detection
      │
      ▼
Scene JSON Generation
      │
      ▼
Keyframe Selection
      │
      ▼
Gemini 2.5 Flash (Vision)
      │
      ▼
Scene Understanding
      │
      ▼
Memory Update
      │
      ▼
Gemini 2.5 Flash (Reasoning)
      │
      ▼
Hazard Warnings
Repeated Action Detection
Interruption Recovery
Cooking Guidance
```

---

## Current Progress

Completed:

- Literature review
- Dataset selection and curation
- EPIC-KITCHENS preprocessing pipeline
- Frame extraction
- YOLO object detection
- Scene JSON generation
- Keyframe selection
- Complete perception pipeline

In Progress:

- Gemini Vision integration
- Scene understanding
- Memory module
- LLM reasoning
- Hazard detection
- Cooking guidance generation

Future Work:

- AR visualization
- Real-time webcam integration
- Live kitchen monitoring

---

## Dataset

This project uses the **EPIC-KITCHENS-100** egocentric cooking dataset.

A curated subset of **210 cooking clips** was selected to represent common kitchen activities relevant to elderly cooking safety, including:

- chopping
- cutting
- slicing
- peeling
- stirring
- pouring
- opening cupboards/fridge
- turning stove and oven on/off
- handling knives
- placing utensils
- mixing ingredients

The dataset itself is **not included** in this repository due to licensing and storage limitations.

---

## Project Structure

```text
KitchenSafetyAssistant/

├── app/
├── pipeline/
│   ├── 00_config.py
│   ├── 01_extract_frames.py
│   ├── 02_detect_objects.py
│   ├── 03_build_scene_json.py
│   ├── 04_select_keyframes.py
│   └── ...
│
├── prompts/
│
├── scripts/
│   ├── dataset preparation
│   ├── dataset verification
│   ├── download utilities
│   └── clip extraction
│
├── dataset/
│   ├── annotations/
│   ├── demo/
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- YOLO11 (Ultralytics)
- Google Gemini 2.5 Flash
- OpenCV
- Pandas
- NumPy
- FFmpeg
- Jupyter Notebook
- Visual Studio Code

---

## Repository Notes

Large datasets and generated outputs are excluded from GitHub using `.gitignore`.

Only the source code, prompts, configuration files, and scripts are tracked.

---

## Author

**Vaishali V**

Master's Dissertation
