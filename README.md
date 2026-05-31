# 🛡️ AI Anti-Spoofing Face Recognition System

> A multi-cue liveness detection system that combines deep learning and classical computer vision to prevent spoofing attacks on face recognition systems.

---

## 📌 Overview

Face recognition systems are vulnerable to **presentation attacks** — where an attacker presents a printed photo or plays a recorded video to deceive the system. This project addresses that vulnerability by implementing a **three-module liveness detection pipeline** that verifies whether the presented face belongs to a live person before allowing identity recognition and attendance logging.

### Key Features

- **EAR Blink Detection** — geometric liveness cue using 68 facial landmarks
- **CNN Texture Analysis** — MobileNetV2 trained on real vs fake face samples
- **Randomised Challenge-Response** — unpredictable head movement prompts
- **Random Forest Fusion** — AI-learned decision boundary replacing fixed weights
- **DeepFace ArcFace Recognition** — 512-D deep metric learning face matching
- **ISO/IEC 30107-3 Evaluation** — APCER, BPCER, ACER metrics

---

## 🏗️ System Architecture

```
Webcam Input
    │
    ▼
Face Detection (dlib HOG + 68 Landmarks)
    │
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼
EAR Blink          CNN Texture           Challenge-Response
Detection          Analysis              (turn/nod/tilt)
(BlinkCounter)     (MobileNetV2)
    │                     │                     │
    └─────────────────────┴─────────────────────┘
                          │
                          ▼
             Random Forest Fusion
             (100 decision trees)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           FAKE ✗                 REAL ✓
       Session Rejected               │
                                      ▼
                         DeepFace ArcFace Recognition
                         (512-D Euclidean distance)
                                      │
                                      ▼
                            Attendance Logged
                            (attendance_log.csv)
```

---

## 🤖 AI Components

| Component        | Method           | Details                                                       |
| ---------------- | ---------------- | ------------------------------------------------------------- |
| Spoof Detection  | MobileNetV2 CNN  | Fine-tuned on 400 device-specific samples, 94.5% val accuracy |
| Liveness Fusion  | Random Forest    | 100 trees, trained on experimental session data               |
| Face Recognition | DeepFace ArcFace | 512-D embedding, angular margin loss, d < 0.68 threshold      |

### Before vs After AI Upgrade

| Metric        | Before (LBP + Weighted) | After (CNN + Random Forest) |
| ------------- | ----------------------- | --------------------------- |
| Real Person   | 100.0%                  | 100.0%                      |
| Printed Photo | 70.0%                   | 90.0%                       |
| Video Replay  | 20.0%                   | 75.0%                       |
| **Overall**   | **63.3%**               | **88.3%**                   |
| APCER         | 0.400                   | 0.125                       |
| ACER          | 0.200                   | 0.063                       |

---

## 📐 Mathematical Models

| Module        | Formula                                        |
| ------------- | ---------------------------------------------- |
| EAR           | `EAR = (‖p₂−p₆‖ + ‖p₃−p₅‖) / (2‖p₁−p₄‖)`       |
| CNN Loss      | `L = −[y·log(p) + (1−y)·log(1−p)]`             |
| Fusion        | `Score = 0.30×EAR + 0.35×CNN + 0.35×Challenge` |
| Random Forest | `P(REAL) = (1/T) Σ hₜ([EAR, CNN, Chall])`      |
| ArcFace       | `d(f₁,f₂) = ‖f₁−f₂‖₂ = √Σ(f₁ᵢ−f₂ᵢ)²`           |
| ACER          | `ACER = (APCER + BPCER) / 2`                   |

---

## 📁 Project Structure

```
AI-Anti-Spoof-Face/
│
├── main.py              # Entry point — webcam loop, keyboard controls, HUD
├── liveness.py          # EAR blink, CNN texture, challenge-response, fusion
├── face_rec.py          # Enrolment, DeepFace ArcFace recognition, attendance
├── experiment.py        # Experiment recording, ISO metrics, ROC curve
├── config.py            # All tunable parameters in one place
├── train_cnn.py         # Collect training data + train MobileNetV2 CNN
├── train_fusion.py      # Train Random Forest from experiment CSV
│
├── requirements.txt     # Python dependencies
├── .gitignore           # Excluded files
└── README.md            # This file
```

### Generated at Runtime (not in repo)

```
spoof_cnn_model.h5          # Trained CNN model
fusion_rf_model.pkl         # Trained Random Forest model
face_db.pkl                 # Enrolled face database
lbp_calibration.pkl         # Personalised LBP calibration
known_faces/                # Face image crops per user
cnn_training_data/          # CNN training images
attendance_log.csv          # Attendance records
experiment_results.csv      # Experiment session data
roc_curve.png               # Generated ROC curve
```

---

## ⚙️ Requirements

- Python 3.12
- Windows 10 / 11

```
opencv-python
dlib
scipy
numpy
scikit-image
matplotlib
scikit-learn
tensorflow
deepface
```

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/AmiZhae/Face-antispoofAI.git
cd Face-antispoofAI
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dlib landmark model**

Download `shape_predictor_68_face_landmarks.dat` from:

```
https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
```

Extract and place it in the project root folder.

---

## 🎮 Usage

### Run the system

```bash
python main.py
```

### Keyboard Controls

| Key | Action                                     |
| --- | ------------------------------------------ |
| `R` | Start liveness session (attendance mode)   |
| `E` | Enrol a new user                           |
| `X` | Experiment mode (real / photo / video)     |
| `T` | Train Random Forest from experiment data   |
| `P` | Print attendance log and experiment report |
| `G` | Generate ROC curve → `roc_curve.png`       |
| `L` | List all enrolled users                    |
| `F` | Export enrolled face images as JPG         |
| `D` | Delete a user                              |
| `Q` | Quit                                       |

---

## 🧠 Training the AI Models

### Step 1 — Enrol yourself first

```
Run main.py → Press E → type your name → Press Enter
```

### Step 2 — Collect CNN training data and train

```bash
python train_cnn.py --collect --train
```

- When prompted for **REAL** samples → sit in front of camera, press `A` for auto-capture
- When prompted for **FAKE** samples → hold a printed photo or play a video to the camera
- Recommended: 200 samples per class

### Step 3 — Run experiments to collect fusion training data

```
Press X in main.py:
  1 = real person   (run 10+ times)
  2 = printed photo (run 10+ times)
  3 = video replay  (run 10+ times)
```

### Step 4 — Train Random Forest

```bash
python train_fusion.py --eval
```

### Step 5 — Verify all AI components are active

Run `main.py` and check top-right corner of the window:

```
CNN: ON        ← green
RF:  ON        ← green
ArcFace: ON    ← green
```

---

## 📊 Evaluation

### Generate before/after comparison

```bash
# Rename your baseline results
rename experiment_results.csv experiment_results_BEFORE.csv

# Run 30 more experiments with AI active, then:
rename experiment_results.csv experiment_results_AFTER.csv

# Generate comparison table
python compare_results.py
```

### Metrics used (ISO/IEC 30107-3)

| Metric | Formula             | Meaning                |
| ------ | ------------------- | ---------------------- |
| APCER  | FP / (FP + TN)      | Attack acceptance rate |
| BPCER  | FN / (FN + TP)      | Genuine rejection rate |
| ACER   | (APCER + BPCER) / 2 | Average error rate     |

---

## 🛠️ Configuration

All parameters are in `config.py`:

```python
# AI model paths
CNN_MODEL_PATH   = "spoof_cnn_model.h5"
RF_MODEL_PATH    = "fusion_rf_model.pkl"

# Blink detection
EAR_THRESHOLD    = 0.25
BLINK_COUNT_MIN  = 2
BLINK_COUNT_MAX  = 4

# Challenge-response
CHALLENGE_COUNT_MIN = 2
CHALLENGE_COUNT_MAX = 3
CHALLENGES = ["turn_left", "turn_right", "nod", "turn_up"]

# Fusion
FUSION_PASS_SCORE = 0.55
CNN_SCORE_MIN     = 0.35

# ArcFace recognition
DEEPFACE_MODEL     = "ArcFace"
DEEPFACE_THRESHOLD = 0.68
```

---

## 📦 Dependencies Explained

| Library         | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `opencv-python` | Webcam capture, face drawing, image processing      |
| `dlib`          | Face detection, 68-point landmarks, ResNet fallback |
| `scipy`         | Euclidean distance for EAR                          |
| `numpy`         | Array operations                                    |
| `scikit-image`  | LBP texture feature extraction                      |
| `matplotlib`    | ROC curve generation                                |
| `scikit-learn`  | Random Forest, cross-validation, AUC                |
| `tensorflow`    | MobileNetV2 CNN training and inference              |
| `deepface`      | ArcFace face recognition                            |

---

## ⚠️ Known Limitations

- LBP texture analysis is sensitive to lighting variation
- Head pose estimation may drift under non-frontal camera angles
- Video replay accuracy (75%) can be improved with more CNN training data
- Recognition confidence is affected by enrolment sample quality
- CNN requires at least 200 samples per class for best results
