# 🛡️ Multi-Cue Anti-Spoofing Face Recognition Attendance System

### Using CNN Texture Analysis, Random Forest Fusion, and ArcFace Identity Recognition

> A real-time liveness detection system that combines deep learning and classical computer vision to prevent spoofing attacks on face recognition attendance systems — achieving **ACER = 0.000** and **AUC = 1.000** across all attack categories.

---

## 📌 Overview

Face recognition systems are vulnerable to **presentation attacks** — where an attacker presents a printed photo, plays a recorded video, or displays a digital screen of a registered user to deceive the system. This project addresses that vulnerability by implementing a **multi-cue liveness detection pipeline** that verifies whether the presented face belongs to a live person before allowing identity recognition and attendance logging.

Two system versions were developed and evaluated:

| Version      | Texture Method      | Fusion Method             | Identity               |
| ------------ | ------------------- | ------------------------- | ---------------------- |
| **Baseline** | LBP Shannon Entropy | Fixed Weighted Sum        | dlib ResNet 128-D      |
| **Proposed** | MobileNetV2 CNN     | Random Forest (100 trees) | DeepFace ArcFace 512-D |

### Key Features

- **EAR Blink Detection** — geometric liveness cue using dlib 68 facial landmarks
- **MobileNetV2 CNN Texture Analysis** — fine-tuned binary classifier (live vs spoof)
- **Randomised Challenge-Response** — unpredictable head movement prompts defeating video replay
- **Random Forest Fusion** — AI-learned decision boundary replacing fixed weighted sum
- **DeepFace ArcFace Recognition** — 512-D deep metric learning with angular margin loss
- **ISO/IEC 30107-3 Evaluation** — APCER, BPCER, ACER, AUC metrics

---

## 📊 Results

### Final Performance — Proposed System (CNN + Random Forest + ArcFace)

| Test Case     | Baseline (LBP + WS) | Proposed (CNN + RF) | Change       |
| ------------- | ------------------- | ------------------- | ------------ |
| Real Person   | 100.0%              | 100.0%              | —            |
| Printed Photo | 70.0%               | 100.0%              | ▲ +30.0%     |
| Video Replay  | 10.0%               | 100.0%              | ▲ +90.0%     |
| **Overall**   | **60.0%**           | **100.0%**          | **▲ +40.0%** |
| APCER         | 0.600               | 0.000               | ▼ −0.600     |
| BPCER         | 0.000               | 0.000               | —            |
| **ACER**      | **0.300**           | **0.000**           | **▼ −0.300** |
| **AUC**       | **0.805**           | **1.000**           | **▲ +0.195** |

> Evaluated on 30 trials: 10 real person, 10 printed photo attack, 10 video replay attack.

### RF Feature Importance

| Feature                      | Importance |
| ---------------------------- | ---------- |
| CNN Texture (S_CNN)          | 47%        |
| EAR Blink (S_EAR)            | 30%        |
| Challenge-Response (S_Chall) | 23%        |

### CNN Training Summary

| Metric               | Value                                     |
| -------------------- | ----------------------------------------- |
| Architecture         | MobileNetV2 (transfer learning, ImageNet) |
| Training Data        | 200 real + 200 fake face crops            |
| Loss Function        | Binary Cross-Entropy + Adam optimiser     |
| Best Epoch           | 7 (validation accuracy: 100%)             |
| Hard Block Threshold | CNN score < 0.35 → FAKE immediately       |

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
(BlinkCounter)     (MobileNetV2)         (solvePnP angles)
EAR < 0.25         score < 0.35          random prompt
≥2 frames          → FAKE (hard block)   yaw > 15° / pitch > 10°
    │                     │                     │
    └─────────────────────┴─────────────────────┘
                          │
                          ▼
             Random Forest Fusion
             (100 decision trees)
             Input: [S_EAR, S_CNN, S_Chall]
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           FAKE ✗                 REAL ✓
       Session Rejected               │
                                      ▼
                         DeepFace ArcFace Recognition
                         (512-D Euclidean distance, d < 0.68)
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                   Unknown Person         Identity Confirmed
                   Not enrolled           Attendance Logged
                                          (attendance_log.csv)
```

---

## 🤖 AI Components

| Component        | Method           | Details                                                                      |
| ---------------- | ---------------- | ---------------------------------------------------------------------------- |
| Spoof Detection  | MobileNetV2 CNN  | Fine-tuned on 400 device-specific samples, 100% val accuracy at epoch 7      |
| Liveness Fusion  | Random Forest    | 100 trees, trained on experimental session data, feature importance verified |
| Face Recognition | DeepFace ArcFace | 512-D embedding, angular margin loss, Euclidean distance d < 0.68            |

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
spoof_cnn_model.h5              # Trained MobileNetV2 CNN model
fusion_rf_model.pkl             # Trained Random Forest fusion model
face_db.pkl                     # Enrolled face database (ArcFace embeddings)
lbp_calibration.pkl             # Personalised LBP calibration (baseline)
known_faces/                    # Face image crops per enrolled user
cnn_training_data/              # CNN training images (real/ and fake/)
attendance_log.csv              # Attendance records with timestamps
experiment_results.csv          # Experiment session data for RF training
experiment_results_BEFORE.csv   # Baseline results (LBP + Weighted Sum)
experiment_results_AFTER.csv    # Proposed results (CNN + Random Forest)
roc_curve.png                   # Generated ROC curve plot
```

---

## ⚙️ Requirements

- Python 3.10 – 3.12
- Windows 10 / 11
- Standard webcam (720p @ 30fps minimum)
- No GPU required — runs on CPU in real time

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

### HUD Status Indicators (top-right corner)

```
CNN:     ON   ← green = CNN model loaded and active
RF:      ON   ← green = Random Forest model loaded and active
ArcFace: ON   ← green = DeepFace ArcFace model active
```

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
- When prompted for **FAKE** samples → hold a printed photo or play a video replay to the camera
- Recommended: **200 samples per class minimum**

### Step 3 — Run experiments to collect fusion training data

```
Press X in main.py:
  1 = real person   (run 10+ times)
  2 = printed photo (run 10+ times)
  3 = video replay  (run 10+ times)
```

### Step 4 — Train Random Forest fusion model

```bash
python train_fusion.py --eval
```

### Step 5 — Verify all AI components are active

Run `main.py` and check the top-right HUD:

```
CNN: ON        ← green
RF:  ON        ← green
ArcFace: ON    ← green
```

## 🛠️ Configuration

All parameters are in `config.py`:

```python
# AI model paths
CNN_MODEL_PATH      = "spoof_cnn_model.h5"
RF_MODEL_PATH       = "fusion_rf_model.pkl"

# Blink detection
EAR_THRESHOLD       = 0.25      # Below this = eye closed
BLINK_COUNT_MIN     = 2         # Minimum blinks required per session
BLINK_COUNT_MAX     = 4         # Maximum blinks to count

# Challenge-response
CHALLENGE_COUNT_MIN = 2         # Minimum challenges per session
CHALLENGE_COUNT_MAX = 3         # Maximum challenges per session
CHALLENGES          = ["turn_left", "turn_right", "nod", "turn_up"]
YAW_THRESHOLD       = 15        # Degrees — turn left/right compliance
PITCH_THRESHOLD     = 10        # Degrees — nod/turn up compliance

# CNN hard block
CNN_SCORE_MIN       = 0.35      # Below this = immediate FAKE rejection

# Random Forest fusion
FUSION_PASS_SCORE   = 0.55      # Baseline weighted sum threshold (legacy)
RF_TREES            = 100       # Number of decision trees

# ArcFace recognition
DEEPFACE_MODEL      = "ArcFace"
DEEPFACE_THRESHOLD  = 0.68      # Euclidean distance — below = same person
DEEPFACE_FALLBACK   = "dlib"    # Fallback if ArcFace unavailable (d < 0.45)
```

---

## 📦 Dependencies Explained

| Library         | Purpose                                                  |
| --------------- | -------------------------------------------------------- |
| `opencv-python` | Webcam capture, face drawing, solvePnP head pose         |
| `dlib`          | HOG face detection, 68-point landmarks, ResNet fallback  |
| `scipy`         | Euclidean distance for EAR computation                   |
| `numpy`         | Array operations throughout                              |
| `scikit-image`  | LBP texture feature extraction (baseline)                |
| `matplotlib`    | ROC curve generation and plotting                        |
| `scikit-learn`  | Random Forest, cross-validation, AUC, feature importance |
| `tensorflow`    | MobileNetV2 CNN training and inference                   |
| `deepface`      | ArcFace 512-D face recognition and embedding             |

---

## ⚠️ Known Limitations

- Evaluation dataset is small (30 trials) — results may not fully generalise to unseen subjects
- 3D silicone mask attacks were not evaluated
- CNN was trained and tested on same-session data — cross-session generalisation not verified
- Head pose estimation may drift under extreme camera angles
- System currently processes one face per frame — multi-person attendance not yet supported
- Recognition confidence depends on enrolment sample quality and lighting conditions
