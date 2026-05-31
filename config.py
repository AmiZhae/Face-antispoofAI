"""
config.py – All tunable parameters in one place.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
LANDMARK_MODEL   = "shape_predictor_68_face_landmarks.dat"
KNOWN_FACES_DIR  = "known_faces"
ATTENDANCE_CSV   = "attendance_log.csv"
RESULTS_CSV      = "experiment_results.csv"
LBP_CALIB_FILE   = "lbp_calibration.pkl"

# ── CNN Spoof Model (replaces LBP) ────────────────────────────────────────────
CNN_MODEL_PATH   = "spoof_cnn_model.h5"   # trained by train_cnn.py
CNN_INPUT_SIZE   = (64, 64)
CNN_SCORE_MIN    = 0.35                   # fusion veto threshold

# ── LBP fallback (used only if CNN model not trained yet) ─────────────────────
LBP_RADIUS          = 1
LBP_N_POINTS        = 8
LBP_ROI_SIZE        = (64, 64)
TEXTURE_ENTROPY_MIN = 2.5
LBP_CHI_THRESHOLD   = 0.08
TEXTURE_SCORE_MIN   = 0.25

# ── EAR / Blink ───────────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 2
BLINK_MIN_FRAMES  = 4
BLINK_COUNT_MIN   = 2
BLINK_COUNT_MAX   = 4

# ── Challenge-Response ────────────────────────────────────────────────────────
CHALLENGES          = ["turn_left", "turn_right", "nod", "turn_up"]
CHALLENGE_COUNT_MIN = 2
CHALLENGE_COUNT_MAX = 3
CHALLENGE_TIMEOUT   = 6.0

# ── Head Pose ─────────────────────────────────────────────────────────────────
POSE_YAW_THRESHOLD   = 12.0
POSE_PITCH_THRESHOLD = 10.0

# ── Random Forest Fusion (replaces hardcoded weights) ────────────────────────
RF_MODEL_PATH    = "fusion_rf_model.pkl"  # trained by train_fusion.py
FUSION_PASS_SCORE = 0.55                  # used only when RF not trained yet
WEIGHT_EAR        = 0.30
WEIGHT_TEXTURE    = 0.35
WEIGHT_CHALLENGE  = 0.35

# ── DeepFace Recognition (replaces dlib ResNet) ───────────────────────────────
DEEPFACE_MODEL      = "ArcFace"           # ArcFace, VGG-Face, Facenet
DEEPFACE_DETECTOR   = "opencv"
DEEPFACE_THRESHOLD  = 0.68               # distance threshold for ArcFace
DEEPFACE_CONFIDENCE_MIN = 0.30

# ── Display ───────────────────────────────────────────────────────────────────
RESULT_HOLD_SECS = 3.0
