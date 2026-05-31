import os
import pickle
import time
import random
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern

import config

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_IDX  = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

_MODEL_PTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
_POSE_IDX = [30, 8, 36, 45, 48, 54]

INSTRUCTIONS = {
    "turn_left":  "Turn head LEFT",
    "turn_right": "Turn head RIGHT",
    "nod":        "Nod DOWN",
    "turn_up":    "Tilt head UP",
}


# ══════════════════════════════════════════════════════════════════════════════
# CNN Spoof Detector (MobileNetV2)
# ══════════════════════════════════════════════════════════════════════════════
class CNNSpoofDetector:
    """
    Thin wrapper around the trained MobileNetV2 spoof model.
    Falls back to LBP if model file not found.
    """
    def __init__(self):
        self._model  = None
        self._loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(config.CNN_MODEL_PATH):
            print("[CNN] Model not found — using LBP fallback.")
            print(f"      Train it: python train_cnn.py --collect --train")
            return
        try:
            import tensorflow as tf
            self._model  = tf.keras.models.load_model(config.CNN_MODEL_PATH)
            self._loaded = True
            print(f"[CNN] MobileNetV2 spoof model loaded — AI texture detection active.")
        except Exception as e:
            print(f"[CNN] Failed to load model: {e}")

    def is_available(self):
        return self._loaded

    def score(self, frame, face_rect) -> float | None:
        """Return confidence [0,1] that face is REAL, or None if unavailable."""
        if not self._loaded:
            return None
        x1 = max(0, face_rect.left());  y1 = max(0, face_rect.top())
        x2 = min(frame.shape[1], face_rect.right())
        y2 = min(frame.shape[0], face_rect.bottom())
        if x2 <= x1 or y2 <= y1:
            return None
        roi = cv2.resize(frame[y1:y2, x1:x2], config.CNN_INPUT_SIZE)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        roi = np.expand_dims(roi, axis=0)
        pred = self._model.predict(roi, verbose=0)[0][0]
        return float(pred)


# Singleton — loaded once at startup
_cnn_detector = CNNSpoofDetector()


def cnn_available():
    return _cnn_detector.is_available()


# ══════════════════════════════════════════════════════════════════════════════
# LBP fallback (used when CNN not trained)
# ══════════════════════════════════════════════════════════════════════════════
def _lbp_hist(roi, eps=1e-7):
    lbp    = local_binary_pattern(roi, config.LBP_N_POINTS, config.LBP_RADIUS, method="uniform")
    n_bins = config.LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0,n_bins), density=True)
    hist = hist.astype("float32"); hist /= (hist.sum() + eps)
    return hist

def _entropy(hist):
    return float(-np.sum(hist * np.log2(hist + 1e-10)))

def _chi_sq(h1, h2):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 0.5 * np.sum(((h1-h2)**2)/(h1+h2+1e-10))
    return float(d)

def extract_face_roi(frame, face_rect):
    x1=max(0,face_rect.left()); y1=max(0,face_rect.top())
    x2=min(frame.shape[1],face_rect.right()); y2=min(frame.shape[0],face_rect.bottom())
    if x2<=x1 or y2<=y1: return None
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray[y1:y2,x1:x2], config.LBP_ROI_SIZE)

_lbp_ref = None
def reload_lbp_calibration():
    global _lbp_ref
    if os.path.exists(config.LBP_CALIB_FILE):
        with open(config.LBP_CALIB_FILE,"rb") as f:
            _lbp_ref = pickle.load(f)
        print("[LBP] Calibration loaded.")
    else:
        _lbp_ref = None

def save_lbp_calibration(rois):
    hists = [_lbp_hist(r) for r in rois if r is not None]
    if not hists: return
    ref = np.mean(hists, axis=0).astype("float32")
    with open(config.LBP_CALIB_FILE,"wb") as f:
        pickle.dump(ref, f)
    print(f"[LBP] Calibration saved ({len(hists)} samples).")

def lbp_is_calibrated():
    return _lbp_ref is not None

reload_lbp_calibration()

def _lbp_score(roi):
    if roi is None: return 0.5
    hist = _lbp_hist(roi)
    ent  = _entropy(hist)
    if ent < config.TEXTURE_ENTROPY_MIN: return 0.0
    if _lbp_ref is not None:
        chi   = _chi_sq(hist, _lbp_ref)
        score = float(np.exp(-chi / config.LBP_CHI_THRESHOLD))
        return float(np.clip(score, 0, 1))
    return float(np.clip((ent - config.TEXTURE_ENTROPY_MIN)/(4.0 - config.TEXTURE_ENTROPY_MIN), 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# Unified spoof score — CNN or LBP
# ══════════════════════════════════════════════════════════════════════════════
def get_spoof_score(frame, face_rect):
    """Returns (score, method_label)."""
    if _cnn_detector.is_available():
        s = _cnn_detector.score(frame, face_rect)
        if s is not None:
            return float(np.clip(s, 0, 1)), "CNN"
    roi = extract_face_roi(frame, face_rect)
    return _lbp_score(roi), ("LBP-cal" if lbp_is_calibrated() else "LBP-ent")


# ══════════════════════════════════════════════════════════════════════════════
# EAR helpers
# ══════════════════════════════════════════════════════════════════════════════
def _ear(pts):
    A=dist.euclidean(pts[1],pts[5]); B=dist.euclidean(pts[2],pts[4]); C=dist.euclidean(pts[0],pts[3])
    return (A+B)/(2.0*C)

def get_ear(shape):
    l=np.array([(shape.part(i).x,shape.part(i).y) for i in LEFT_EYE_IDX])
    r=np.array([(shape.part(i).x,shape.part(i).y) for i in RIGHT_EYE_IDX])
    return (_ear(l)+_ear(r))/2.0

def eye_points(shape):
    l=np.array([(shape.part(i).x,shape.part(i).y) for i in LEFT_EYE_IDX],dtype=np.int32)
    r=np.array([(shape.part(i).x,shape.part(i).y) for i in RIGHT_EYE_IDX],dtype=np.int32)
    return l,r


# ══════════════════════════════════════════════════════════════════════════════
# Blink Counter
# ══════════════════════════════════════════════════════════════════════════════
class BlinkCounter:
    def __init__(self, window=config.CHALLENGE_TIMEOUT):
        self.blinks_required = random.randint(config.BLINK_COUNT_MIN, config.BLINK_COUNT_MAX)
        self.window=window; self._count=0; self._below=0; self._start=None
        print(f"[BLINK] Required this session: {self.blinks_required}")

    def reset(self): self._count=0; self._below=0; self._start=None
    def start(self): self._start=time.time()

    @property
    def elapsed(self): return (time.time()-self._start) if self._start else 0.0
    @property
    def remaining(self): return max(0.0, self.window-self.elapsed)
    @property
    def count(self): return self._count

    def update(self, ear):
        if ear < config.EAR_THRESHOLD: self._below+=1
        else:
            if self._below >= config.BLINK_MIN_FRAMES: self._count+=1
            self._below=0

    def score(self): return min(1.0, self._count/max(1,self.blinks_required))


# ══════════════════════════════════════════════════════════════════════════════
# Head Pose
# ══════════════════════════════════════════════════════════════════════════════
def estimate_pose(shape, frame_shape):
    h,w=frame_shape[:2]; focal=w
    cam=np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],dtype=np.float64)
    pts=np.array([(shape.part(i).x,shape.part(i).y) for i in _POSE_IDX],dtype=np.float64)
    ok,rv,_=cv2.solvePnP(_MODEL_PTS,pts,cam,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0,0.0
    rm,_=cv2.Rodrigues(rv); sy=np.sqrt(rm[0,0]**2+rm[1,0]**2)
    if sy>1e-6:
        pitch=float(np.degrees(np.arctan2(rm[2,1],rm[2,2])))
        yaw=float(np.degrees(np.arctan2(-rm[2,0],sy)))
    else:
        pitch=float(np.degrees(np.arctan2(-rm[1,2],rm[1,1])))
        yaw=float(np.degrees(np.arctan2(-rm[2,0],sy)))
    return yaw,pitch


# ══════════════════════════════════════════════════════════════════════════════
# Challenge-Response
# ══════════════════════════════════════════════════════════════════════════════
class ChallengeSession:
    def __init__(self):
        num=random.randint(config.CHALLENGE_COUNT_MIN, config.CHALLENGE_COUNT_MAX)
        pool=config.CHALLENGES[:]; random.shuffle(pool)
        self._challenges=pool[:num]; self._idx=0; self._start=None
        self._passed=0; self._done=False
        self._b_below=0; self._b_count=0; self._base_yaw=None; self._base_pitch=None
        print(f"[CHALLENGE] Required: {num} ({', '.join(self._challenges)})")

    def start(self): self._start=time.time()

    @property
    def done(self): return self._done
    @property
    def score(self): return self._passed/max(1,len(self._challenges))

    def current_instruction(self):
        if self._done or self._idx>=len(self._challenges): return "Done"
        return INSTRUCTIONS.get(self._challenges[self._idx],"")

    @property
    def remaining(self):
        if self._start is None: return config.CHALLENGE_TIMEOUT
        return max(0.0, config.CHALLENGE_TIMEOUT-(time.time()-self._start))

    def _reset_state(self):
        self._b_below=0; self._b_count=0; self._base_yaw=None; self._base_pitch=None

    def _next(self):
        self._idx+=1; self._start=time.time(); self._reset_state()
        if self._idx>=len(self._challenges): self._done=True

    def update(self, shape, ear, frame_shape):
        if self._done: return "done"
        ch=self._challenges[self._idx]; elapsed=time.time()-self._start
        result=self._eval(ch,shape,ear,frame_shape)
        if result=="pass":
            self._passed+=1; self._next()
            return "done" if self._done else "pass"
        if elapsed>=config.CHALLENGE_TIMEOUT:
            self._next()
            return "done" if self._done else "fail"
        return "waiting"

    def _eval(self, ch, shape, ear, frame_shape):
        yaw,pitch=estimate_pose(shape,frame_shape)
        if ch=="turn_left":
            if self._base_yaw is None: self._base_yaw=yaw; return "waiting"
            return "pass" if (yaw-self._base_yaw)>config.POSE_YAW_THRESHOLD else "waiting"
        if ch=="turn_right":
            if self._base_yaw is None: self._base_yaw=yaw; return "waiting"
            return "pass" if (yaw-self._base_yaw)<-config.POSE_YAW_THRESHOLD else "waiting"
        if ch=="nod":
            if self._base_pitch is None: self._base_pitch=pitch; return "waiting"
            return "pass" if (pitch-self._base_pitch)>config.POSE_PITCH_THRESHOLD else "waiting"
        if ch=="turn_up":
            if self._base_pitch is None: self._base_pitch=pitch; return "waiting"
            return "pass" if (self._base_pitch-pitch)>config.POSE_PITCH_THRESHOLD else "waiting"
        return "waiting"


# ══════════════════════════════════════════════════════════════════════════════
# Random Forest Fusion (AI) — with weighted sum fallback
# ══════════════════════════════════════════════════════════════════════════════
_rf_model = None

def _load_rf():
    global _rf_model
    if os.path.exists(config.RF_MODEL_PATH):
        with open(config.RF_MODEL_PATH,"rb") as f:
            _rf_model = pickle.load(f)
        print("[RF] Random Forest fusion model loaded — AI decision making active.")
    else:
        print("[RF] No trained model found — using weighted sum fallback.")
        print("     Train it: python train_fusion.py")

_load_rf()

def rf_available():
    return _rf_model is not None


@dataclass
class LivenessResult:
    ear_score      : float
    texture_score  : float
    texture_method : str
    challenge_score: float
    final_score    : float
    is_real        : bool
    fusion_method  : str = "weighted"
    reason         : str = ""

    def label(self): return "REAL" if self.is_real else "FAKE"


def fuse(ear_s, tex_s, tex_method, chg_s) -> LivenessResult:
    ear_s = float(np.clip(ear_s, 0, 1))
    tex_s = float(np.clip(tex_s, 0, 1))
    chg_s = float(np.clip(chg_s, 0, 1))

    # Hard blocks apply regardless of fusion method
    if tex_s < config.CNN_SCORE_MIN:
        return LivenessResult(ear_s, tex_s, tex_method, chg_s,
                              0.0, False, "hard-block", "texture block")
    if ear_s == 0.0 and chg_s < 0.5:
        return LivenessResult(ear_s, tex_s, tex_method, chg_s,
                              0.0, False, "hard-block", "no blink + challenge fail")

    features = np.array([[ear_s, tex_s, chg_s]])

    if _rf_model is not None:
        # AI fusion — Random Forest
        prob    = _rf_model.predict_proba(features)[0]
        # prob[1] = probability of REAL class
        real_prob = float(prob[1]) if len(prob) > 1 else float(_rf_model.predict(features)[0])
        return LivenessResult(ear_s, tex_s, tex_method, chg_s,
                              real_prob, real_prob >= 0.5, "RandomForest")
    else:
        # Fallback weighted sum
        final = float(np.clip(
            config.WEIGHT_EAR * ear_s +
            config.WEIGHT_TEXTURE * tex_s +
            config.WEIGHT_CHALLENGE * chg_s, 0, 1))
        return LivenessResult(ear_s, tex_s, tex_method, chg_s,
                              final, final >= config.FUSION_PASS_SCORE, "weighted")
