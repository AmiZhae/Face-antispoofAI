"""
main.py – AI-powered Liveness Detection + Face Recognition Attendance System

AI Components:
  1. MobileNetV2 CNN   — spoof texture detection (replaces LBP)
  2. Random Forest     — learned fusion classifier (replaces fixed weights)
  3. DeepFace ArcFace  — deep face recognition (replaces dlib ResNet)

KEYBOARD CONTROLS
──────────────────
  R  – Start liveness check (attendance mode)
  E  – Enrol new user
  X  – Experiment mode (real / photo / video)
  T  – Train AI models (CNN + Random Forest)
  P  – Print attendance + experiment report
  G  – Generate ROC curve
  L  – List enrolled users
  D  – Delete a user
  Q  – Quit
"""

import time
import cv2
import dlib
import numpy as np

import config
from liveness   import (BlinkCounter, ChallengeSession, LivenessResult,
                        eye_points, fuse, get_ear, get_spoof_score,
                        extract_face_roi, lbp_is_calibrated,
                        cnn_available, rf_available)
from face_rec   import (delete_user, enrol_webcam, export_face_images,
                        list_users, load_db, log_attendance,
                        print_attendance, recognise, DEEPFACE_AVAILABLE)
from experiment import generate_roc, print_report, save_result

GREEN  = (0,   220,  80)
RED    = (0,    40, 230)
YELLOW = (0,   200, 255)
CYAN   = (255, 200,   0)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
DARK   = (30,   30,  30)
ORANGE = (0,   165, 255)
PURPLE = (200,   0, 200)

WINDOW = "Liveness Detection FYP"


# ══════════════════════════════════════════════════════════════════════════════
# HUD helpers
# ══════════════════════════════════════════════════════════════════════════════
def put(frame, text, pos, color=WHITE, scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_score_bar(frame, label, score, y, color):
    put(frame, label, (10,y), color, 0.42, 1)
    put(frame, f"{score:.2f}", (160,y), color, 0.42, 1)
    bar_y = y+5
    cv2.rectangle(frame,(10,bar_y),(180,bar_y+10),(40,40,40),-1)
    cv2.rectangle(frame,(10,bar_y),(10+int(score*170),bar_y+10),color,-1)


def draw_result_banner(frame, result: LivenessResult):
    h,w = frame.shape[:2]
    color = GREEN if result.is_real else RED
    cv2.rectangle(frame,(0,h-58),(w,h),DARK,-1)
    put(frame,f">> {result.label()} <<",(w//2-85,h-22),color,1.1,3)
    tag = f"[{result.reason}]" if result.reason else f"score {result.final_score:.2f}"
    put(frame,tag,(w//2-60,h-6),GRAY,0.45,1)


def draw_ai_badges(frame, tex_method, fusion_method, rec_method=""):
    """Show which AI components are active."""
    h,w = frame.shape[:2]
    # Texture method badge
    tcol = GREEN if tex_method=="CNN" else (GREEN if "cal" in tex_method else ORANGE)
    put(frame, f"TEX:{tex_method}", (w-165, h-110), tcol, 0.42, 1)
    # Fusion method badge
    fcol = PURPLE if fusion_method=="RandomForest" else ORANGE
    put(frame, f"FUS:{fusion_method}", (w-165, h-92), fcol, 0.42, 1)
    # Recognition method badge
    if rec_method:
        rcol = GREEN if rec_method=="ArcFace" else ORANGE
        put(frame, f"REC:{rec_method}", (w-165, h-74), rcol, 0.42, 1)


def draw_ai_status(frame):
    """Top-right corner — show which AI models are loaded."""
    h,w = frame.shape[:2]
    items = [
        (f"CNN:{'ON' if cnn_available() else 'OFF'}", GREEN if cnn_available() else ORANGE),
        (f"RF:{'ON' if rf_available() else 'OFF'}",  GREEN if rf_available() else ORANGE),
        (f"ArcFace:{'ON' if DEEPFACE_AVAILABLE else 'OFF'}", GREEN if DEEPFACE_AVAILABLE else ORANGE),
    ]
    for i,(text,col) in enumerate(items):
        put(frame, text, (w-130, 22+i*18), col, 0.38, 1)


def draw_controls(frame, mode):
    h = frame.shape[0]
    if mode=="experiment_pick":
        line="1=real  2=photo  3=video  ESC=cancel"
    elif mode in ("typing_enrol","typing_delete"):
        line="Type name in PowerShell, press ENTER — then click this window"
    elif mode=="training":
        line="Training AI models... please wait"
    else:
        line="R=start  E=enrol  X=exp  T=train AI  P=report  G=ROC  L=list  D=delete  Q=quit"
    put(frame, line, (8,h-68), GRAY, 0.37, 1)


def draw_typing_overlay(frame, prompt):
    h,w = frame.shape[:2]
    cv2.rectangle(frame,(0,h//2-40),(w,h//2+40),DARK,-1)
    cv2.rectangle(frame,(2,h//2-38),(w-2,h//2+38),YELLOW,2)
    put(frame, prompt, (20,h//2+8), YELLOW, 0.55, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Enrolment (reuses main camera)
# ══════════════════════════════════════════════════════════════════════════════
def run_enrolment(name, detector, predictor, rec_model, cap):
    db = enrol_webcam(name, detector, predictor, rec_model, cap, n_samples=20)
    return db


# ══════════════════════════════════════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════════════════════════════════════
class Session:
    def __init__(self, exp_case=None):
        self.blink         = BlinkCounter()
        self.challenge     = ChallengeSession()
        self.result        = None
        self.identity      = None
        self.id_conf       = 0.0
        self.id_method     = ""
        self.exp_case      = exp_case
        self._logged       = False
        self._spoof_buf    = []
        self._spoof_method = "LBP-ent"

    def start(self):
        self.blink.start()
        self.challenge.start()

    @property
    def complete(self): return self.result is not None

    def update(self, frame, shape, face_rect, predictor, rec_model, db):
        ear = get_ear(shape)
        self.blink.update(ear)
        self.challenge.update(shape, ear, frame.shape)

        spoof_s, spoof_method = get_spoof_score(frame, face_rect)
        self._spoof_method = spoof_method
        self._spoof_buf.append(spoof_s)
        avg_spoof = float(np.mean(self._spoof_buf[-10:]))

        if self.challenge.done and self.result is None:
            self.result = fuse(self.blink.score(), avg_spoof,
                               spoof_method, self.challenge.score)
            if self.result.is_real:
                self.identity, self.id_conf, self.id_method = recognise(
                    frame, face_rect, predictor, rec_model, db)

        return ear, spoof_s, spoof_method

    def finalize(self):
        if self._logged or self.result is None: return
        self._logged = True
        if self.exp_case:
            save_result(self.result, self.exp_case)
        elif self.result.is_real and self.identity and self.identity != "Unknown":
            log_attendance(self.identity, self.result.final_score,
                           self.id_conf, self.result.fusion_method,
                           self.result.texture_method)


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("[INFO] Loading dlib models...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.LANDMARK_MODEL)
    # dlib rec model still needed as fallback
    try:
        rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    except Exception:
        rec_model = None
        print("[WARN] dlib face rec model not found — DeepFace only mode.")

    db = load_db()

    print(f"\n[AI STATUS]")
    print(f"  Texture  : {'MobileNetV2 CNN' if cnn_available() else 'LBP (train CNN: python train_cnn.py --collect --train)'}")
    print(f"  Fusion   : {'Random Forest' if rf_available() else 'Weighted sum (train RF: python train_fusion.py)'}")
    print(f"  Face Rec : {'DeepFace ArcFace' if DEEPFACE_AVAILABLE else 'dlib ResNet (install: pip install deepface tensorflow)'}")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cv2.namedWindow(WINDOW)

    session      = None
    last_ear     = 0.0
    last_spoof   = 0.5
    last_sm      = "LBP-ent"
    result_until = 0.0
    mode         = "idle"

    print("[INFO] Ready.")
    print("       R=start  E=enrol  X=exp  T=train AI  P=report  G=ROC  L=list  D=delete  Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces[:1]:
            shape = predictor(gray, face)
            lp,rp = eye_points(shape)
            cv2.drawContours(frame,[cv2.convexHull(lp)],-1,CYAN,1)
            cv2.drawContours(frame,[cv2.convexHull(rp)],-1,CYAN,1)
            x1,y1,x2,y2=face.left(),face.top(),face.right(),face.bottom()
            cv2.rectangle(frame,(x1,y1),(x2,y2),
                          YELLOW if mode=="running" else GRAY,1)
            if session and not session.complete:
                last_ear,last_spoof,last_sm = session.update(
                    frame,shape,face,predictor,rec_model,db)

        # ── HUD ───────────────────────────────────────────────────────────────
        if session:
            scol = GREEN if "CNN" in last_sm else (GREEN if "cal" in last_sm else ORANGE)
            draw_score_bar(frame,"EAR",  session.blink.score(),22,CYAN)
            draw_score_bar(frame,last_sm,last_spoof,           54,scol)
            draw_score_bar(frame,"CHALL",session.challenge.score,86,YELLOW)
            put(frame,
                f"Blinks:{session.blink.count}/{session.blink.blinks_required}"
                f"  Chall:{len(session.challenge._challenges)}",
                (10,112),WHITE,0.42,1)

            if session.exp_case:
                put(frame,f"[EXP:{session.exp_case.upper()}]",
                    (frame.shape[1]-165,24),YELLOW,0.55,1)

            if not session.challenge.done:
                put(frame,f">> {session.challenge.current_instruction()} <<",
                    (frame.shape[1]//2-130,34),YELLOW,0.75,2)
                put(frame,f"{session.challenge.remaining:.1f}s",
                    (frame.shape[1]-65,34),YELLOW,0.65,2)

            if session.complete:
                draw_result_banner(frame,session.result)
                session.finalize()
                draw_ai_badges(frame, session.result.texture_method,
                               session.result.fusion_method,
                               session.id_method)
                if session.identity:
                    col = GREEN if session.identity!="Unknown" else RED
                    put(frame,f"ID:{session.identity} ({session.id_conf:.0%}) [{session.id_method}]",
                        (10,frame.shape[0]-74),col,0.6,2)
                if result_until>0 and time.time()>result_until:
                    session=None; mode="idle"; result_until=0.0

        elif mode=="experiment_pick":
            h,w=frame.shape[:2]
            cv2.rectangle(frame,(w//2-170,h//2-60),(w//2+170,h//2+60),DARK,-1)
            put(frame,"Select test case:",(w//2-135,h//2-36),WHITE,0.65,2)
            put(frame,"1  Real person",  (w//2-125,h//2-10),GREEN, 0.60,1)
            put(frame,"2  Printed photo",(w//2-125,h//2+14),YELLOW,0.60,1)
            put(frame,"3  Video replay", (w//2-125,h//2+38),RED,   0.60,1)

        elif mode in ("typing_enrol","typing_delete"):
            prompt=("ENROL: check PowerShell terminal to type name"
                    if mode=="typing_enrol"
                    else "DELETE: check PowerShell terminal to type name")
            draw_typing_overlay(frame,prompt)

        elif mode=="training":
            h,w=frame.shape[:2]
            cv2.rectangle(frame,(0,h//2-30),(w,h//2+30),DARK,-1)
            put(frame,"Training AI models — check terminal...",(w//2-200,h//2+8),YELLOW,0.65,2)

        else:
            put(frame,"Press R to start liveness check",(10,34),GRAY,0.6,1)

        draw_ai_status(frame)
        draw_controls(frame,mode)
        cv2.imshow(WINDOW,frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if mode in ("typing_enrol","typing_delete"):
            if mode=="typing_enrol":
                name=input("\nEnter name to enrol (blank to cancel): ").strip()
                if name: db=run_enrolment(name,detector,predictor,rec_model,cap)
            else:
                name=input("\nEnter name to DELETE (blank to cancel): ").strip()
                if name: db=delete_user(name)
            cv2.setWindowProperty(WINDOW,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW,frame); cv2.waitKey(300)
            mode="idle"; session=None; result_until=0.0
            continue

        if key==ord("q"): break

        elif key==ord("r") and mode in ("idle","done"):
            session=Session(); session.start()
            mode="running"; result_until=0.0
            print("[INFO] Session started.")

        elif key==ord("x") and mode in ("idle","done"):
            mode="experiment_pick"

        elif mode=="experiment_pick":
            case=None
            if   key==ord("1"): case="real"
            elif key==ord("2"): case="photo"
            elif key==ord("3"): case="video"
            elif key==27:       mode="idle"
            if case:
                session=Session(exp_case=case); session.start()
                mode="running"; result_until=0.0
                print(f"[EXP] case={case}")

        elif key==ord("e") and mode in ("idle","done"):
            mode="typing_enrol"

        elif key==ord("d") and mode in ("idle","done"):
            mode="typing_delete"

        elif key==ord("t") and mode in ("idle","done"):
            # Train AI models
            print("\n[TRAIN] Training Random Forest fusion model...")
            import subprocess, sys
            subprocess.run([sys.executable,"train_fusion.py","--eval"])
            # Reload RF model
            from liveness import _load_rf
            _load_rf()
            print("[TRAIN] To train CNN: python train_cnn.py --collect --train")

        elif key==ord("l"): list_users()
        elif key==ord("p"): print_attendance(); print_report()
        elif key==ord("g"): generate_roc()

        if session and session.complete and result_until==0.0:
            result_until=time.time()+config.RESULT_HOLD_SECS
            mode="done"

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited.")


if __name__=="__main__":
    main()