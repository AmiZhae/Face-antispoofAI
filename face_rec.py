import os
import csv
import pickle
from datetime import datetime

import cv2
import numpy as np

import config

DB_PATH        = os.path.join(config.KNOWN_FACES_DIR, "face_db.pkl")
ATTEND_HEADERS = ["timestamp", "date", "time", "name", "liveness_score",
                  "confidence", "fusion_method", "texture_method"]

GREEN  = (0,  220,  80)
YELLOW = (0,  200, 255)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)

# ── Try importing DeepFace ────────────────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("[DEEPFACE] ArcFace face recognition ready.")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[DEEPFACE] Not installed — install: pip install deepface tensorflow")
    print("           Falling back to dlib ResNet.")


# ══════════════════════════════════════════════════════════════════════════════
# Database helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH,"rb") as f:
        return pickle.load(f)

def save_db(db):
    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
    with open(DB_PATH,"wb") as f:
        pickle.dump(db, f)

def list_users():
    db = load_db()
    if not db:
        print("No users enrolled yet."); return
    print("\nEnrolled users:")
    for name, data in db.items():
        n = len(data.get("images",[]) or data) if isinstance(data,dict) else len(data)
        person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
        has_imgs = os.path.exists(person_dir)
        print(f"  {name:<24} ({n} samples)  {'[images saved]' if has_imgs else '[pkl only]'}")
    print()


def export_face_images():
    """Export face images from pkl to visible JPG files in known_faces/name/."""
    db = load_db()
    if not db:
        print("No users enrolled."); return
    for name, data in db.items():
        imgs = data.get("images", []) if isinstance(data, dict) else []
        if not imgs:
            print(f"[EXPORT] '{name}' has no stored images (descriptors only).")
            continue
        person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(person_dir, f"{name}_{i:03d}.jpg"), img)
        print(f"[EXPORT] '{name}' → {len(imgs)} images saved to {person_dir}/")

def delete_user(name):
    db = load_db()
    if name in db:
        del db[name]; save_db(db)
        print(f"[DELETE] Removed '{name}'.")
    else:
        print(f"[DELETE] '{name}' not found.")
    return load_db()


# ══════════════════════════════════════════════════════════════════════════════
# Enrolment — stores face images for DeepFace + embeddings for dlib fallback
# ══════════════════════════════════════════════════════════════════════════════
def enrol_webcam(name, detector, predictor, rec_model, cap, n_samples=20):
    """
    Capture face samples. Stores:
      - Raw BGR face crops for DeepFace ArcFace
      - dlib 128-D descriptors as fallback
    Also captures LBP calibration samples.
    """
    from liveness import extract_face_roi, save_lbp_calibration, reload_lbp_calibration

    db    = load_db()
    imgs  = []   # BGR face crops for DeepFace
    descs = []   # dlib descriptors for fallback
    rois  = []   # LBP calibration

    print(f"[ENROL] Enrolling '{name}' — {n_samples} samples...")

    while len(imgs) < n_samples:
        ret, frame = cap.read()
        if not ret: break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        import dlib
        dlib_det = dlib.get_frontal_face_detector()
        faces = dlib_det(gray, 0)

        for face in faces[:1]:
            # Crop face for DeepFace
            x1=max(0,face.left()); y1=max(0,face.top())
            x2=min(frame.shape[1],face.right()); y2=min(frame.shape[0],face.bottom())
            crop = cv2.resize(frame[y1:y2,x1:x2], (160,160))
            imgs.append(crop)

            # dlib descriptor for fallback (only if model available)
            if rec_model is not None:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shape = predictor(rgb, face)
                desc  = np.array(rec_model.compute_face_descriptor(rgb, shape))
                descs.append(desc)

            # LBP calibration ROI
            roi = extract_face_roi(frame, face)
            if roi is not None: rois.append(roi)

            cv2.rectangle(frame,(x1,y1),(x2,y2),GREEN,2)

        pct = int(len(imgs)/n_samples*100)
        cv2.putText(frame, f"Enrolling: {name}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        cv2.putText(frame, f"Captured: {len(imgs)}/{n_samples}  ({pct}%)",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        cv2.rectangle(frame,(10,70),(310,84),(60,60,60),-1)
        cv2.rectangle(frame,(10,70),(10+int(pct*3),84),GREEN,-1)
        cv2.putText(frame, "Press Q to finish early", (10,104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        cv2.imshow("Liveness Detection FYP", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if imgs:
        db[name] = {"images": imgs, "descriptors": descs}
        save_db(db)
        # Also save face crops as visible JPG files inside known_faces/name/
        person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(person_dir, f"{name}_{i:03d}.jpg"), img)
        print(f"[ENROL] Saved {len(imgs)} samples for '{name}'.")
        print(f"[ENROL] Face images saved to {person_dir}/")
    if rois:
        save_lbp_calibration(rois)
        reload_lbp_calibration()
    return load_db()


# ══════════════════════════════════════════════════════════════════════════════
# Recognition — DeepFace ArcFace with dlib fallback
# ══════════════════════════════════════════════════════════════════════════════
def recognise(frame, face_rect, predictor, rec_model, db):
    """
    Match face using DeepFace ArcFace (primary) or dlib ResNet (fallback).
    Returns (name, confidence, method).
    """
    if not db:
        return "Unknown", 0.0, "none"

    x1=max(0,face_rect.left()); y1=max(0,face_rect.top())
    x2=min(frame.shape[1],face_rect.right()); y2=min(frame.shape[0],face_rect.bottom())
    if x2<=x1 or y2<=y1:
        return "Unknown", 0.0, "none"

    face_crop = cv2.resize(frame[y1:y2,x1:x2], (160,160))

    # ── DeepFace ArcFace ──────────────────────────────────────────────────────
    if DEEPFACE_AVAILABLE:
        best_name = "Unknown"
        best_dist = float("inf")

        for name, data in db.items():
            stored_imgs = data.get("images",[]) if isinstance(data,dict) else []
            # Compare against up to 5 stored images, take the best
            for stored_img in stored_imgs[:5]:
                try:
                    result = DeepFace.verify(
                        face_crop, stored_img,
                        model_name     = config.DEEPFACE_MODEL,
                        detector_backend="skip",   # face already cropped
                        enforce_detection=False,
                    )
                    d = result.get("distance", 1.0)
                    if d < best_dist:
                        best_dist = d
                        best_name = name
                except Exception:
                    continue

        if best_dist > config.DEEPFACE_THRESHOLD:
            return "Unknown", 0.0, "ArcFace"

        conf = float(np.clip(1.0 - best_dist/config.DEEPFACE_THRESHOLD, 0, 1))
        return best_name, conf, "ArcFace"

    # ── dlib fallback ─────────────────────────────────────────────────────────
    if rec_model is None:
        return "Unknown", 0.0, "none"
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    shape = predictor(rgb, face_rect)
    desc  = np.array(rec_model.compute_face_descriptor(rgb, shape))

    best_name = "Unknown"
    best_dist = float("inf")

    for name, data in db.items():
        stored = data.get("descriptors",[]) if isinstance(data,dict) else data
        dists  = sorted([np.linalg.norm(desc-d) for d in stored])
        avg    = float(np.mean(dists[:5])) if dists else 1.0
        if avg < best_dist:
            best_dist = avg; best_name = name

    if best_dist > 0.45:
        return "Unknown", 0.0, "dlib"
    conf = float(np.clip(1.0-best_dist/0.45, 0, 1))
    return best_name, conf, "dlib"


# ══════════════════════════════════════════════════════════════════════════════
# Attendance
# ══════════════════════════════════════════════════════════════════════════════
def log_attendance(name, liveness_score, confidence, fusion_method="", texture_method=""):
    now = datetime.now()
    row = {
        "timestamp"      : now.isoformat(timespec="seconds"),
        "date"           : now.strftime("%Y-%m-%d"),
        "time"           : now.strftime("%H:%M:%S"),
        "name"           : name,
        "liveness_score" : f"{liveness_score:.3f}",
        "confidence"     : f"{confidence:.3f}",
        "fusion_method"  : fusion_method,
        "texture_method" : texture_method,
    }
    exists = os.path.exists(config.ATTENDANCE_CSV)
    with open(config.ATTENDANCE_CSV,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=ATTEND_HEADERS)
        if not exists: w.writeheader()
        w.writerow(row)
    print(f"[ATTENDANCE] {name} at {row['time']} "
          f"(fusion={fusion_method}, texture={texture_method})")

def print_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(config.ATTENDANCE_CSV):
        print("No attendance records yet."); return
    with open(config.ATTENDANCE_CSV,newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["date"]==today]
    if not rows: print(f"No attendance for {today}."); return
    print(f"\n── Attendance {today} {'─'*30}")
    print(f"{'Name':<20}{'Time':>8}{'Score':>8}{'Conf':>7}{'Fusion':>12}{'Texture':>10}")
    print("─"*65)
    for r in rows:
        print(f"{r['name']:<20}{r['time']:>8}"
              f"{float(r['liveness_score']):>8.2f}"
              f"{float(r['confidence']):>7.2f}"
              f"{r.get('fusion_method',''):>12}"
              f"{r.get('texture_method',''):>10}")
    print()