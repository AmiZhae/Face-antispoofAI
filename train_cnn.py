import argparse
import os
import time
import cv2
import numpy as np

import config

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "cnn_training_data"
REAL_DIR   = os.path.join(DATA_DIR, "real")
FAKE_DIR   = os.path.join(DATA_DIR, "fake")
MODEL_PATH = config.CNN_MODEL_PATH
IMG_SIZE   = config.CNN_INPUT_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# Data Collection
# ══════════════════════════════════════════════════════════════════════════════
def collect_samples(label: str, n_samples: int = 200):
    """
    Collect face ROI samples from the webcam.

    label     : 'real' or 'fake'
    n_samples : how many images to capture (200 per class recommended)

    Instructions:
      real → sit in front of camera normally
      fake → hold a printed photo or play a video to the camera
    """
    import dlib
    save_dir = REAL_DIR if label == "real" else FAKE_DIR
    os.makedirs(save_dir, exist_ok=True)

    detector = dlib.get_frontal_face_detector()
    cap      = cv2.VideoCapture(0)
    saved    = 0

    print(f"\n[COLLECT] Label='{label}' | Target={n_samples} samples")
    if label == "real":
        print("         Look at the camera naturally. Vary your angle slightly.")
    else:
        print("         Hold a PRINTED PHOTO or play a VIDEO to the camera.")
    print("         Press S to save a frame manually, or A for auto-capture.")
    print("         Press Q to finish early.\n")

    auto_mode  = False
    last_saved = 0

    while saved < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        roi_saved = False
        for face in faces[:1]:
            x1 = max(0, face.left());   y1 = max(0, face.top())
            x2 = min(frame.shape[1], face.right())
            y2 = min(frame.shape[0], face.bottom())
            roi = cv2.resize(frame[y1:y2, x1:x2], IMG_SIZE)

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

            if auto_mode and time.time() - last_saved > 0.15:
                fname = os.path.join(save_dir, f"{label}_{saved:04d}.jpg")
                cv2.imwrite(fname, roi)
                saved += 1
                last_saved = time.time()
                roi_saved = True

        color = (0,200,255) if label == "real" else (0,0,255)
        cv2.putText(frame, f"Label: {label.upper()}  Saved: {saved}/{n_samples}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "A=auto  S=manual  Q=quit",
                    (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        if auto_mode:
            cv2.putText(frame, "AUTO ON", (10,86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        pct = int(saved / n_samples * 100)
        cv2.rectangle(frame, (10,96),(310,110),(40,40,40),-1)
        cv2.rectangle(frame, (10,96),(10+int(pct*3),110),(0,200,100),-1)

        cv2.imshow("CNN Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"[AUTO] {'ON' if auto_mode else 'OFF'}")
        elif key == ord("s") and faces:
            fname = os.path.join(save_dir, f"{label}_{saved:04d}.jpg")
            cv2.imwrite(fname, roi)
            saved += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[COLLECT] Saved {saved} '{label}' samples to {save_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Training curve plot
# ══════════════════════════════════════════════════════════════════════════════
def _plot_training_curve():
    import csv
    import matplotlib.pyplot as plt

    if not os.path.exists('training_log.csv'):
        return

    epochs, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    with open('training_log.csv') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']) + 1)
            train_acc.append(float(row['accuracy']) * 100)
            val_acc.append(float(row['val_accuracy']) * 100)
            train_loss.append(float(row['loss']))
            val_loss.append(float(row['val_loss']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
    fig.suptitle('MobileNetV2 CNN Training History — Spoof Detection',
                 fontsize=14, fontweight='bold')

    # Accuracy
    ax1.plot(epochs, train_acc, color='#2563EB', lw=2, marker='o',
             markersize=4, label='Training Accuracy')
    ax1.plot(epochs, val_acc, color='#16A34A', lw=2, marker='s',
             markersize=4, label='Validation Accuracy', ls='--')
    best_ep = val_loss.index(min(val_loss))
    ax1.axvline(x=epochs[best_ep], color='#DC2626', lw=1.2, ls=':', alpha=0.7)
    ax1.annotate(f'Best epoch: {epochs[best_ep]}\nVal acc: {val_acc[best_ep]:.1f}%',
                 xy=(epochs[best_ep], val_acc[best_ep]),
                 xytext=(epochs[best_ep]+1.2, val_acc[best_ep]-10),
                 fontsize=8.5, color='#DC2626',
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.0))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_ylim(50, 105); ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', color='#EEEEEE', lw=0.8)

    # Loss
    ax2.plot(epochs, train_loss, color='#2563EB', lw=2, marker='o',
             markersize=4, label='Training Loss')
    ax2.plot(epochs, val_loss, color='#16A34A', lw=2, marker='s',
             markersize=4, label='Validation Loss', ls='--')
    ax2.axvline(x=epochs[best_ep], color='#DC2626', lw=1.2, ls=':', alpha=0.7)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', color='#EEEEEE', lw=0.8)

    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[SAVED]  Training curve saved to training_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("[ERROR] Install TensorFlow: pip install tensorflow")
        return

    print("\n[TRAIN] Loading images...")

    X, y = [], []
    for label, folder in [(1, REAL_DIR), (0, FAKE_DIR)]:
        if not os.path.exists(folder):
            print(f"[ERROR] Folder not found: {folder} — run --collect first.")
            return
        files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label)
        print(f"  {label} ({'real' if label==1 else 'fake'}): {len(files)} images")

    if len(X) < 20:
        print("[ERROR] Not enough training data. Collect at least 20 images per class.")
        return

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n[TRAIN] Train={len(X_train)}  Val={len(X_val)}")

    # ── Build model: MobileNetV2 + custom head ────────────────────────────────
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",   # pre-trained on ImageNet
    )
    # Freeze base layers — only train the head
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)   # 1 = real, 0 = fake

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[TRAIN] Model built. Trainable params: "
          f"{sum(p.numpy().size for p in model.trainable_variables):,}")

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        CSVLogger('training_log.csv', append=False),
    ]

    print("[TRAIN] Training (Phase 1 — head only)...")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=20, batch_size=16,
              callbacks=callbacks, verbose=1)

    # ── Fine-tune top layers of base ──────────────────────────────────────────
    print("\n[TRAIN] Fine-tuning top 30 layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=10, batch_size=16,
              callbacks=callbacks, verbose=1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULT] Validation accuracy: {acc*100:.1f}%  Loss: {loss:.4f}")
    print(f"[SAVED]  Model saved to {MODEL_PATH}")
    print(f"[SAVED]  Training log saved to training_log.csv")

    # ── Plot training curve ───────────────────────────────────────────────────
    _plot_training_curve()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CNN spoof detector trainer")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train",   action="store_true", help="Train the CNN model")
    parser.add_argument("--samples", type=int, default=200, help="Samples per class")
    args = parser.parse_args()

    if not args.collect and not args.train:
        parser.print_help()
        return

    if args.collect:
        print("=" * 50)
        print("Step 1/2: Collect REAL face samples")
        print("=" * 50)
        collect_samples("real", args.samples)

        print("\n" + "=" * 50)
        print("Step 2/2: Collect FAKE samples (photo/video)")
        print("=" * 50)
        collect_samples("fake", args.samples)

    if args.train:
        train_model()


if __name__ == "__main__":
    main()