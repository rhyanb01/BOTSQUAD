import os
import csv
import json
import joblib
import numpy as np
from collections import Counter
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -------- Config --------
SAVE_DIR = "saved_numbers"
LABELS_CSV = "labels.csv"
MODEL_PATH = "number_model.pkl"
META_PATH = "number_model_meta.json"
IMG_SIZE = 28
WHITE_THRESHOLD = 240

def load_dataset():
    csv_path = os.path.join(SAVE_DIR, LABELS_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Make sure you have labels.csv in {SAVE_DIR}.")

    X, y = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            label = str(row["label"])
            img_path = os.path.join(SAVE_DIR, filename)
            if not os.path.exists(img_path):
                print(f"[WARN] Missing image file listed in CSV: {img_path}")
                continue
            X.append(preprocess_image(img_path))
            y.append(label)
    if not X:
        raise RuntimeError("No training samples found. Add drawings and labels first.")
    return np.array(X), np.array(y)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img_inv = ImageOps.invert(img)

    arr = np.array(img_inv)
    mask = arr > (255 - WHITE_THRESHOLD)
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        cropped = img_inv
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped = img_inv.crop((x_min, y_min, x_max, y_max))

    w, h = cropped.size
    side = max(w, h)
    padded = Image.new("L", (side, side), color=0)
    padded.paste(cropped, ((side - w) // 2, (side - h) // 2))

    small = padded.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    vec = np.asarray(small, dtype=np.float32) / 255.0
    return vec.flatten()

def can_stratify(labels, test_size=0.2):
    """Return True if every class has at least 2 samples and test split can hold 1+ per class."""
    counts = Counter(labels)
    if min(counts.values()) < 2:
        return False
    # Rough check: expected test samples per class >= 1
    n = len(labels)
    for c in counts.values():
        if c * test_size < 1:
            return False
    return True

def main():
    print("[INFO] Loading dataset...")
    X, y = load_dataset()
    print(f"[INFO] Loaded {len(X)} samples, input dim = {X.shape[1]}.")

    counts = Counter(y)
    print("[INFO] Class distribution:")
    for cls, cnt in sorted(counts.items(), key=lambda kv: kv[0]):
        print(f"  {cls}: {cnt}")

    # Decide how to split
    use_stratified_split = can_stratify(y, test_size=0.2)
    if use_stratified_split:
        print("[INFO] Using stratified train/val split (test_size=0.2).")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("[WARN] Some classes have < 2 samples. Skipping validation split and training on ALL data.")
        X_train, y_train = X, y
        X_val, y_val = None, None

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    print("[INFO] Training model...")
    clf.fit(X_train, y_train)

    if X_val is not None:
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"[INFO] Validation accuracy: {acc:.3f}")
        try:
            print("[INFO] Classification report:")
            print(classification_report(y_val, y_pred))
            print("[INFO] Confusion matrix:")
            print(confusion_matrix(y_val, y_pred))
        except Exception:
            pass
    else:
        print("[INFO] No validation split (insufficient per-class samples).")

    joblib.dump(clf, MODEL_PATH)
    meta = {
        "img_size": IMG_SIZE,
        "white_threshold": WHITE_THRESHOLD,
        "save_dir": SAVE_DIR,
        "classes": clf.classes_.tolist()
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved model to ./{MODEL_PATH}")
    print(f"[INFO] Saved meta  to ./{META_PATH}")
    print("[DONE]")

if __name__ == "__main__":
    main()
