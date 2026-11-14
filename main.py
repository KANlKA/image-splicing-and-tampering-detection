"""
Image Splicing and Tampering Detection (Siamese Neural Network)

Folder layout expected next to this file:
    original/   1.jpg, 2.jpg, ...   (untampered images)
    edited/     1.jpg, 2.jpg, ...   (tampered versions, same file stems)

Label convention: 1 = same (untampered), 0 = tampered.

Usage:
    python main_v2.py             # train (or load saved model), evaluate on held-out test set
    python main_v2.py --retrain   # ignore the saved model and train again
    python main_v2.py --cv        # 5-fold cross-validation (most reliable number to quote)

Why this version differs from a basic script:
  * Train / validation / test split is done BY IMAGE before pairs are built,
    so the same original image never appears in two splits (no leakage).
  * Early stopping uses the validation set; reported metrics use the untouched test set.
  * Positive pairs are (original, lightly perturbed original) so the task is not trivial.
  * Regularization: BatchNorm, Dropout, global average pooling, flips as augmentation.
"""
import argparse
import glob
import os
import random
import re

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import callbacks, layers, models

# ----------------------------- configuration ------------------------------
SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 16
MAX_EPOCHS = 60
AUG_COPIES = 4          # how many augmented copies of each training image pair set
ORIGINAL_DIR = "original"
EDITED_DIR = "edited"
DIFF_DIR = "differences"
MODEL_PATH = "siamese_v2.keras"   # new name so it never clashes with the old script's model file
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------ data loading ------------------------------
def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32") / 255.0


def trailing_number(stem):
    """'original12' -> 12, 'edited_7' -> 7, '3' -> 3, 'photo' -> None."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def index_edited():
    """Map pair number -> edited file path, e.g. edited1.jpeg -> 1."""
    index = {}
    for path in glob.glob(os.path.join(EDITED_DIR, "*")):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() not in EXTS:
            continue
        num = trailing_number(stem)
        if num is not None:
            index[num] = path
    return index


def sorted_originals():
    """Original image paths sorted numerically (1, 2, 3, ... 10, 11), not as text."""
    paths = [p for p in glob.glob(os.path.join(ORIGINAL_DIR, "*"))
             if os.path.splitext(p)[1].lower() in EXTS]

    def key(p):
        num = trailing_number(os.path.splitext(os.path.basename(p))[0])
        return (num if num is not None else 10 ** 9, p)

    return sorted(paths, key=key)


def load_dataset():
    originals, editeds, names = [], [], []
    edited_index = index_edited()
    for path in sorted_originals():
        stem, ext = os.path.splitext(os.path.basename(path))
        # match by the number at the end of the file name: original7 <-> edited7
        edited_path = edited_index.get(trailing_number(stem))
        if edited_path is None:
            print(f"[skip] no edited version found for {path}")
            continue
        a, b = cv2.imread(path), cv2.imread(edited_path)
        if a is None or b is None:
            print(f"[skip] could not read {path} or {edited_path}")
            continue
        originals.append(preprocess(a))
        editeds.append(preprocess(b))
        names.append((stem, path, edited_path))
    if not names:
        sample = sorted(os.listdir(EDITED_DIR))[:5] if os.path.isdir(EDITED_DIR) else "folder missing"
        raise SystemExit("No image pairs found. Run this from the project folder.\n"
                         f"  edited/ contains (first 5): {sample}\n"
                         "  File names must end in the same number, e.g. original7.jpeg and edited7.jpeg.")
    return np.array(originals), np.array(editeds), names


# ------------------------------ pair creation -----------------------------
def perturb(img, rng):
    """Light brightness change + noise so a 'same' pair is not an exact copy."""
    out = img * rng.uniform(0.9, 1.1) + rng.normal(0, 0.01, img.shape)
    return np.clip(out, 0, 1).astype("float32")


def random_flip(a, b, rng):
    """Apply the same random flips to both images of a pair."""
    if rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    if rng.random() < 0.5:
        a, b = a[::-1], b[::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def make_pairs(originals, editeds, idx, rng, copies=1, augment=False):
    """For every image index build one positive (same=1) and one negative (tampered=0) pair."""
    X1, X2, y = [], [], []
    for _ in range(copies):
        for i in idx:
            orig, edit = originals[i], editeds[i]
            for second, label in ((perturb(orig, rng), 1.0), (edit, 0.0)):
                a, b = orig, second
                if augment:
                    a, b = random_flip(a, b, rng)
                X1.append(a)
                X2.append(b)
                y.append(label)
    X1, X2, y = np.array(X1), np.array(X2), np.array(y, dtype="float32")
    perm = rng.permutation(len(y))
    return X1[perm], X2[perm], y[perm]


def split_ids(n):
    """Split image indices ~70/15/15 into train / val / test (by image, not by pair)."""
    ids = np.arange(n)
    train_val, test = train_test_split(ids, test_size=0.15, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.18, random_state=SEED)
    return train, val, test


# --------------------------------- model ----------------------------------
@tf.keras.utils.register_keras_serializable(package="tamper")
class DistanceLayer(layers.Layer):
    """Absolute difference between the two embeddings."""

    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)


def build_base(input_shape):
    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in (32, 64, 128):          # 3 convolutional layers
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(inp, x, name="base_cnn")


def build_siamese(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base = build_base(input_shape)
    in_a = layers.Input(shape=input_shape)
    in_b = layers.Input(shape=input_shape)
    dist = DistanceLayer()([base(in_a), base(in_b)])
    x = layers.Dense(32, activation="relu")(dist)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # 1 = same, 0 = tampered
    model = models.Model([in_a, in_b], out, name="siamese")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(train, val, verbose=1):
    X1tr, X2tr, ytr = train
    X1va, X2va, yva = val
    model = build_siamese()
    stop = callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                   restore_best_weights=True)
    hist = model.fit([X1tr, X2tr], ytr,
                     validation_data=([X1va, X2va], yva),
                     epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[stop], verbose=verbose)
    return model, hist


# ------------------------------- evaluation -------------------------------
def evaluate(model, test):
    X1, X2, y = test
    y = y.astype(int)
    prob = model.predict([X1, X2], verbose=0).ravel()
    pred = (prob > 0.5).astype(int)

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
    print("\n===== TEST SET RESULTS =====")
    print(f"Test pairs: {len(y)}  (tampered={int((y == 0).sum())}, same={int((y == 1).sum())})")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(classification_report(y, pred, target_names=["Tampered", "Same"], zero_division=0))

    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Tampered", "Same"], yticklabels=["Tampered", "Same"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y, prob)
        plt.figure(figsize=(4.5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=150)
        plt.close()
    return acc, auc


def plot_history(hist):
    np.save("training_history.npy", hist.history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="validation")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()


# --------------------------- cross-validation -----------------------------
def cross_validate(originals, editeds, folds=5):
    n = len(originals)
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for fold, (train_val_ids, test_ids) in enumerate(kf.split(np.arange(n)), 1):
        train_ids, val_ids = train_test_split(train_val_ids, test_size=0.15,
                                              random_state=SEED)
        rng = np.random.default_rng(SEED + fold)
        train = make_pairs(originals, editeds, train_ids, rng, AUG_COPIES, augment=True)
        val = make_pairs(originals, editeds, val_ids, rng)
        test = make_pairs(originals, editeds, test_ids, rng)

        tf.keras.backend.clear_session()
        model, _ = train_model(train, val, verbose=0)

        X1, X2, y = test
        prob = model.predict([X1, X2], verbose=0).ravel()
        acc = accuracy_score(y.astype(int), (prob > 0.5).astype(int))
        auc = roc_auc_score(y.astype(int), prob)
        accs.append(acc)
        aucs.append(auc)
        print(f"Fold {fold}/{folds}: test pairs={len(y)}  acc={acc:.3f}  auc={auc:.3f}")

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print(f"Images: {n}  | Pairs per image: 2 (1 same + 1 tampered)")
    print(f"Accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"ROC-AUC:  {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")


# --------------------- highlighting the tampered regions ------------------
def highlight_differences(orig_path, edit_path, out_path, min_area=100):
    a, b = cv2.imread(orig_path), cv2.imread(edit_path)
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    diff = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, regions = b.copy(), 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            regions += 1
    cv2.imwrite(out_path, out)
    return regions


def compare_pairs(model, originals, editeds, names, picks):
    os.makedirs(DIFF_DIR, exist_ok=True)
    for i in picks:
        stem, orig_path, edit_path = names[i]
        print(f"\n=== Comparing pair {stem} ===")
        out_path = os.path.join(DIFF_DIR, f"highlighted_{stem}.jpeg")
        regions = highlight_differences(orig_path, edit_path, out_path)
        print(f"Traditional comparison saved to {out_path} ({regions} region(s) found)")
        score = float(model.predict([originals[i][None], editeds[i][None]], verbose=0)[0][0])
        print(f"ML similarity score: {score * 100:.2f}%")
        print("ML VERDICT:", "SAME" if score > 0.5 else "EDITED")


# ----------------------------------- main ---------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="run 5-fold cross-validation")
    parser.add_argument("--retrain", action="store_true", help="ignore saved model")
    args = parser.parse_args()

    set_seed()
    print("Loading images...")
    originals, editeds, names = load_dataset()
    n = len(names)
    print(f"Found {n} original/edited image pairs.")
    if n < 10:
        raise SystemExit("Need at least 10 image pairs to split the data properly.")

    if args.cv:
        cross_validate(originals, editeds)
        return

    train_ids, val_ids, test_ids = split_ids(n)
    rng = np.random.default_rng(SEED)
    train = make_pairs(originals, editeds, train_ids, rng, AUG_COPIES, augment=True)
    val = make_pairs(originals, editeds, val_ids, rng)
    test = make_pairs(originals, editeds, test_ids, rng)
    print(f"Images -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Pairs  -> train: {len(train[2])} (with augmentation), "
          f"val: {len(val[2])}, test: {len(test[2])}")

    model = None
    if os.path.exists(MODEL_PATH) and not args.retrain:
        try:
            print("Loading saved model...")
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as exc:  # stale / incompatible file from an older script
            print(f"Could not load {MODEL_PATH} ({type(exc).__name__}); training a new model instead.")
            model = None
    if model is None:
        print("Training new model...")
        model, hist = train_model(train, val)
        model.save(MODEL_PATH)
        plot_history(hist)

    evaluate(model, test)

    picks = sorted({0, n // 2, n - 1})
    compare_pairs(model, originals, editeds, names, picks)

    print(f"\nSaved: {MODEL_PATH}, training_history.png/.npy, "
          "confusion_matrix.png, roc_curve.png, differences/")


if __name__ == "__main__":
    main()