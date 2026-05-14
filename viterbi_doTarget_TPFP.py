from viterbi import doTarget_5, RunViterbiTracker
import os
import pickle
import numpy as np
from tqdm import tqdm

# assumes these exist in this file:
# RunViterbiTracker
# doTarget_5


def read_annotation(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    parts = content.split()
    if len(parts) != 2:
        raise ValueError(f"Bad annotation format in {txt_path}: {content}")

    presence = int(parts[0])
    column = int(parts[1])
    return presence, column


def compute_metrics(scores, labels, threshold):
    preds = np.asarray(scores) > threshold
    labels = np.asarray(labels).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    acc = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def main():
    test_demon_mats_path = (
        "C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/"
        "model_training_demon_mat/data/test_demon_mats"
    )

    test_data_path = (
        "C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/"
        "model_training_demon_mat/data/test_data"
    )

    filenames = [
        f for f in os.listdir(test_demon_mats_path)
        if f.lower().endswith(".pkl")
    ]

    print(f"Found {len(filenames)} DEMON files.")

    scores = []
    labels = []
    used_files = []

    for fname in tqdm(filenames, desc="Running Viterbi + doTarget"):
        pkl_path = os.path.join(test_demon_mats_path, fname)

        stem = os.path.splitext(fname)[0]
        txt_path = os.path.join(test_data_path, stem + ".txt")

        if not os.path.exists(txt_path):
            print(f"[WARNING] Missing annotation for {fname}, skipping.")
            continue

        try:
            presence, column = read_annotation(txt_path)

            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)

            if "DemonMat" not in obj:
                print(f"[WARNING] Missing DemonMat in {fname}, skipping.")
                continue

            mat = np.asarray(obj["DemonMat"])

            if mat.ndim != 2:
                print(f"[WARNING] Bad shape in {fname}: {mat.shape}, skipping.")
                continue

            tracker, prob_mat = RunViterbiTracker(mat)

            _, score = doTarget_5(
                tracker=tracker,
                ResWidth=5,
                M=mat,
                ratioMultProbMat=0,
                use_abs=True,
            )

            scores.append(score)
            labels.append(presence)
            used_files.append(fname)

        except Exception as e:
            print(f"[WARNING] Failed processing {fname}: {e}")
            continue

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    print("\nLoaded valid samples:", len(scores))
    print("Positive samples:", int((labels == 1).sum()))
    print("Negative samples:", int((labels == 0).sum()))
    print(
        "Score stats:",
        "min=", float(scores.min()),
        "max=", float(scores.max()),
        "mean=", float(scores.mean()),
        "median=", float(np.median(scores)),
    )

    thresholds = np.linspace(scores.min(), scores.max(), 301)

    results = [compute_metrics(scores, labels, t) for t in thresholds]

    best_f1 = max(results, key=lambda x: x["f1"])
    best_acc = max(results, key=lambda x: x["accuracy"])

    print("\n========== Best threshold by F1 ==========")
    for k, v in best_f1.items():
        print(f"{k}: {v}")

    print("\n========== Best threshold by accuracy ==========")
    for k, v in best_acc.items():
        print(f"{k}: {v}")

    print("\nSuggested thresholds to inspect:")
    for t in [160, 180, 200, 220]:
        if scores.min() <= t <= scores.max():
            m = compute_metrics(scores, labels, t)
            print(
                f"threshold={t} | "
                f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']} | "
                f"precision={m['precision']:.4f} "
                f"recall={m['recall']:.4f} "
                f"f1={m['f1']:.4f} "
                f"acc={m['accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()