import os
import re
import numpy as np
import soundfile as sf

from viterbi import RunViterbi
from helper_funcs.custom_demon import buildDemonMat, plot_demon_matrix
from plotWav import main_plotWav


# =========================
# CONFIG
# =========================
data_path = "../data/test_data/"
anotations_path = "../data/test_data/"
START_INDEX = 84  # 1-based index
LOG_FILE = "./inference_output_test-data.txt"  # <-- change this to your log file path

write_anotations = True
manual_anotations = True

ONLY_ERROR_TYPES = {"FP", "FN"}  # only re-annotate false positives/false negatives

os.makedirs(anotations_path, exist_ok=True)


def read_text_safely(file_path):
    encodings = ["utf-16", "utf-8-sig", "utf-8", "cp1255", "cp1252"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()

            if "Presence probability" in text:
                print(f"Using log encoding: {enc}")
                return text

        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Could not read log file: {file_path}")


def parse_inference_log(log_file):
    text = read_text_safely(log_file)
    blocks = text.split("---------------------------------------------")

    records = {}

    for block in blocks:
        if "Audio file:" not in block:
            continue

        audio_match = re.search(r"Audio file:\s*(.+)", block)
        pred_match = re.search(r"Vessel detected:\s*(True|False)", block)
        prob_match = re.search(r"Presence probability:\s*([0-9.]+)", block)
        pred_col_match = re.search(r"Predicted column:\s*(-?\d+)", block)
        gt_col_match = re.search(r"ground truth column \(viterbi mean\):\s*(-?\d+)", block)
        gt_presence_match = re.search(r"ground truth presence:\s*([01])", block)

        if not all([
            audio_match,
            pred_match,
            prob_match,
            pred_col_match,
            gt_col_match,
            gt_presence_match,
        ]):
            continue

        audio_path = audio_match.group(1).strip()
        filename = os.path.basename(audio_path.replace("\\", "/"))

        pred_presence = pred_match.group(1) == "True"
        gt_presence = int(gt_presence_match.group(1))

        if pred_presence and gt_presence == 1:
            error_type = "TP"
        elif pred_presence and gt_presence == 0:
            error_type = "FP"
        elif (not pred_presence) and gt_presence == 0:
            error_type = "TN"
        elif (not pred_presence) and gt_presence == 1:
            error_type = "FN"
        else:
            error_type = "UNKNOWN"

        records[filename] = {
            "filename": filename,
            "audio_path_from_log": audio_path,
            "presence_probability": float(prob_match.group(1)),
            "pred_presence": int(pred_presence),
            "vessel_detected": pred_presence,
            "predicted_column": int(pred_col_match.group(1)),
            "gt_column": int(gt_col_match.group(1)),
            "gt_presence": gt_presence,
            "error_type": error_type,
        }

    return records


def annotate_file(filename, record):
    audio_path = os.path.join(data_path, filename)

    print("\n---------------------------------------------")
    print(f"File: {filename}")
    print(f"Error type: {record['error_type']}")
    print(f"Presence probability: {record['presence_probability']:.4f}")
    print(f"Model vessel detected: {record['vessel_detected']}")
    print(f"Model predicted column: {record['predicted_column']}")
    print(f"Old GT column: {record['gt_column']}")
    print(f"Old GT presence: {record['gt_presence']}")
    print("---------------------------------------------")

    data, fs = sf.read(audio_path)
    sig = data[:, 0] if data.ndim > 1 else data

    demonMat, snrVec = buildDemonMat(
        sig=sig,
        fs=fs,
        BlockLen=1,
        fpass=100,
        fstop=1500,
        fpassDemon=300,
    )

    print("DemonMat shape:", demonMat.shape)
    print("SNRVec shape:", snrVec.shape)

    try:
        DetectFlag, tracker = RunViterbi(demonMat, Th=0)
        print(f"Viterbi median tracker: {int(np.median(tracker) - 1)}")
    except Exception as e:
        print(f"[WARNING] Viterbi failed: {e}")

    if manual_anotations:
        # main_plotWav(audio_path) 
        plot_demon_matrix(demonMat)

        print("\nManual annotation:")
        print("Use presence=0 and column=-1 for no vessel.")
        print("Use presence=1 and column=<column_index> for vessel.")
        print("Press Enter without typing to keep old value.")

        presence_in = input(f"Enter boat_presence [{record['gt_presence']}]: ").strip()
        column_in = input(f"Enter column [{record['gt_column']}]: ").strip()

        new_presence = record["gt_presence"] if presence_in == "" else int(presence_in)
        new_column = record["gt_column"] if column_in == "" else int(column_in)

    else:
        new_presence = record["gt_presence"]
        new_column = record["gt_column"]

    if new_presence == 0:
        new_column = -1

    if new_presence == 1 and new_column < 0:
        print("[WARNING] presence=1 but column<0. Please fix manually.")
        new_column = int(input("Enter valid column: ").strip())

    if write_anotations:
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(anotations_path, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(f"{new_presence} {new_column}")

        print(f"Saved annotation: {txt_path} -> {new_presence} {new_column}")


def main():
    records = parse_inference_log(LOG_FILE)

    print(f"Parsed records from log: {len(records)}")

    selected = [
        r for r in records.values()
        if r["error_type"] in ONLY_ERROR_TYPES
    ]

    print(f"Selected records for re-annotation: {len(selected)}")
    print(f"FP count: {sum(1 for r in selected if r['error_type'] == 'FP')}")
    print(f"FN count: {sum(1 for r in selected if r['error_type'] == 'FN')}")

    selected = sorted(
        selected,
        key=lambda r: (r["error_type"], r["filename"])
    )

    selected_to_process = selected[START_INDEX - 1:]

    for idx, record in enumerate(selected_to_process, start=START_INDEX):
        filename = record["filename"]

        if not filename.endswith(".wav") and not filename.endswith(".flac"):
            continue

        audio_path = os.path.join(data_path, filename)

        if not os.path.exists(audio_path):
            print(f"[WARNING] Audio file missing locally: {audio_path}")
            continue

        print(f"\n[{idx}/{len(selected)}]")
        annotate_file(filename, record)

    print(f"\nDone processing FP/FN files from log.")


if __name__ == "__main__":
    main()