# onnx_inference.py
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import onnxruntime as ort

from helper_funcs.custom_demon import buildDemonMat


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize_matrix(x, eps=1e-6):
    x = x.astype(np.float32)
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def fit_matrix_size(matrix, target_height, target_width):
    h, w = matrix.shape

    # fit width
    if w > target_width:
        matrix = matrix[:, :target_width]
    elif w < target_width:
        pad_right = target_width - w
        matrix = np.pad(matrix, ((0, 0), (0, pad_right)), mode="constant")

    # fit height
    h, w = matrix.shape

    if h > target_height:
        start = (h - target_height) // 2
        matrix = matrix[start:start + target_height, :]
    elif h < target_height:
        pad_total = target_height - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        matrix = np.pad(matrix, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    return matrix


def load_audio(audio_path):
    data, fs = sf.read(str(audio_path))

    if data.ndim > 1:
        sig = data[:, 0]
    else:
        sig = data

    return np.asarray(sig).squeeze(), fs


def prepare_input(
    audio_path,
    target_height,
    target_width,
    normalize,
    block_len,
    fpass,
    fstop,
    fpass_demon,
):
    sig, fs = load_audio(audio_path)

    demon_mat, snr_vec = buildDemonMat(
        sig=sig,
        fs=fs,
        BlockLen=block_len,
        fpass=fpass,
        fstop=fstop,
        fpassDemon=fpass_demon,
    )

    demon_mat = np.asarray(demon_mat, dtype=np.float32)

    matrix = fit_matrix_size(
        demon_mat,
        target_height=target_height,
        target_width=target_width,
    )

    if normalize:
        matrix = normalize_matrix(matrix)

    # ONNX expects: (batch, channel, height, width)
    x = matrix.astype(np.float32)[None, None, :, :]

    return x, demon_mat.shape, matrix.shape


def run_onnx(model_path, x):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})

    presence_logit = outputs[0]
    column_logits = outputs[1]

    return presence_logit, column_logits


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True, help="Path to .wav or .flac file")
    parser.add_argument("--model", required=True, help="Path to .onnx file")

    parser.add_argument("--threshold", type=float, default=0.40)

    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--width", type=int, default=298)

    parser.add_argument("--normalize", action="store_true", default=True)

    parser.add_argument("--block-len", type=float, default=1.0)
    parser.add_argument("--fpass", type=float, default=100)
    parser.add_argument("--fstop", type=float, default=1500)
    parser.add_argument("--fpass-demon", type=float, default=300)

    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)

    x, original_shape, input_shape = prepare_input(
        audio_path=audio_path,
        target_height=args.height,
        target_width=args.width,
        normalize=args.normalize,
        block_len=args.block_len,
        fpass=args.fpass,
        fstop=args.fstop,
        fpass_demon=args.fpass_demon,
    )

    presence_logit, column_logits = run_onnx(model_path, x)

    presence_prob = float(sigmoid(presence_logit).squeeze())
    raw_best_column = int(np.argmax(column_logits, axis=1)[0])

    vessel_detected = presence_prob >= args.threshold
    predicted_column = raw_best_column if vessel_detected else -1

    print("\n========== ONNX Vessel Detection ==========")
    print(f"Audio file:              {audio_path}")
    print(f"ONNX model:              {model_path}")
    print(f"Original DEMON shape:    {original_shape}")
    print(f"Model input shape:       {input_shape}")
    print("-------------------------------------------")
    print(f"Presence probability:    {presence_prob:.4f}")
    print(f"Threshold:               {args.threshold:.2f}")
    print(f"Vessel detected:         {vessel_detected}")
    print(f"Predicted column:        {predicted_column}")
    print(f"Raw best column:         {raw_best_column}")
    print("===========================================\n")


if __name__ == "__main__":
    main()