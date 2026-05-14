# inference_run.py
import argparse
import os
import torch
import numpy as np
import soundfile as sf

from model import VesselNet
from helper_funcs.custom_demon import buildDemonMat
from dataset import fit_matrix_size, normalize_matrix


def load_audio(audio_path):
    data, fs = sf.read(audio_path)

    if data.ndim > 1:
        sig = data[:, 0]
    else:
        sig = data

    sig = np.asarray(sig).squeeze()
    return sig, fs


def load_model(model_path, device):
    model = VesselNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model,
    audio_path,
    device,
    target_height,
    target_width,
    threshold,
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

    demon_mat = np.asarray(demon_mat)

    # We do not know presence/column yet during inference.
    # Use presence=0 and column=-1 just to apply fixed-size crop/pad.
    matrix, _ = fit_matrix_size(
        matrix=demon_mat,
        presence=0,
        column=-1,
        target_height=target_height,
        target_width=target_width,
    )

    if normalize:
        matrix = normalize_matrix(matrix)

    x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = x.to(device)

    presence_logit, column_logits = model(x)

    presence_prob = torch.sigmoid(presence_logit).item()
    predicted_column = int(column_logits.argmax(dim=1).item())

    vessel_detected = presence_prob >= threshold

    return {
        "audio_path": audio_path,
        "presence_probability": presence_prob,
        "vessel_detected": vessel_detected,
        "predicted_column": predicted_column if vessel_detected else -1,
        "raw_best_column": predicted_column,
        "matrix_shape_before_fit": demon_mat.shape,
        "matrix_shape_after_fit": matrix.shape,
        "snr_mean": float(np.mean(snr_vec)) if len(snr_vec) > 0 else None,
        "snr_max": float(np.max(snr_vec)) if len(snr_vec) > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=False, help="Path to .wav or .flac file", default = None)
    parser.add_argument("--model", required=False, help="Path to model .pth file", default = './models/best_model_so_far.pth')

    parser.add_argument("--target-height", type=int, default=300)
    parser.add_argument("--target-width", type=int, default=298)

    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--normalize", action="store_true", default = True)

    parser.add_argument("--block-len", type=float, default=1.0)
    parser.add_argument("--fpass", type=float, default=100)
    parser.add_argument("--fstop", type=float, default=1500)
    parser.add_argument("--fpass-demon", type=float, default=300)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    # result = run_inference(
    #     model=model,
    #     audio_path=args.audio,
    #     device=device,
    #     target_height=args.target_height,
    #     target_width=args.target_width,
    #     threshold=args.threshold,
    #     normalize=args.normalize,
    #     block_len=args.block_len,
    #     fpass=args.fpass,
    #     fstop=args.fstop,
    #     fpass_demon=args.fpass_demon,
    #     )

    print("\n========== Vessel Detection Result ==========")
    print(f"Device:                  {device}")
    print(f"Decision threshold:      {args.threshold:.2f}")
    # data_path = 'C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/model_training_demon_mat/data/test_data'
    data_path = 'C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/model_training_demon_mat/data/train_data'
    file_list = os.listdir(data_path)
    audio_files = [f for f in file_list if f.endswith('.wav') or f.endswith('.flac')]
    print(f"Found {len(audio_files)} audio files in directory.\n")
    for f in audio_files:
        result = run_inference(
        model=model,
        audio_path=os.path.join(data_path, f),
        device=device,
        target_height=args.target_height,
        target_width=args.target_width,
        threshold=args.threshold,
        normalize=args.normalize,
        block_len=args.block_len,
        fpass=args.fpass,
        fstop=args.fstop,
        fpass_demon=args.fpass_demon,
        )

        # read the corresponding annotation file if it exists
        anotation = f.replace('.wav', '.txt').replace('.flac', '.txt')
        annotation_path = os.path.join(data_path, anotation)
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as ann_file:
                annotation = ann_file.read().strip()
                annotation = annotation.split(' ')
                # print(f"Annotation:              {annotation}")
        else:
            print("Annotation:              Not found")

        # print(f"Original DEMON shape:    {result['matrix_shape_before_fit']}")
        # print(f"Model input shape:       {result['matrix_shape_after_fit']}")
        print("---------------------------------------------")
        print(f"Audio file:              {result['audio_path']}")
        print(f"Presence probability:    {result['presence_probability']:.4f}")
        print(f"Vessel detected:         {result['vessel_detected']}")
        print(f"Predicted column:        {result['predicted_column']}")
        print(f"ground truth column (viterbi mean): {annotation[1] if 'annotation' in locals() else 'N/A'}")
        print(f"ground truth presence:   {annotation[0] if 'annotation' in locals() else 'N/A'}")

        # exit()

        # print(f"Raw best column:         {result['raw_best_column']}")
        # print("---------------------------------------------")
        # print(f"SNR mean:                {result['snr_mean']}")
        # print(f"SNR max:                 {result['snr_max']}")
        # print("=============================================/n")


if __name__ == "__main__":
    main()