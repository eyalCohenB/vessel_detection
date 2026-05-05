# dataset.py
from pathlib import Path
import pickle

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from helper_funcs.custom_demon import buildDemonMat


def normalize_matrix(x, eps=1e-6):
    x = x.astype(np.float32)
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def extract_demon_matrix_from_pickle(obj, pkl_path):
    """
    Supports several common pickle formats:
    1. raw numpy array
    2. dict with key: 'demon_mat'
    3. dict with key: 'DemonMat'
    4. dict with key: 'matrix'
    """
    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, dict):
        for key in ["demon_mat", "DemonMat", "matrix", "demon_matrix"]:
            if key in obj:
                return obj[key]

    raise ValueError(
        f"Could not find DEMON matrix in pickle file: {pkl_path}. "
        f"Expected numpy array or dict with one of these keys: "
        f"'demon_mat', 'DemonMat', 'matrix', 'demon_matrix'."
    )


def fit_matrix_size(matrix, presence, column, target_height, target_width):
    H, W = matrix.shape

    # -----------------
    # Fit width
    # -----------------
    if W > target_width:
        if presence == 1:
            start = max(0, column - target_width // 2)
            end = start + target_width

            if end > W:
                end = W
                start = W - target_width

            matrix = matrix[:, start:end]
            column = column - start
        else:
            matrix = matrix[:, :target_width]

    elif W < target_width:
        pad_right = target_width - W
        matrix = np.pad(matrix, ((0, 0), (0, pad_right)), mode="constant")

    # -----------------
    # Fit height
    # -----------------
    H, W = matrix.shape

    if H > target_height:
        start = (H - target_height) // 2
        end = start + target_height
        matrix = matrix[start:end, :]

    elif H < target_height:
        pad_total = target_height - H
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        matrix = np.pad(matrix, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    return matrix, column


class VesselDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_height,
        target_width,
        demon_dir=None,
        use_precomputed=True,
        supported_exts=(".wav", ".flac"),
        pkl_ext=".pkl",
        block_len=1.0,
        fpass=100,
        fstop=1500,
        fpass_demon=300,
        normalize=False,
    ):
        self.data_dir = Path(data_dir)
        self.demon_dir = Path(demon_dir) if demon_dir is not None else None

        self.target_height = target_height
        self.target_width = target_width
        self.use_precomputed = use_precomputed

        self.supported_exts = tuple(ext.lower() for ext in supported_exts)
        self.pkl_ext = pkl_ext

        self.block_len = block_len
        self.fpass = fpass
        self.fstop = fstop
        self.fpass_demon = fpass_demon
        self.normalize = normalize

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if self.use_precomputed:
            if self.demon_dir is None:
                raise ValueError("demon_dir must be provided when use_precomputed=True")
            if not self.demon_dir.exists():
                raise FileNotFoundError(f"DEMON matrix directory not found: {self.demon_dir}")

        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in: {self.data_dir}")

    def _find_audio_files(self):
        audio_files = []

        for ext in self.supported_exts:
            audio_files.extend(self.data_dir.glob(f"*{ext}"))
            audio_files.extend(self.data_dir.glob(f"*{ext.upper()}"))

        return sorted(set(audio_files))

    def _get_pkl_path_for_audio(self, audio_path):
        return self.demon_dir / f"{audio_path.stem}{self.pkl_ext}"

    def _is_audio_readable(self, audio_path):
        try:
            sf.info(str(audio_path))
            return True
        except Exception as e:
            print(f"[WARNING] Unreadable audio file {audio_path.name}, skipping. Error: {e}")
            return False

    def _is_pickle_readable(self, pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)

            matrix = extract_demon_matrix_from_pickle(obj, pkl_path)
            matrix = np.asarray(matrix)

            if matrix.ndim != 2:
                print(
                    f"[WARNING] Pickle matrix is not 2D in {pkl_path.name}. "
                    f"Got shape {matrix.shape}, skipping."
                )
                return False

            return True

        except Exception as e:
            print(f"[WARNING] Bad pickle file {pkl_path.name}, skipping. Error: {e}")
            return False

    def _build_index(self):
        samples = []

        audio_files = self._find_audio_files()

        for audio_path in audio_files:
            txt_path = audio_path.with_suffix(".txt")

            if not txt_path.exists():
                print(f"[WARNING] Missing annotation for {audio_path.name}, skipping.")
                continue

            try:
                presence, column = self._read_annotation(txt_path)
            except Exception as e:
                print(f"[WARNING] Failed to parse {txt_path.name}: {e}")
                continue

            pkl_path = None

            if self.use_precomputed:
                pkl_path = self._get_pkl_path_for_audio(audio_path)

                if not pkl_path.exists():
                    print(
                        f"[WARNING] Missing precomputed DEMON matrix for "
                        f"{audio_path.name}. Expected: {pkl_path}"
                    )
                    continue

                if not self._is_pickle_readable(pkl_path):
                    continue

            else:
                if not self._is_audio_readable(audio_path):
                    continue

            samples.append({
                "audio_path": audio_path,
                "txt_path": txt_path,
                "pkl_path": pkl_path,
                "presence": presence,
                "column": column,
            })

        return samples

    def _read_annotation(self, txt_path):
        content = txt_path.read_text(encoding="utf-8").strip()
        parts = content.split()

        if len(parts) != 2:
            raise ValueError(
                f"Annotation must have exactly 2 values. Got: '{content}'"
            )

        presence = int(parts[0])
        column = int(parts[1])

        if presence not in (0, 1):
            raise ValueError(f"Presence must be 0 or 1, got {presence}")

        if presence == 0 and column != -1:
            raise ValueError(f"If presence=0, column must be -1. Got {column}")

        if presence == 1 and column < 0:
            raise ValueError(f"If presence=1, column must be >= 0. Got {column}")

        return presence, column

    def _load_audio(self, audio_path):
        try:
            data, fs = sf.read(str(audio_path))
        except Exception as e:
            raise RuntimeError(f"Failed reading audio file {audio_path}: {e}")

        if data.ndim > 1:
            sig = data[:, 0]
        else:
            sig = data

        sig = np.asarray(sig).squeeze()

        if sig.ndim != 1:
            raise ValueError(
                f"Audio signal must become 1D after channel selection. "
                f"Got shape {sig.shape} for file {audio_path}"
            )

        return sig, fs

    def _build_demon_matrix_from_audio(self, audio_path):
        sig, fs = self._load_audio(audio_path)

        demon_mat, snr_vec = buildDemonMat(
            sig=sig,
            fs=fs,
            BlockLen=self.block_len,
            fpass=self.fpass,
            fstop=self.fstop,
            fpassDemon=self.fpass_demon,
        )

        demon_mat = np.asarray(demon_mat)

        if demon_mat.ndim != 2:
            raise ValueError(
                f"buildDemonMat must return a 2D DemonMat. "
                f"Got shape {demon_mat.shape} for file {audio_path}"
            )

        return demon_mat

    def _load_demon_matrix_from_pickle(self, pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        matrix = extract_demon_matrix_from_pickle(obj, pkl_path)
        matrix = np.asarray(matrix)

        if matrix.ndim != 2:
            raise ValueError(
                f"Loaded DEMON matrix must be 2D. "
                f"Got shape {matrix.shape} from {pkl_path}"
            )

        return matrix

    def _get_matrix(self, item):
        if self.use_precomputed:
            return self._load_demon_matrix_from_pickle(item["pkl_path"])

        return self._build_demon_matrix_from_audio(item["audio_path"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        matrix = self._get_matrix(item)

        matrix, column = fit_matrix_size(
            matrix=matrix,
            presence=item["presence"],
            column=item["column"],
            target_height=self.target_height,
            target_width=self.target_width,
        )

        if self.normalize:
            matrix = normalize_matrix(matrix)

        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        y_presence = torch.tensor(item["presence"], dtype=torch.float32)
        y_column = torch.tensor(column, dtype=torch.long)

        return {
            "x": x,
            "presence": y_presence,
            "column": y_column,
            "audio_path": str(item["audio_path"]),
            "pkl_path": str(item["pkl_path"]) if item["pkl_path"] is not None else "",
        }