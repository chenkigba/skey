import glob
import logging
import os
import urllib.request
from typing import Any, Dict, Iterator, List, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm

from .src.chromanet import ChromaNet
from .src.hcqt import VQT, CropCQT

logging.basicConfig(level=logging.INFO)

key_map = {
    0: "A Major",
    1: "Bb Major",
    2: "B Major",
    3: "C Major",
    4: "C# Major",
    5: "D Major",
    6: "D# Major",
    7: "E Major",
    8: "F Major",
    9: "F# Major",
    10: "G Major",
    11: "G# Major",
    12: "B minor",
    13: "C minor",
    14: "C# minor",
    15: "D minor",
    16: "D# minor",
    17: "E minor",
    18: "F minor",
    19: "F# minor",
    20: "G minor",
    21: "G# minor",
    22: "A minor",
    23: "Bb minor",
}


def download_checkpoint_if_missing(path: str = "~/.cache/skey/skey.pt"):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print("Checkpoint not found. Downloading from GitHub...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://github.com/deezer/s-key/raw/main/skey.pt"
        urllib.request.urlretrieve(url, path)
        print(f"✅ Checkpoint downloaded to {path}")
    return path


def yield_audio_paths(paths: List[str]) -> Iterator[Dict[str, Any]]:
    for idx in np.random.permutation(len(paths)):
        yield {"idx": idx, "song_path": paths[idx]}


def load_audio(
    song_path: str, sr: float, mono: bool = True, normalize: bool = True
) -> torch.Tensor:
    try:
        waveform_np, sr = librosa.load(song_path, sr=sr, mono=mono)
    except Exception as e:
        logging.warning(f"Failed to load {song_path}: {e}")
        return torch.zeros(1, 1)

    if waveform_np.ndim == 1:
        waveform_np = np.expand_dims(waveform_np, axis=0)
    if mono:
        waveform_np = np.mean(waveform_np, axis=0, keepdims=True)

    waveform = torch.from_numpy(waveform_np).float()

    if normalize:
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

    return waveform


class AudioDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths: List[str], sr: int, device: torch.device):
        self.paths = paths
        self.sr = sr
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        for data in yield_audio_paths(self.paths):
            audio = load_audio(data["song_path"], self.sr).to(self.device)
            if torch.max(torch.abs(audio)) > 0:
                yield audio, data["song_path"]


def load_model_components(
    ckpt: Dict[str, Any], device: torch.device
) -> Tuple[VQT, ChromaNet, CropCQT]:
    hcqt = VQT(
        harmonics=[1],
        fmin=27.5,
        n_bins=99,
    ).to(device)

    chromanet = ChromaNet(
        n_bins=84,
        n_harmonics=1,
        out_channels=[2, 3, 40, 40, 30, 10, 3],
        kernels=[7, 7, 7, 7, 7, 5, 5],
        temperature=1,
    ).to(device)

    hcqt.load_state_dict(
        {k.replace("hcqt.", ""): v for k, v in ckpt["stone"].items() if "hcqt" in k}
    )

    chromanet.load_state_dict(
        {
            k.replace("chromanet.", ""): v
            for k, v in ckpt["stone"].items()
            if "chromanet" in k
        }
    )

    hcqt.eval()
    chromanet.eval()
    crop_fn = CropCQT(84)
    return hcqt, chromanet, crop_fn


def infer_key(
    hcqt: VQT,
    chromanet: ChromaNet,
    crop_fn: CropCQT,
    batch: torch.Tensor,
    device: torch.device,
) -> str:
    new_batch = batch.unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            cropped = crop_fn(hcqt(new_batch), torch.zeros(1).to(device))
            logits = chromanet(cropped)
            return key_map[int(torch.mean(logits, dim=0).argmax())]
    except Exception as e:
        logging.warning(f"Inference failed (likely short audio): {e}")
        return "error"


def load_checkpoint(path: str) -> Dict[str, Any]:
    logging.info(f"Loading checkpoint from {path}")
    return torch.load(path, map_location="cpu")


def key_detection(
    ckpt_path: str, audio_dir: str, extension: str = "wav", device: str = "cpu"
) -> None:
    if (
        device != "cpu"
        and not torch.cuda.is_available()
        and not torch.backends.mps.is_available()
    ):
        logging.warning("CUDA and MPS not available. Falling back to CPU.")
        device = "cpu"

    d = torch.device(device)
    if ckpt_path.lower() == "auto":
        ckpt_path = download_checkpoint_if_missing()
    ckpt = load_checkpoint(ckpt_path)
    sr = ckpt["audio"]["sr"]
    model_name = "-".join(ckpt_path.split("/")[-5:-1])

    hcqt, chromanet, crop_fn = load_model_components(ckpt, d)

    audio_files = glob.glob(f"{audio_dir}/**/*.{extension}", recursive=True)
    dataset = AudioDataset(audio_files, sr, d)

    print(f"\n🔑 Computing key for {len(audio_files)} audio files on {d}...\n")

    results = {
        path: infer_key(hcqt, chromanet, crop_fn, audio, d)
        for audio, path in tqdm(dataset, desc="Processing")
    }

    out_dir = os.path.join(audio_dir, "prediction", model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preds.npz")
    np.savez(out_path, **results)
    print(f"\n✅ Predictions saved to: {out_path}\n")
