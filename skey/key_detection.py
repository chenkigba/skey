import csv
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


def download_checkpoint_if_missing(path: str = "./models/skey.pt"):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print("Checkpoint not found. Downloading from GitHub...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://github.com/deezer/s-key/raw/main/skey.pt"
        urllib.request.urlretrieve(url, path)
        print(f"✅ Checkpoint downloaded to {path}")
    return path


def yield_audio_paths(paths: List[str]) -> Iterator[Dict[str, Any]]:
    """
    Yields audio file paths in a randomized order.

    Args:
        paths (List[str]): List of audio file paths.

    Returns:
        Iterator[Dict[str, Any]]: An iterator that yields dictionaries containing
        the index and the corresponding audio file path.
    """
    for idx in np.random.permutation(len(paths)):
        yield {"idx": idx, "song_path": paths[idx]}


def load_audio(
    song_path: str, sr: float, mono: bool = True, normalize: bool = True
) -> torch.Tensor:
    """
    Loads an audio file and returns its waveform as a PyTorch tensor.

    Args:
        song_path (str): Path to the audio file.
        sr (float): Sampling rate for the audio file.
        mono (bool, optional): Whether to convert the audio to mono. Defaults to True.
        normalize (bool, optional): Whether to normalize the waveform. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the audio waveform. If loading fails, 
                      returns a tensor of zeros with shape (1, 1).
    """
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
    """
    A PyTorch IterableDataset for loading audio files.

    Args:
        paths (List[str]): List of paths to audio files.
        sr (int): Sampling rate for loading audio files.
        device (torch.device): Device to load the audio tensors onto.

    Yields:
        Tuple[torch.Tensor, str]: A tuple containing the audio tensor and the file path.
    """
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
    """
    Loads model components (VQT, ChromaNet, and CropCQT) and initializes them with checkpoint data.

    Args:
        ckpt (Dict[str, Any]): Checkpoint dictionary containing model weights.
        device (torch.device): Device to load the models onto.

    Returns:
        Tuple[VQT, ChromaNet, CropCQT]: Initialized VQT, ChromaNet, and CropCQT components.
    """
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
    """
    Infers the musical key of an audio batch using the provided model components.

    Args:
        hcqt (VQT): VQT model component for harmonic constant-Q transform.
        chromanet (ChromaNet): ChromaNet model component for key prediction.
        crop_fn (CropCQT): CropCQT function for cropping the CQT output.
        batch (torch.Tensor): Audio batch tensor.
        device (torch.device): Device to perform inference on.

    Returns:
        str: Predicted musical key as a string. Returns "error" if inference fails.
    """
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
    """
    Loads a checkpoint file from the specified path.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary.
    """
    logging.info(f"Loading checkpoint from {path}")
    return torch.load(path, map_location="cpu")


def key_detection(
    ckpt_path: str, audio_path: str, extension: str = "wav", device: str = "cpu"
) -> None:
    """
    Detects the musical key of audio files using a pre-trained model.

    Args:
        ckpt_path (str): Path to the model checkpoint file. Use "auto" to download the default checkpoint.
        audio_path (str): Path to the audio file or directory containing audio files.
        extension (str, optional): File extension of audio files to process. No need to pass this argument when audio_path is a single audio file. Defaults to "wav".
        device (str, optional): Device to perform inference on ("cpu", "cuda", or "mps"). Defaults to "cpu".

    Returns:
        None: Prints the predicted key(s) and saves results to a CSV file if multiple files are processed.
    """
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

    hcqt, chromanet, crop_fn = load_model_components(ckpt, d)

    # Determine if input is a single file or a directory
    audio_files = [audio_path] if os.path.isfile(audio_path) else glob.glob(os.path.join(audio_path, f"**/*.{extension}"), recursive=True)

    print(f"\n🔑 Computing key for {len(audio_files)} audio files on {d}...\n")

    results = {
        path: infer_key(hcqt, chromanet, crop_fn, load_audio(path, sr).to(d), d)
        for path in tqdm(audio_files, desc="Processing")
    }

    if len(audio_files) == 1:
        print(f"\n✅ Predicted key for {audio_files[0]}: {results[audio_files[0]]}\n")
    else:
        out_dir = os.path.join(audio_path, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        out_csv_path = os.path.join(out_dir, "predictions.csv")

        # Save results to a CSV file
        with open(out_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Audio File", "Predicted Key"])
            for path, key in results.items():
                writer.writerow([path, key])

        print(f"\n✅ Predictions saved to: {out_csv_path}\n")
