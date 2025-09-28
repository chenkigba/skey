import contextlib
import csv
import glob
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from .chromanet import ChromaNet
from .hcqt import VQT, CropCQT

# 全局：单线程执行器 + 预加载 Future + 路径缓存，确保幂等与单次加载
_executor: ThreadPoolExecutor | None = None
_global_future: Future | None = None
_global_ckpt_path: str | Path | None = None
_lock = threading.Lock()

def _select_device(explicit_device: str | None) -> torch.device:
    """优先使用显式指定；否则自动选择（CUDA > MPS > CPU）。"""
    if explicit_device:
        if explicit_device != "cpu" and not torch.cuda.is_available() and not torch.backends.mps.is_available():
            logging.warning("CUDA and MPS not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _configure_torch_backends(device: torch.device) -> None:
    """与 MelBandRoformer 一致：启用 cudnn.benchmark/TF32 等轻量优化。"""
    try:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        logging.debug(f"Ignore backend tuning error: {e}")

def _load_setup_impl(device_str: str | None, ckpt_path: str | Path):
    d = _select_device(device_str)
    _configure_torch_backends(d)
    ckpt = load_checkpoint(ckpt_path)
    sr = ckpt["audio"]["sr"]
    hcqt, chromanet, crop_fn = load_model_components(ckpt, d)
    return d, sr, hcqt, chromanet, crop_fn

def preload(ckpt_path: str | Path | None = None, device: str | None = None) -> Future:
    """
    后台异步预加载 skey 模型，返回全局 Future（幂等）。
    - 首次调用：启动单线程线程池任务并缓存
    - 后续调用：复用已存在的 Future；如传入不同 ckpt 路径，将告警并忽略
    """
    global _executor, _global_future, _global_ckpt_path
    if ckpt_path is None:
        ckpt_path = DEFAULT_CHECKPOINT_PATH
    with _lock:
        if _global_future is not None:
            if str(ckpt_path) != str(_global_ckpt_path):
                logging.warning(
                    f"skey: 已有预加载任务存在（ckpt={_global_ckpt_path}），忽略新的 ckpt 请求：{ckpt_path}"
                )
            return _global_future
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skey-preload")
        _global_ckpt_path = str(ckpt_path)
        _global_future = _executor.submit(_load_setup_impl, device, ckpt_path)
        logging.info(f"skey: 已启动后台预加载（ckpt={ckpt_path}）")
        return _global_future

logging.basicConfig(level=logging.INFO)

DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "models/skey.pt"

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


def load_audio(song_path: str, sr: float, mono: bool = True, normalize: bool = True) -> torch.Tensor:
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
    if not Path(song_path).exists():
        raise FileNotFoundError(f"File {song_path} not found.")

    try:
        waveform, waveform_sr = torchaudio.load(song_path, backend="soundfile")
    except Exception as e:
        logging.error(f"Failed to load {song_path}: {e}")
        raise ValueError(f"Could not load audio file {song_path}: {e}")

    # Resample if necessary
    if waveform_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=waveform_sr, new_freq=sr)(waveform)

    # Convert to mono if specified
    if mono == True and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if normalize:
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

    # logging.info(f"Waveform shape: {waveform.shape}")
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


def load_model_components(ckpt: Dict[str, Any], device: torch.device) -> Tuple[VQT, ChromaNet, CropCQT]:
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

    hcqt.load_state_dict({k.replace("hcqt.", ""): v for k, v in ckpt["stone"].items() if "hcqt" in k})

    chromanet.load_state_dict({k.replace("chromanet.", ""): v for k, v in ckpt["stone"].items() if "chromanet" in k})

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
        # inference mode + autocast for GPU/MPS
        if device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif device.type == "mps":
            autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode():
            with autocast_ctx:
                cropped = crop_fn(hcqt(new_batch), torch.zeros(1, device=device))
                logits = chromanet(cropped)
                return key_map[int(logits.squeeze(0).argmax().item())]
    except Exception as e:
        logging.warning(f"Inference failed (likely short audio): {e}")
        return "error"


def load_checkpoint(checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH) -> Dict[str, Any]:
    """
    Loads a checkpoint file from the specified checkpoint_path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary.
    """

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(str(checkpoint_path)):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def detect_key(
    audio_path: str,
    extension: str = "wav",
    device: str = "cpu",
    ckpt_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    cli: bool = False,
) -> list[str] | None:
    """
    Detects the musical key of audio files using a pre-trained model.

    Args:
        ckpt_path (str): Path to the model checkpoint file.
        audio_path (str): Path to the audio file or directory containing audio files.
        extension (str, optional): File extension of audio files to process. No need to pass this argument when audio_path is a single audio file. Defaults to "wav".
        device (str, optional): Device to perform inference on ("cpu", "cuda", or "mps"). Defaults to "cpu".
        cli (bool, optional): If True, prints results to console. If False, returns results. Defaults to False.

    Returns:
        list[str] | None: List of predicted keys for the audio files. Returns None if no audio files are found.
    """
    d, sr, hcqt, chromanet, crop_fn = setup_skey(device, ckpt_path)

    # Determine if input is a single file or a directory
    audio_files = (
        [audio_path]
        if os.path.isfile(audio_path)
        else glob.glob(os.path.join(audio_path, f"**/*.{extension}"), recursive=True)
    )

    if len(audio_files) == 0:
        raise FileNotFoundError(f"No audio files found in {audio_path}.")

    logging.info(f"\n🔑 Computing key for {len(audio_files)} audio files on {d}...\n")

    results = [
        infer_key(hcqt, chromanet, crop_fn, load_audio(path, sr), d)
        for path in tqdm(audio_files, desc="Processing")
    ]

    if len(audio_files) == 1:
        print(f"\n✅ Predicted key for {audio_files[0]}: {results[0]}\n")
    else:
        out_dir = os.path.join(audio_path, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        out_csv_path = os.path.join(out_dir, "predictions.csv")

        # Save results to a CSV file
        with open(out_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Audio File", "Predicted Key"])
            for i, key in enumerate(results):
                writer.writerow([audio_files[i], key])
        print(f"\n✅ Predictions saved to: {out_csv_path}\n")

    if not cli:
        return results

def setup_skey(device, ckpt_path):
    """
    统一推理侧入口：
    - 若已有全局 Future（预加载）：阻塞等待并复用模型；如路径不一致，仅告警
    - 否则：同步加载一次
    """
    global _global_future, _global_ckpt_path
    with _lock:
        fut = _global_future
        pre_ckpt = _global_ckpt_path
    if fut is not None:
        if str(ckpt_path) != str(pre_ckpt):
            logging.warning(
                f"skey: 预加载 ckpt 与当前请求不一致（preloaded={pre_ckpt}, requested={ckpt_path}），复用已加载模型。"
            )
        return fut.result()
    return _load_setup_impl(device, ckpt_path)
