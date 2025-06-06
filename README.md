# S-KEY

`skey` is a Python package for state-of-the-art automatic **musical key detection** from audio recordings, based on the S-KEY model proposed by Yuexuan Kong et al. The package provides an efficient pipeline for loading audio and inferring musical key using a trained deep learning model ChromaNet.

- 📄 [S-KEY: Self-supervised Learning of Major and Minor Keys from Audio](https://arxiv.org/abs/2501.12907)  
- ✅ Accepted at [ICASSP 2025](https://ieeexplore.ieee.org/xpl/conhome/10887540/proceeding)

## Features

- 🎼 End-to-end musical key detection from raw audio
- 🧠 A open-sourced pretrained model
- ⚙️  Simple CLI and Python API
- 💽 Support for `.wav`, `.mp3`, etc.
- 🔌 CPU and GPU support



## Installation

```bash
pip install .
```

## Usage

### 🔧 Command Line Interface (CLI)

```bash
skey path/to/audio
```

This will run key detection on the specified audio file or directory using the default model and settings.

To specify additional options, use the following arguments:

```bash
skey path/to/audio --checkpoint path/to/model.pt --ext mp3 --device cpu
```

- `--checkpoint`: Path to a custom model checkpoint (`.pt`). If not provided, the default model is used.
- `--ext`: Audio file extension (default: `wav`), if `path/to/audio` is a directory. Else infers the extension from the file. Supports all formats readable by torchaudio.
- `--device`: Device to run on (default: `cpu`, e.g., `cuda`, `mps`).


**Arguments**:

| Argument                | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `path/to/audio`     | Path to directory with audio files or a single audio file |
| `--checkpoint`          | Path to model checkpoint (`.pt`). Loads default if not provided. |
| `--ext`                 | Audio file extension (default: `wav`, supports all formats that can be read by torchaudio) if `path/to/audio` is a directory |
| `--device`              | Device to run on (default: `cpu`, e.g., `cuda`, `mps`)                  |


### 🐍 Python API

```python
from skey import detect_key

detect_key(
    audio_dir="path/to/audio",
    extension="mp3",
    device="cpu"
)
```

**Parameters**:

* `audio_dir` (str): Path to the audio file or directory containing audio files
* `ckpt_path` (str or None, optional): Path to the model checkpoint file. If `None`, the default model is used.
* `extension` (str, optional): File extension (default: `"wav"`)
* `device` (str, optional): Device to run on (default: `cpu`)

## 🗂️ Code organization

```
skey
├── Dockerfile
├── LICENSE
├── README.md
├── pyproject.toml
├── skey
│   ├── __init__.py
│   ├── cli.py
│   ├── key_detection.py
│   └── src
│       ├── chromanet.py
│       ├── convnext.py
│       └── hcqt.py
└── training_utils
    ├── config
    │   └── skey.gin
    ├── skey.py
    └── skey_loss.py
```

⚠️ The `training_utils/` directory is **not used** in the `skey` package for inference. However, it is **essential** if you plan to **retrain the model**. It contains:

* full model definition
* loss functions
* Configuration file (`skey.gin`)

To retrain, you will need to plug in your own dataloader and training loop using this codebase as a foundation.

## 📚 Reference

If you use this work in your research, please cite:

```
@INPROCEEDINGS{kongskey2025,
  author={Kong, Yuexuan and Meseguer-Brocal, Gabriel and Lostanlen, Vincent and Lagrange, Mathieu and Hennequin, Romain},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={S-KEY: Self-supervised Learning of Major and Minor Keys from Audio}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10890222}}
```

## 📄 License

The code of **SKEY** is [MIT-licensed](LICENSE).
