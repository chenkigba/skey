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
skey auto path/to/audio_dir --ext mp3 --device cpu
```

**Arguments**:

| Argument                | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `auto` or path to `.pt` | `"auto"` will download the pretrained model automatically to path `~/.cache/skey/skey.pt` |
| `path/to/audio_dir`     | Directory of audio files to analyze                       |
| `--ext`                 | Audio file extension (default: `wav`, supports all formats that can be read by torchaudio)                     |
| `--device`              | Device to run on (default: `cpu`)                  |


### 🐍 Python API

```python
from skey.key_detection import key_detection

key_detection(
    ckpt_path="auto",  # or "path/to/skey.pt"
    audio_dir="path/to/audio_dir",
    extension="mp3",
    device="cpu"
)
```

**Parameters**:

* `ckpt_path` (str): Use `"auto"` to download checkpoint or provide a local `.pt` file
* `audio_dir` (str): Directory with audio files
* `extension` (str): File extension (default: `"wav"`)
* `device` (str): Device to run on (default: `cpu`)

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
