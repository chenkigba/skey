# skey/cli.py

import argparse
from skey.key_detection import key_detection, download_checkpoint_if_missing

def main():
    parser = argparse.ArgumentParser(description="Key detection from audio")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt) or 'auto' to download")
    parser.add_argument("audio_dir", help="Path to directory with audio files")
    parser.add_argument("--ext", default="wav", help="Audio file extension (default: wav)")
    parser.add_argument("--device", default="cpu", help="Computation device (e.g., 'cpu' or 'cuda:0')")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path.lower() == "auto":
        ckpt_path = download_checkpoint_if_missing()

    key_detection(ckpt_path, args.audio_dir, extension=args.ext, device=args.device)
