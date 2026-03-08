#!/usr/bin/env python3
"""
Download Google Speech Commands dataset (v0.02) directly from the official
tarballs and unpack into a local directory structure.

We deliberately avoid `datasets.load_dataset("google/speech_commands")` because
dataset scripts are disabled in the current environment.
"""

from pathlib import Path
from typing import Iterable
import tarfile
import urllib.request


BASE_URL = "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.02/v0.02_{split}.tar.gz"
SPLITS = ("train", "validation", "test")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def extract_tar(tar_path: Path, dest_dir: Path) -> None:
    print(f"  Extracting {tar_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def iter_files(paths: Iterable[Path]) -> int:
    count = 0
    for p in paths:
        if p.is_file():
            count += 1
    return count


def main() -> int:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    sc_root = project_root / "datasets" / "speech_commands" / "v0.02"

    print("=" * 60)
    print("Google Speech Commands Dataset Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {sc_root}")
    print("Source: TensorFlow Speech Commands v0.02 tarballs via HF S3\n")

    total_files = 0
    total_bytes = 0

    for split in SPLITS:
        url = BASE_URL.format(split=split)
        split_dir = sc_root / split
        tar_path = sc_root / f"v0.02_{split}.tar.gz"

        try:
            download_file(url, tar_path)
            extract_tar(tar_path, split_dir)
        except Exception as e:
            print(f"\n✗ Failed to download/extract split '{split}': {e}")
            return 1

        # Basic stats
        wav_files = list(split_dir.rglob("*.wav"))
        split_files = len(wav_files)
        split_bytes = sum(f.stat().st_size for f in wav_files)
        print(f"  Split '{split}': {split_files} WAV files, {split_bytes / (1024 * 1024):.1f} MB")

        total_files += split_files
        total_bytes += split_bytes

    print("\n✓ Speech Commands v0.02 downloaded and unpacked.")
    print(f"  Total WAV files: {total_files}")
    print(f"  Approx total size: {total_bytes / (1024 * 1024):.1f} MB")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

