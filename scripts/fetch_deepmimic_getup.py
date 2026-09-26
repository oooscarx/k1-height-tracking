#!/usr/bin/env python3
"""Download the pinned DeepMimic get-up references used by K1 AMP training."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import urllib.request

DEEP_MIMIC_COMMIT = "1f915c52fcd4b95b5f5f15b759ae91bd81e9a801"
FILES = {
    "data/characters/humanoid3d.txt": "17469797e00704e82c53f70e3c08fd0e5adaf437b9c838b808f1173ca78c16a6",
    "data/motions/humanoid3d_getup_facedown.txt": (
        "1bd1a96e9175f65eb66e10f225473b4467bd70c68e283f4006b2ab090e337c0b"
    ),
    "data/motions/humanoid3d_getup_faceup.txt": (
        "5936c32d51dad9b0cd5e61df681079d160436cdce845c312f200d7a347c4040d"
    ),
}
RAW_ROOT = f"https://raw.githubusercontent.com/xbpeng/DeepMimic/{DEEP_MIMIC_COMMIT}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(output_dir: pathlib.Path) -> None:
    for relative_path, expected_hash in FILES.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and _sha256(destination.read_bytes()) == expected_hash:
            print(f"verified {destination}")
            continue

        url = f"{RAW_ROOT}/{relative_path}"
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        actual_hash = _sha256(payload)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
            )
        destination.write_bytes(payload)
        print(f"verified {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets/deepmimic"),
        help="Destination directory (default: datasets/deepmimic)",
    )
    args = parser.parse_args()
    fetch(args.output_dir.resolve())


if __name__ == "__main__":
    main()
