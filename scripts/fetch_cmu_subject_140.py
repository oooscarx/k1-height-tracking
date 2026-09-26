#!/usr/bin/env python3
"""Download and verify the CMU Subject 140 get-up motion files."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import ssl
import urllib.request

CMU_ROOT = "https://mocap.cs.cmu.edu/subjects/140"
FILES = {
    "140.asf": "13d3c7488671d3902c0ea7faa672e1c79465c034c050de88668ace164e519a37",
    "140_01.amc": "35f699e9f25acc67fc0aac2779f2031c53107345d776dfbd514b0e7f65e55cc6",
    "140_02.amc": "bf8606f20b51b00a2858e350c7ebce6389efabe24a5fd36b90601810cadfdcb4",
    "140_03.amc": "84072adc179aafe389d8d5dd6d936ce212cef7e496e8938045f5127be97743d8",
    "140_04.amc": "373e32db8697ee5688fa64cccab71c2106c34e79b6dfd9559d50b249d5cb10e5",
    "140_08.amc": "afaafee69db963a51059e324355d089e95d48f71117cd44e4304d68bf1a4dbf8",
    "140_09.amc": "61bda1bd7d85ccd3e2c93685268ec1891a4a147138c89bb69c83f82054b7eb9f",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(output_dir: pathlib.Path, insecure: bool) -> None:
    context = ssl._create_unverified_context() if insecure else None
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in FILES.items():
        destination = output_dir / filename
        if destination.is_file() and _sha256(destination.read_bytes()) == expected_hash:
            print(f"verified {destination}")
            continue

        url = f"{CMU_ROOT}/{filename}"
        print(f"downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "booster-train/1.0"})
        with urllib.request.urlopen(request, timeout=180, context=context) as response:
            payload = response.read()
        actual_hash = _sha256(payload)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        destination.write_bytes(payload)
        print(f"verified {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets/cmu_mocap/subject_140"),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification; pinned SHA-256 hashes are still enforced.",
    )
    args = parser.parse_args()
    fetch(args.output_dir.resolve(), args.insecure)


if __name__ == "__main__":
    main()
