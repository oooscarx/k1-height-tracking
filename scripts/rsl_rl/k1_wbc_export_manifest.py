#!/usr/bin/env python3
"""Record and verify the checkpoint provenance of exported WBC policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "k1_wbc_export_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str | int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"export provenance file is missing or empty: {path}")
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_export_manifest(
    checkpoint: Path,
    export_dir: Path,
    jit_filename: str,
    onnx_filename: str,
) -> Path:
    checkpoint = checkpoint.resolve()
    export_dir = export_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "checkpoint": _file_record(checkpoint),
        "jit": _file_record(export_dir / jit_filename),
        "onnx": _file_record(export_dir / onnx_filename),
    }
    manifest = export_dir / MANIFEST_NAME
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    return manifest


def verify_export_manifest(checkpoint: Path, export_dir: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    export_dir = export_dir.resolve()
    manifest = export_dir / MANIFEST_NAME
    if not manifest.is_file() or manifest.stat().st_size <= 0:
        raise RuntimeError(f"export provenance manifest is missing or empty: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported export provenance schema: {payload.get('schema_version')}"
        )
    expected_checkpoint = _file_record(checkpoint)
    if payload.get("checkpoint") != expected_checkpoint:
        raise RuntimeError(
            "exported policy checkpoint provenance mismatch: "
            f"expected={expected_checkpoint} actual={payload.get('checkpoint')}"
        )
    for kind in ("jit", "onnx"):
        record = payload.get(kind)
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise RuntimeError(f"invalid {kind} export provenance record: {record}")
        if Path(record["name"]).name != record["name"]:
            raise RuntimeError(f"invalid {kind} export filename: {record['name']}")
        actual = _file_record(export_dir / record["name"])
        if record != actual:
            raise RuntimeError(
                f"{kind} export provenance mismatch: expected={record} actual={actual}"
            )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("checkpoint", type=Path)
    write_parser.add_argument("export_dir", type=Path)
    write_parser.add_argument("jit_filename")
    write_parser.add_argument("onnx_filename")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("checkpoint", type=Path)
    verify_parser.add_argument("export_dir", type=Path)
    args = parser.parse_args()

    if args.command == "write":
        manifest = write_export_manifest(
            args.checkpoint,
            args.export_dir,
            args.jit_filename,
            args.onnx_filename,
        )
        print(manifest)
    else:
        payload = verify_export_manifest(args.checkpoint, args.export_dir)
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
