from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rsl_rl"
    / "k1_wbc_export_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("k1_wbc_export_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


def _write_artifacts(root: Path) -> tuple[Path, Path]:
    checkpoint = root / "model_42.pt"
    checkpoint.write_bytes(b"checkpoint")
    export_dir = root / "exported"
    export_dir.mkdir()
    (export_dir / "policy.pt").write_bytes(b"jit")
    (export_dir / "policy.onnx").write_bytes(b"onnx")
    MANIFEST.write_export_manifest(
        checkpoint,
        export_dir,
        "policy.pt",
        "policy.onnx",
    )
    return checkpoint, export_dir


def test_export_manifest_verifies_checkpoint_and_exports(tmp_path: Path) -> None:
    checkpoint, export_dir = _write_artifacts(tmp_path)

    payload = MANIFEST.verify_export_manifest(checkpoint, export_dir)

    assert payload["checkpoint"]["name"] == "model_42.pt"
    assert payload["jit"]["name"] == "policy.pt"
    assert payload["onnx"]["name"] == "policy.onnx"


@pytest.mark.parametrize("target", ["checkpoint", "jit", "onnx"])
def test_export_manifest_rejects_tampered_artifact(
    tmp_path: Path,
    target: str,
) -> None:
    checkpoint, export_dir = _write_artifacts(tmp_path)
    paths = {
        "checkpoint": checkpoint,
        "jit": export_dir / "policy.pt",
        "onnx": export_dir / "policy.onnx",
    }
    paths[target].write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        MANIFEST.verify_export_manifest(checkpoint, export_dir)
