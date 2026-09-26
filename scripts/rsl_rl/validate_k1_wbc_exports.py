#!/usr/bin/env python3
"""Validate JIT and ONNX WBC policy exports against each other."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

from k1_wbc_export_manifest import verify_export_manifest

REPORT_NAME = "k1_wbc_export_validation.json"


def _static_tensor_shape(value_info: Any, label: str) -> tuple[int, ...]:
    dimensions = value_info.type.tensor_type.shape.dim
    shape = tuple(int(dimension.dim_value) for dimension in dimensions)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise RuntimeError(f"{label} must have a static positive shape: {shape}")
    return shape


def validate_exports(
    checkpoint: Path,
    export_dir: Path,
    *,
    samples: int = 32,
    seed: int = 42,
    absolute_tolerance: float = 5.0e-5,
    relative_tolerance: float = 5.0e-4,
    output: Path | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("export validation tolerances must be non-negative")

    checkpoint = checkpoint.resolve()
    export_dir = export_dir.resolve()
    manifest = verify_export_manifest(checkpoint, export_dir)
    jit_path = export_dir / manifest["jit"]["name"]
    onnx_path = export_dir / manifest["onnx"]["name"]

    jit_policy = torch.jit.load(str(jit_path), map_location="cpu").eval()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    if len(onnx_model.graph.input) != 1 or len(onnx_model.graph.output) != 1:
        raise RuntimeError(
            "WBC ONNX policy must have exactly one input and one output: "
            f"inputs={len(onnx_model.graph.input)} outputs={len(onnx_model.graph.output)}"
        )
    input_info = onnx_model.graph.input[0]
    output_info = onnx_model.graph.output[0]
    input_shape = _static_tensor_shape(input_info, "ONNX input")
    output_shape = _static_tensor_shape(output_info, "ONNX output")
    evaluator = ReferenceEvaluator(onnx_model)

    rng = np.random.default_rng(seed)
    observations = [np.zeros(input_shape, dtype=np.float32)]
    for index in range(samples - 1):
        if index % 2 == 0:
            observation = rng.standard_normal(input_shape).astype(np.float32)
        else:
            observation = rng.uniform(-10.0, 10.0, input_shape).astype(np.float32)
        observations.append(observation)

    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for sample_index, observation in enumerate(observations):
        with torch.inference_mode():
            jit_actions = jit_policy(torch.from_numpy(observation))
        if not torch.is_tensor(jit_actions):
            raise RuntimeError(
                f"JIT policy returned a non-tensor for sample {sample_index}: "
                f"{type(jit_actions).__name__}"
            )
        jit_array = jit_actions.detach().cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            onnx_array = evaluator.run(None, {input_info.name: observation})[0]
        if jit_array.shape != output_shape or onnx_array.shape != output_shape:
            raise RuntimeError(
                f"export output shape mismatch for sample {sample_index}: "
                f"expected={output_shape} jit={jit_array.shape} onnx={onnx_array.shape}"
            )
        if not np.isfinite(jit_array).all() or not np.isfinite(onnx_array).all():
            raise RuntimeError(f"non-finite export output for sample {sample_index}")
        absolute_error = np.abs(jit_array - onnx_array)
        scale = np.maximum(np.maximum(np.abs(jit_array), np.abs(onnx_array)), 1.0e-6)
        relative_error = absolute_error / scale
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(absolute_error.max(initial=0.0)),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            float(relative_error.max(initial=0.0)),
        )
        tolerance = absolute_tolerance + relative_tolerance * scale
        if np.any(absolute_error > tolerance):
            raise RuntimeError(
                f"JIT/ONNX output mismatch for sample {sample_index}: "
                f"max_abs={float(absolute_error.max())} "
                f"max_rel={float(relative_error.max())}"
            )

    report: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": manifest["checkpoint"],
        "jit": manifest["jit"],
        "onnx": manifest["onnx"],
        "input_name": input_info.name,
        "input_shape": list(input_shape),
        "output_name": output_info.name,
        "output_shape": list(output_shape),
        "samples": samples,
        "seed": seed,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "passed": True,
    }
    report_path = output or export_dir / REPORT_NAME
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--absolute-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--relative-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_exports(
        args.checkpoint,
        args.export_dir,
        samples=args.samples,
        seed=args.seed,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        output=args.output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
