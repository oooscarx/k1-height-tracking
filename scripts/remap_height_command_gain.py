#!/usr/bin/env python3
"""Change checkpoint height-command sensitivity without moving a pivot pose."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_rows(value: str | None, size: int) -> slice:
    if value is None:
        return slice(0, size)
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("actor rows must use START:STOP syntax")
    start, stop = (int(part) for part in parts)
    if not 0 <= start < stop <= size:
        raise argparse.ArgumentTypeError(f"actor rows must be inside 0:{size}")
    return slice(start, stop)


def remap_first_layer(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    gain: float,
    pivot: float,
    history: int,
    rows: slice,
) -> float:
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("expected a linear-layer weight and bias")
    if history < 1 or history > weight.shape[1]:
        raise ValueError("height history width is outside the observation width")
    original = weight[rows, -history:].clone()
    weight[rows, -history:] = original * gain
    bias[rows] += pivot * (1.0 - gain) * original.sum(dim=1)

    test_input = torch.zeros(weight.shape[1], dtype=weight.dtype)
    test_input[-history:] = pivot
    before = original @ test_input[-history:]
    after = weight[rows] @ test_input + bias[rows]
    expected = before + (bias[rows] - pivot * (1.0 - gain) * original.sum(dim=1))
    return float(torch.max(torch.abs(after - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--pivot", type=float, default=0.68)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument(
        "--actor-rows",
        default=None,
        help="Optional first-layer hidden-unit slice; by default remap every hidden unit.",
    )
    args = parser.parse_args()
    if args.gain <= 0.0:
        parser.error("--gain must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor_weight = state["actor.0.weight"]
    actor_bias = state["actor.0.bias"]
    critic_weight = state["critic.0.weight"]
    critic_bias = state["critic.0.bias"]
    actor_rows = parse_rows(args.actor_rows, actor_weight.shape[0])

    actor_error = remap_first_layer(
        actor_weight,
        actor_bias,
        gain=args.gain,
        pivot=args.pivot,
        history=args.history,
        rows=actor_rows,
    )
    critic_error = remap_first_layer(
        critic_weight,
        critic_bias,
        gain=args.gain,
        pivot=args.pivot,
        history=args.history,
        rows=slice(0, critic_weight.shape[0]),
    )
    if max(actor_error, critic_error) > 1.0e-5:
        raise RuntimeError(
            f"pivot invariance failed: actor={actor_error:.3e}, critic={critic_error:.3e}"
        )

    infos = dict(checkpoint.get("infos") or {})
    infos["height_command_gain_remap"] = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "gain": args.gain,
        "pivot_m": args.pivot,
        "history": args.history,
        "actor_rows": [actor_rows.start, actor_rows.stop],
        "actor_pivot_error": actor_error,
        "critic_pivot_error": critic_error,
    }
    checkpoint["infos"] = infos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Wrote {args.output}")
    print(infos["height_command_gain_remap"])


if __name__ == "__main__":
    main()
