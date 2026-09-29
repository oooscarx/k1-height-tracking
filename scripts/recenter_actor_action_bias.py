#!/usr/bin/env python3
"""Shift selected actor outputs while preserving their local sensitivities."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_shift(value: str, action_count: int) -> tuple[int, float, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("action shifts must use INDEX:SOURCE:TARGET syntax")
    index, source, target = int(parts[0]), float(parts[1]), float(parts[2])
    if not 0 <= index < action_count:
        raise argparse.ArgumentTypeError(f"action index must be inside 0:{action_count}")
    return index, source, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--shift",
        action="append",
        required=True,
        help="Output row and measured source/desired target means as INDEX:SOURCE:TARGET.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor_bias_keys = [
        key
        for key, value in state.items()
        if key.startswith("actor.") and key.endswith(".bias") and value.shape == (22,)
    ]
    if len(actor_bias_keys) != 1:
        raise ValueError(f"expected one 22-action actor bias, found {actor_bias_keys}")
    bias_key = actor_bias_keys[0]
    bias = state[bias_key]

    shifts = []
    seen = set()
    for raw_shift in args.shift:
        index, source, target = parse_shift(raw_shift, bias.numel())
        if index in seen:
            parser.error(f"action index {index} is repeated")
        seen.add(index)
        delta = target - source
        bias[index] += delta
        shifts.append(
            {
                "action_index": index,
                "source_mean": source,
                "target_mean": target,
                "bias_delta": delta,
            }
        )

    infos = dict(checkpoint.get("infos") or {})
    infos["actor_action_bias_recenter"] = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "actor_bias_key": bias_key,
        "shifts": shifts,
    }
    checkpoint["infos"] = infos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Wrote {args.output}")
    print(infos["actor_action_bias_recenter"])


if __name__ == "__main__":
    main()
