"""Remove native recovery lead-in and terminal wait frames from the AMP style clip."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    trajectory_index = data["trajectory_index"]
    positive = np.flatnonzero(trajectory_index > trajectory_index[0])
    changes = np.flatnonzero(np.diff(trajectory_index) > 0)
    if positive.size == 0 or changes.size == 0:
        raise ValueError("trajectory_index does not contain an active recovery segment")

    start = max(int(positive[0]) - 1, 0)
    stop = int(changes[-1]) + 2
    frame_count = trajectory_index.shape[0]
    output = {}
    for name, value in data.items():
        output[name] = value[start:stop] if value.ndim > 0 and value.shape[0] == frame_count else value
    output["trajectory_index"] = output["trajectory_index"] - output["trajectory_index"][0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(
        f"Wrote {args.output}: frames={stop - start}, "
        f"duration={(stop - start) / float(np.asarray(data['fps']).reshape(-1)[0]):.2f}s, "
        f"source_frames=[{start}, {stop})"
    )


if __name__ == "__main__":
    main()
