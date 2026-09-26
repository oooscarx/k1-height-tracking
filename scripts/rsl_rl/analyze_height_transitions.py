"""Summarize independent command segments, not individual frames as trials."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def band(value, edges):
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return "outside"


def summary(records):
    settled = [r for r in records if r["settled_s"] > 0]
    eligible = [r for r in settled if r["settled_s"] >= 1.0 - 1e-6]
    total_time = sum(r["settled_s"] for r in settled)
    out = dict(
        segments=len(records), settled_segments=len(settled), success_eligible_segments=len(eligible),
        mean_abs_error=sum(r["mean_abs_error"] * r["settled_s"] for r in settled) / total_time if total_time else None,
        mean_signed_error=sum(r["mean_signed_error"] * r["settled_s"] for r in settled) / total_time if total_time else None,
        settled_frame_success=sum(r["within_threshold_fraction"] * r["settled_s"] for r in settled) / total_time if total_time else None,
        sustained_success=sum(r["sustained_success"] for r in eligible) / len(eligible) if eligible else None,
        stable_segment_success=sum(r["within_threshold_fraction"] >= .9 for r in eligible) / len(eligible) if eligible else None,
        eligible_mean_abs_error=sum(r["mean_abs_error"] * r["settled_s"] for r in eligible) / sum(r["settled_s"] for r in eligible) if eligible else None,
        absolute_error_env_seconds=sum(r["mean_abs_error"] * r["settled_s"] for r in settled),
        environment_count=len({r["cluster"] for r in records}),
    )
    for name, score in (("success", lambda r: r["sustained_success"]),
                        ("stable_segment", lambda r: r["within_threshold_fraction"] >= .9)):
        clusters = defaultdict(list)
        for r in eligible:
            clusters[r["cluster"]].append(int(score(r)))
        if len(clusters) >= 4:
            rng = random.Random(731)
            groups = list(clusters.values())
            estimates = []
            for _ in range(500):
                sample = [x for group in rng.choices(groups, k=len(groups)) for x in group]
                estimates.append(sum(sample) / len(sample))
            estimates.sort()
            out[name + "_cluster_bootstrap_ci95"] = [estimates[12], estimates[487]]
    return out


def group(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[str(key(r))].append(r)
    return {k: summary(v) for k, v in sorted(groups.items())}


def analyze(paths):
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            r["cluster"] = f"{path}:{r['env_id']}"
            records.append(r)
    natural = [r for r in records if r["start_kind"] == "natural"]
    positive = [r for r in natural if r["target_height"] >= 0]
    complete = [r for r in positive if r["end_reason"] == "resample"]
    delta = lambda r: band(r["delta_height"], [-2, -.2, -.08, .08, .2, 2])
    target = lambda r: band(r["target_height"], [0, .15, .3, .45, .6, .68, .721])
    contact = lambda r: "stable_two_feet" if r["stable_two_feet"] else r["start_contact"]
    hold = lambda r: band(r["planned_hold_s"], [1, 2, 3, 4, 5, 7.01])
    episodes = defaultdict(list)
    episode_proxies = []
    for r in records:
        episodes[r["cluster"]].append(r)
        if r["end_reason"] in ("timeout", "termination_or_reset"):
            episode = episodes.pop(r["cluster"])
            entry = {}
            for name, minimum in (("all", 0), ("high", .6)):
                valid = [s for s in episode if s["target_height"] >= minimum and s["settled_s"] > 0]
                seconds = sum(s["settled_s"] for s in valid)
                entry[name + "_seconds"] = seconds
                entry[name + "_error_zero_if_empty"] = sum(s["mean_abs_error"] * s["settled_s"] for s in valid) / seconds if seconds else 0
            episode_proxies.append(entry)
    return dict(
        definitions=dict(primary="natural positive-height commands ending by resampling, excluding episode/recording censoring",
                         error="time-weighted absolute error after 2s",
                         success="at least 1 continuous second within 8cm AFTER 2s; denominator requires >=1s observed settled time",
                         stable_segment="at least 90% of settled frames within 8cm; requires >=1s observed settled time",
                         confidence="95% bootstrap over environment clusters, not over frames",
                         short_commands="no settled samples is missing evidence, not failure"),
        coverage=dict(all_segments=len(records), natural_segments=len(natural),
                      total_observed_env_seconds=sum(r["observed_s"] for r in records),
                      positive_settled_env_seconds=sum(r["settled_s"] for r in records if r["target_height"] >= 0),
                      natural_negative_fraction=sum(r["target_height"] < 0 for r in natural) / max(1, len(natural)),
                      natural_short_fraction=sum(r["planned_hold_s"] <= 2 for r in natural) / max(1, len(natural)),
                      positive_end_reasons=dict(Counter(r["end_reason"] for r in positive))),
        overall=summary(complete),
        by_start_height=group(complete, lambda r: band(r["start_height"], [-1, .15, .3, .45, .6, .75, 2])),
        by_delta=group(complete, delta), by_contact=group(complete, contact),
        by_hold=group(complete, hold), by_target=group(complete, target),
        by_previous_relaxation=group(complete, lambda r: r["previous_target"] < 0),
        exact_072_delta=group([r for r in complete if abs(r["target_height"] - .72) < 1e-5], delta),
        exact_072_contact_delta=group([r for r in complete if abs(r["target_height"] - .72) < 1e-5], lambda r: (contact(r), delta(r))),
        target_delta=group(complete, lambda r: (target(r), delta(r))),
        contact_delta=group(complete, lambda r: (contact(r), delta(r))),
        target_delta_hold=group(complete, lambda r: (target(r), delta(r), hold(r))),
        stable_contact_target_delta=group([r for r in complete if r["stable_two_feet"]], lambda r: (target(r), delta(r))),
        stable_contact_low_speed=summary([r for r in complete if r["stable_two_feet"] and r["start_speed"] < .15]),
        stable_contact_low_speed_delta=group([r for r in complete if r["stable_two_feet"] and r["start_speed"] < .15], delta),
        timeout_observed=summary([r for r in positive if r["end_reason"] == "timeout"]),
        training_metric_proxy=dict(
            note="reconstructed complete episodes; excludes training metric one-step timing difference",
            episodes=len(episode_proxies),
            **{name: {"mean_error_zero_if_empty": sum(e[name + "_error_zero_if_empty"] for e in episode_proxies) / max(1, len(episode_proxies)),
                      "empty_episodes": sum(e[name + "_seconds"] == 0 for e in episode_proxies)} for name in ("all", "high")},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(analyze(args.inputs), indent=2))
