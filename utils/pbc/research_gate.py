#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Compare a complete research run with a frozen hold baseline.

Inputs have a `cases` array. Baseline rows contain case_id, source_sha256,
hold (five metrics or null), preserve, and output_sha256. Candidate rows
contain case_id, source_sha256, metrics (or null), output_sha256, and
hold_certificate_verified. That last field must come from the proof runner,
not from a synthesis success status. This script compares evidence; it does
not itself prove circuit equivalence. Missing cases never count as ties.

Reject total-gate and T increases against old hold. Secondary metric increases
are always reported; --all-metrics rejects them. Against preserved network,
report exact gate wins/ties separately from an allowed excess of at most three
gates per T saved.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

METRICS = ("total", "t", "twoq", "depth", "tdepth")


def compare(baseline, candidate, *, all_metrics=False):
    issues = []

    def index(rows, label):
        result = {}
        for row in rows:
            key = row["case_id"]
            if key in result:
                issues.append(f"{label}: duplicate {key}")
            result[key] = row
        return result

    old = index(baseline["cases"], "baseline")
    new = index(candidate["cases"], "candidate")
    for key in sorted(old.keys() ^ new.keys()):
        issues.append(f"case coverage mismatch: {key}")
    comparisons = Counter()
    target_comparisons = Counter()
    improved = []
    tradeoffs = []
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        if not a.get("source_sha256") or b.get("source_sha256") != a["source_sha256"]:
            issues.append(f"input binding mismatch: {key}")
        before, after = a.get("hold"), b.get("metrics")
        if after is None:
            comparisons["failed"] += 1
            target_comparisons["failed"] += 1
            if before is not None:
                issues.append(f"new compilation failure: {key}")
            continue
        if any(type(after.get(m)) is not int or after[m] < 0 for m in METRICS):
            issues.append(f"invalid metric vector: {key}")
            continue
        unchanged = bool(a.get("output_sha256")) and (
            b.get("output_sha256") == a["output_sha256"] and after == before
        )
        if not unchanged and b.get("hold_certificate_verified") is not True:
            issues.append(f"changed output lacks verified hold certificate: {key}")
        if not b.get("output_sha256"):
            issues.append(f"missing output hash: {key}")
        if before is not None:
            for metric in METRICS if all_metrics else ("total",):
                if after[metric] > before[metric]:
                    issues.append(f"{metric} regression: {key}: {before[metric]} -> {after[metric]}")
            ceiling = before
            if not ceiling or type(ceiling.get("t")) is not int:
                issues.append(f"missing T ceiling evidence: {key}")
            elif after["t"] > ceiling["t"]:
                issues.append(f"T ceiling exceeded: {key}: {after['t']} > {ceiling['t']}")
            changes = {m: [before[m], after[m]] for m in METRICS[1:]
                       if after[m] > before[m]}
            if changes:
                tradeoffs.append({"case_id": key, "increases": changes})
            if after["total"] < before["total"]:
                improved.append(key)
        target = a["preserve"]["total"]
        relation = "win" if after["total"] < target else "tie" if after["total"] == target else "loss"
        comparisons[relation] += 1
        saved_t = a["preserve"]["t"] - after["t"]
        allowed_tradeoff = (relation == "loss" and saved_t > 0
                            and after["total"] - target <= 3 * saved_t)
        target_comparisons["t_tradeoff" if allowed_tradeoff else relation] += 1
    return {"admissible": not issues, "issues": issues,
            "t_ceiling": "hold", "all_metrics": all_metrics,
            "versus_target": dict(target_comparisons),
            "secondary_tradeoffs": tradeoffs,
            "improved_cases": improved, "versus_preserve": dict(comparisons),
            "baseline_cases": len(old), "candidate_cases": len(new)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--all-metrics", action="store_true",
                        help="also reject entangler, depth and T-depth increases")
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()),
                     json.loads(args.candidate.read_text()),
                     all_metrics=args.all_metrics)
    print(json.dumps(result, indent=2))
    return 0 if result["admissible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
