"""Summarize evaluation-image-pipeline cells of one model: success, paired differences, flips per task, input distances.

Every cell is a run directory written by run_openvla_libero_logged.py or run_oft_libero_logged.py.
The first directory is the reference; every other cell is paired with it on (task_id, episode_idx), after
checking that both episodes started from the same initial state.

Per cell: successes with a Wilson 95% CI; paired difference to the reference; discordant pairs
(reference success -> cell failure, and the reverse) with an exact two-sided McNemar p-value; the share of
pairs that changed outcome; flips per task; the mean first-input change of the JPEG round trip (jpeg_input_mad);
and, when both runs saved their first policy inputs, the mean absolute pixel distance to the reference input.

Usage: python summarize_pipeline_cells.py REFERENCE_DIR CELL_DIR [CELL_DIR ...] [--out summary.md]
"""
import argparse
import json
import math
import os
from collections import Counter

import numpy as np


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "episodes.jsonl"))]
    eps = {(r["task_id"], r["episode_idx"]): r for r in rows}
    assert len(eps) == len(rows), f"{run_dir}: duplicate episodes"
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    return eps, meta


def input_distance(ref_dir, cell_dir, ref, cell, keys):
    from PIL import Image

    dists = []
    for k in keys:
        a, b = ref[k].get("policy_input_png"), cell[k].get("policy_input_png")
        if not a or not b:
            return None
        pairs = list(a.values()) if isinstance(a, dict) else [a]
        other = list(b.values()) if isinstance(b, dict) else [b]
        for pa, pb in zip(pairs, other):
            ia = np.asarray(Image.open(os.path.join(ref_dir, pa)), dtype=np.int16)
            ib = np.asarray(Image.open(os.path.join(cell_dir, pb)), dtype=np.int16)
            dists.append(float(np.mean(np.abs(ia - ib))))
    return float(np.mean(dists)) if dists else None


def mean_mad(rows):
    vals = []
    for r in rows:
        m = r.get("jpeg_input_mad")
        if m is None:
            continue
        vals.extend(m.values() if isinstance(m, dict) else [m])
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref_dir, cells = a.dirs[0], a.dirs[1:]
    ref, ref_meta = load(ref_dir)
    lines = [f"Reference: {ref_dir} ({ref_meta.get('checkpoint')}, image pipeline {ref_meta.get('image_pipeline')})", ""]
    k = sum(r["success"] for r in ref.values())
    lo, hi = wilson(k, len(ref))
    lines.append(f"| cell | success | Wilson 95% CI | diff vs ref (pts) | ref ok -> fail | ref fail -> ok | changed | McNemar p | JPEG input change | input distance to ref |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    mad = mean_mad(ref.values())
    lines.append(f"| {os.path.basename(ref_dir.rstrip('/'))} (reference) | {k}/{len(ref)} = {100 * k / len(ref):.1f}% | [{lo:.1f}, {hi:.1f}] | - | - | - | - | - | "
                 f"{'' if mad is None else f'{mad:.2f}'} | - |")
    per_task_lines = []
    for cell_dir in cells:
        cell, meta = load(cell_dir)
        keys = sorted(set(ref) & set(cell))
        assert keys and len(keys) == len(ref) == len(cell), f"{cell_dir}: episode sets differ from the reference"
        bad = [key for key in keys if ref[key]["init_state_sha1"] != cell[key]["init_state_sha1"]]
        assert not bad, f"{cell_dir}: initial states differ at {bad[:5]}"
        kc = sum(cell[key]["success"] for key in keys)
        lo, hi = wilson(kc, len(keys))
        b = sum(ref[key]["success"] and not cell[key]["success"] for key in keys)
        c = sum(cell[key]["success"] and not ref[key]["success"] for key in keys)
        diff = 100 * (kc - sum(ref[key]["success"] for key in keys)) / len(keys)
        dist = input_distance(ref_dir, cell_dir, ref, cell, keys)
        mad = mean_mad(cell.values())
        name = os.path.basename(cell_dir.rstrip("/"))
        lines.append(f"| {name} | {kc}/{len(keys)} = {100 * kc / len(keys):.1f}% | [{lo:.1f}, {hi:.1f}] | {diff:+.1f} | {b} | {c} | "
                     f"{b + c} ({100 * (b + c) / len(keys):.2f}%) | {mcnemar_exact(b, c):.3g} | {'' if mad is None else f'{mad:.2f}'} | "
                     f"{'' if dist is None else f'{dist:.3f}'} |")
        flips = Counter(key[0] for key in keys if ref[key]["success"] != cell[key]["success"])
        per_task_lines.append(f"{name}: changed episodes per task " + ", ".join(f"t{t}={flips.get(t, 0)}" for t in sorted({key[0] for key in keys})))
    lines += ["", *per_task_lines]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
