"""Compare two runs episode by episode: same scenes? same initial noise?

Inputs are reset logs written by run_lerobot_libero_logged.py or scene manifests from scene_manifest.py
(any mix). Episodes are matched by (task_id, init_state_idx); only counted episodes are used
(caller "explicit"; manifests have no caller field). If a reset log has its "_noise.jsonl" companion next to
it, the noise draws are compared too.

What each check means:
    fixtures identical      - same fixture placement (exact float equality of body_pos and body_quat).
    first frames identical  - same rendered first frame (only meaningful on the same kind of machine).
    initial noise identical - the first flow-matching noise draw of the episode is the same.
    noise streams identical - every noise draw of the task's batch is the same, in order.

Typical reads: a rerun with the same seed should match on all four; a policy-seed change with
SCENE_SEED_SHIFT pinning should match on fixtures and first frames and differ on noise.

Usage: python compare_episode_logs.py A_resets.jsonl B_resets.jsonl
"""
import json
import os
import sys

import numpy as np


def load_resets(path):
    with open(path) as f:
        rows = [r for r in map(json.loads, f) if r.get("caller", "explicit") == "explicit"]
    return {(r["task_id"], r["init_state_idx"]): r for r in rows}


def noise_path(reset_path):
    base = reset_path[: -len(".jsonl")] if reset_path.endswith(".jsonl") else reset_path
    path = base + "_noise.jsonl"
    return path if os.path.exists(path) else None


def load_noise(path):
    """rows: {(batch, sub_env): sha1 of that episode's first draw}; streams: {task_id: [sha1 per draw in order]}"""
    rows, streams = {}, {}
    with open(path) as f:
        for r in map(json.loads, f):
            streams.setdefault(r["task_id"], []).append(r["sha1"])
            for i, h in enumerate(r.get("row_sha1") or []):
                rows[(r["batch"], i)] = h
    return rows, streams


def main(path_a, path_b):
    a, b = load_resets(path_a), load_resets(path_b)
    keys = sorted(set(a) & set(b))
    print(f"episodes matched by (task, initial state): {len(keys)} (A has {len(a)}, B has {len(b)})")
    same_fix, same_frame, n_frame, shifts = 0, 0, 0, []
    per_task = {}
    for key in keys:
        fa, fb = a[key]["fixtures"], b[key]["fixtures"]
        identical = fa == fb
        same_fix += identical
        if fa and fb:
            d = max(1000 * float(np.linalg.norm(np.subtract(fa[n]["pos"][:2], fb[n]["pos"][:2]))) for n in fa)
            shifts.append(d)
        if a[key].get("frame_sha1") and b[key].get("frame_sha1"):
            n_frame += 1
            same_frame += a[key]["frame_sha1"] == b[key]["frame_sha1"]
        t = per_task.setdefault(key[0], [0, 0])
        t[0] += identical
        t[1] += 1
    print(f"fixtures identical: {same_fix}/{len(keys)}"
          + (f"; largest fixture xy shift per episode: median {np.median(shifts):.1f} mm, max {max(shifts):.1f} mm"
             if shifts else "; no fixtures in these tasks"))
    print("  per task: " + ", ".join(f"{t}: {v[0]}/{v[1]}" for t, v in sorted(per_task.items())))
    print(f"first frames identical: {same_frame}/{n_frame}")

    na, nb = noise_path(path_a), noise_path(path_b)
    if not (na and nb):
        print("noise: not compared (no _noise.jsonl next to both logs)")
        return
    (rows_a, streams_a), (rows_b, streams_b) = load_noise(na), load_noise(nb)
    same_init = n_init = 0
    for key in keys:
        ra, rb = a[key], b[key]
        ha = rows_a.get((ra.get("batch"), ra.get("sub_env")))
        hb = rows_b.get((rb.get("batch"), rb.get("sub_env")))
        if ha and hb:
            n_init += 1
            same_init += ha == hb
    print(f"initial noise identical: {same_init}/{n_init}")
    tasks = sorted(set(streams_a) & set(streams_b))
    same_stream = sum(streams_a[t] == streams_b[t] for t in tasks)
    print(f"noise streams identical: {same_stream}/{len(tasks)} tasks "
          f"(draws per task A: {[len(streams_a[t]) for t in tasks]}, B: {[len(streams_b[t]) for t in tasks]})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
