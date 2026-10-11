"""Summarise the four start-pose arms written by run_lerobot_variants.py (standard library only).

Usage: python summarize_start_pose_arms.py <run dir prefix>
       (reads <prefix><arm>/resets_episodes.jsonl for arm in no_shift, as_is, debias_state, homing)

Prints: success with Wilson 95% CI per arm; paired comparisons (same variant and seed) with the
discordant counts, an exact McNemar p-value and the paired difference with a 95% CI; success per base
task; as-is and homing success by size of the start-pose offset; homing residuals.
"""
import json
import math
import os
import sys

ARMS = ("no_shift", "as_is", "debias_state", "homing")
PAIRS = (("as_is", "no_shift"), ("debias_state", "no_shift"), ("homing", "no_shift"),
         ("debias_state", "as_is"), ("homing", "as_is"))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_diff_ci(b, c, n, z=1.96):
    """Difference in success (first minus second) for paired binary data, Wald CI on the discordant pairs."""
    d = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return 100 * d, 100 * (d - z * se), 100 * (d + z * se)


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["variant_id"], r["seed"])] = r
    return rows


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/lp_robot_start_pose_smolvla_spatial_"
    data = {}
    for arm in ARMS:
        path = f"{prefix}{arm}/resets_episodes.jsonl"
        if os.path.exists(path):
            data[arm] = load(path)
    print("== success per arm (Wilson 95% CI)")
    for arm, rows in data.items():
        k, n = sum(r["success"] for r in rows.values()), len(rows)
        p, lo, hi = wilson(k, n)
        print(f"{arm:13s} {k:4d}/{n:<4d} {p:5.1f}  [{lo:5.1f}, {hi:5.1f}]")

    print("\n== paired (same variant, same seed): A vs B")
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        keys = sorted(set(data[a]) & set(data[b]))
        n = len(keys)
        only_a = sum(data[a][k]["success"] and not data[b][k]["success"] for k in keys)
        only_b = sum(data[b][k]["success"] and not data[a][k]["success"] for k in keys)
        d, lo, hi = paired_diff_ci(only_a, only_b, n)
        print(f"{a:13s} vs {b:9s} n={n:3d}  only A {only_a:3d}  only B {only_b:3d}  "
              f"diff {d:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]  McNemar p={mcnemar_exact(only_a, only_b):.2g}")

    if "no_shift" in data:
        print("\n== success per base task (n per task)")
        tasks = sorted({r["base_task"] for r in data["no_shift"].values()})
        print("task".ljust(46) + "".join(f"{a[:8]:>10s}" for a in data))
        for t in tasks:
            line = t.replace("pick_up_the_black_bowl_", "")[:44].ljust(46)
            for arm, rows in data.items():
                sel = [r for r in rows.values() if r["base_task"] == t]
                line += f"{sum(r['success'] for r in sel):>5d}/{len(sel):<4d}"
            print(line)

    ref = data.get("as_is") or data.get("homing")
    if ref:
        print("\n== success by start-pose offset (position, mm), terciles over the as-is arm")
        offs = sorted(r["offset_pos_m"] for r in ref.values())
        cuts = [offs[len(offs) // 3], offs[2 * len(offs) // 3]]
        bins = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9)]
        for lo_m, hi_m in bins:
            label = f"{lo_m * 1000:5.1f}-{hi_m * 1000:5.1f}" if hi_m < 1e8 else f"{lo_m * 1000:5.1f}+    "
            line = f"{label} mm"
            for arm in ("no_shift", "as_is", "debias_state", "homing"):
                if arm not in data:
                    continue
                sel = [data[arm][k] for k, r in ref.items() if lo_m <= r["offset_pos_m"] < hi_m and k in data[arm]]
                line += f"  {arm[:8]} {sum(r['success'] for r in sel):3d}/{len(sel):<3d}"
            print(line)

    if "homing" in data:
        h = list(data["homing"].values())
        pos = sorted(r["homing_pos_err_m"] * 1000 for r in h)
        rot = sorted(r["homing_rot_err_rad"] for r in h)
        steps = sorted(r["homing_steps"] for r in h)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
        print("\n== homing residuals (median / 90th percentile / max)")
        print(f"position mm {q(pos, .5):.2f} / {q(pos, .9):.2f} / {pos[-1]:.2f}; rotation rad {q(rot, .5):.4f} / "
              f"{q(rot, .9):.4f} / {rot[-1]:.4f}; controller steps {q(steps, .5)} / {q(steps, .9)} / {steps[-1]}")
        bad = [r for r in h if r["homing_pos_err_m"] > 0.01 or r["homing_rot_err_rad"] > 0.05]
        print(f"episodes with residual above 10 mm or 0.05 rad: {len(bad)} "
              f"(success {sum(r['success'] for r in bad)}/{len(bad)})")


if __name__ == "__main__":
    main()
