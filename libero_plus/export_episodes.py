"""Export per-episode results of LIBERO-Plus category runs (OpenVLA-OFT logs) to CSV."""
import csv, glob, json, os, re, sys
from collections import defaultdict

CATEGORY = os.environ.get("LP_CATEGORY", "Camera Viewpoints")
BENCH = "/workspace/LIBERO-plus/libero/libero/benchmark/task_classification.json"
rows_meta = [t for t in json.load(open(BENCH))["libero_spatial"] if t["category"] == CATEGORY]

for run_dir in sys.argv[1:]:
    log = sorted(glob.glob(os.path.join(run_dir, "EVAL-*.txt")))[-1]
    tasks, results = [], []
    for line in open(log, encoding="utf-8"):
        line = line.strip()
        if line.startswith("Task: "):
            tasks.append(line[len("Task: "):])
        elif line.startswith("Success: "):
            results.append(line.endswith("True"))
    assert len(tasks) == len(results) == len(rows_meta), (log, len(tasks), len(results), len(rows_meta))
    out = os.path.join(run_dir, "episodes.csv")
    per_base = defaultdict(lambda: [0, 0])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        extra = [k for k in rows_meta[0] if k not in ("id", "category")]
        w.writerow(["libero_plus_id", "base_task", "camera_params", "success"] + extra)
        for meta, desc, ok in zip(rows_meta, tasks, results):
            m = re.match(r"(.*?) view (.*)$", desc)
            base, params = (m.group(1), m.group(2)) if m else (desc, "")
            w.writerow([meta["id"], base, params, int(ok)] + [meta[k] for k in extra])
            per_base[base][0] += ok
            per_base[base][1] += 1
    print(f"\n== {os.path.basename(run_dir)}: {sum(results)}/{len(results)} -> {out}")
    for base, (k, n) in sorted(per_base.items()):
        print(f"  {k:3d}/{n:3d}  {base}")
