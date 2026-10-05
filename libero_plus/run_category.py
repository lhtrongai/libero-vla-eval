"""Run the original OpenVLA-OFT LIBERO eval on one LIBERO-Plus perturbation category.
The original script is imported, never edited; the task suite is wrapped so it only exposes
the task ids of that category (from LIBERO-Plus task_classification.json)."""
import json, os, sys

sys.path.insert(0, os.environ.get("EVAL_REPO", "/workspace/openvla-oft"))

# Optional: force the attention implementation when the original script hardcodes one we cannot run
if os.environ.get("FORCE_ATTN"):
    import transformers
    _orig_from_pretrained = transformers.AutoModelForVision2Seq.from_pretrained
    def _from_pretrained_forced(*args, **kwargs):
        kwargs["attn_implementation"] = os.environ["FORCE_ATTN"]
        print(f"[attn] forced {os.environ['FORCE_ATTN']}", flush=True)
        return _orig_from_pretrained(*args, **kwargs)
    transformers.AutoModelForVision2Seq.from_pretrained = _from_pretrained_forced

import experiments.robot.libero.run_libero_eval as rle

CATEGORY = os.environ.get("LP_CATEGORY", "Camera Viewpoints")
LIMIT = int(os.environ.get("LP_LIMIT", "0"))

class CategorySubset:
    def __init__(self, suite, ids):
        self._suite, self._ids = suite, ids
    @property
    def n_tasks(self):
        return len(self._ids)
    def get_num_tasks(self):
        return len(self._ids)
    def get_task(self, i):
        return self._suite.get_task(self._ids[i])
    def get_task_init_states(self, i):
        return self._suite.get_task_init_states(self._ids[i])
    def __getattr__(self, name):
        return getattr(self._suite, name)

_original_get_benchmark_dict = rle.benchmark.get_benchmark_dict

def get_benchmark_dict_subset(*args, **kwargs):
    d = dict(_original_get_benchmark_dict(*args, **kwargs))
    bench_dir = os.path.dirname(rle.benchmark.__file__)
    cls = json.load(open(os.path.join(bench_dir, "task_classification.json")))
    for name in list(d):
        if name in cls:
            ids = [t["id"] - 1 for t in cls[name] if t["category"] == CATEGORY]
            if LIMIT:
                ids = ids[:LIMIT]
            factory = d[name]
            d[name] = (lambda f=factory, ids=ids: CategorySubset(f(), ids))
            print(f"[category] {name}: {CATEGORY}, {len(ids)} tasks, ids {ids[0]}-{ids[-1]}", flush=True)
    return d

rle.benchmark.get_benchmark_dict = get_benchmark_dict_subset

if __name__ == "__main__":
    rle.eval_libero()
