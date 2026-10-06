"""Run the original OpenVLA-OFT LIBERO eval on a subset of task ids of one suite.
The original script is imported, never edited; the task suite is wrapped so it only exposes
the task ids listed in TASK_IDS (comma-separated, 0-indexed). Episode i uses initial_states[i]."""
import os, sys

sys.path.insert(0, os.environ.get("EVAL_REPO", "/workspace/openvla-oft"))

import mujoco, robosuite
import experiments.robot.libero.run_libero_eval as rle

TASK_IDS = [int(x) for x in os.environ["TASK_IDS"].split(",")]
print(f"[env] mujoco {mujoco.__version__} | robosuite {robosuite.__version__} | task ids {TASK_IDS}", flush=True)

class TaskSubset:
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
    for name in list(d):
        factory = d[name]
        d[name] = (lambda f=factory: TaskSubset(f(), TASK_IDS))
    return d

rle.benchmark.get_benchmark_dict = get_benchmark_dict_subset

if __name__ == "__main__":
    rle.eval_libero()
