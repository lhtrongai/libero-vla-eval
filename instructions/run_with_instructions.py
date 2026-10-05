"""Run the original OpenVLA-OFT LIBERO eval with an optional instruction replacement (INSTRUCTION_MAP)
and an optional initial-state offset (INIT_OFFSET). The original script is imported, never edited."""
import json, os, sys

sys.path.insert(0, "/workspace/openvla-oft")
import experiments.robot.libero.run_libero_eval as rle
from libero.libero.benchmark import Benchmark

if "--initial_states_path" in " ".join(sys.argv):
    raise SystemExit("Custom initial-state files key on the instruction text; use the default initial states.")

MAP_PATH = os.environ.get("INSTRUCTION_MAP")
MAPPING = json.load(open(MAP_PATH, encoding="utf-8"))["map"] if MAP_PATH else None
OFFSET = int(os.environ.get("INIT_OFFSET", "0"))

if MAPPING is not None:
    _original_get_libero_env = rle.get_libero_env
    def get_libero_env_swapped(task, *args, **kwargs):
        env, description = _original_get_libero_env(task, *args, **kwargs)
        if description not in MAPPING:
            raise KeyError(f"No replacement for instruction: {description!r}")
        print(f"[instruction swap] {description!r} -> {MAPPING[description]!r}", flush=True)
        return env, MAPPING[description]
    rle.get_libero_env = get_libero_env_swapped

if OFFSET:
    _original_init_states = Benchmark.get_task_init_states
    def get_task_init_states_offset(self, i):
        states = _original_init_states(self, i)
        print(f"[init offset] task {i}: initial states {OFFSET}-{len(states) - 1}", flush=True)
        return states[OFFSET:]
    Benchmark.get_task_init_states = get_task_init_states_offset

if __name__ == "__main__":
    rle.eval_libero()
