"""Run OpenVLA's original LIBERO evaluation script unchanged and add a per-episode record.

The wrapper imports experiments/robot/libero/run_libero_eval.py and replaces only two names
that script imported: get_libero_env (to observe set_init_state and step calls) and
save_rollout_video (called once per episode with the success flag). Every episode is appended
to episodes.jsonl in OUT_DIR. The initial state passed to the environment is checked against
the LIBERO init file for (task_id, episode_idx), so paired comparisons rest on a verified match.

Usage: OUT_DIR=/path/to/output python run_openvla_libero_logged.py <run_libero_eval.py arguments>
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

OPENVLA_ROOT = os.environ.get("OPENVLA_ROOT", "/workspace/openvla")
OUT_DIR = os.path.abspath(os.environ["OUT_DIR"])
sys.path.insert(0, OPENVLA_ROOT)

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from libero.libero import benchmark  # noqa: E402

# The original get_vla() requests flash_attention_2. Force the attention kernel from ATTN_IMPL
# (default sdpa) so every run uses one recorded kernel; the evaluation script itself is unchanged.
import transformers  # noqa: E402

ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")
_original_from_pretrained = transformers.AutoModelForVision2Seq.from_pretrained


def _from_pretrained_with_attn(*a, **kw):
    kw["attn_implementation"] = ATTN_IMPL
    return _original_from_pretrained(*a, **kw)


transformers.AutoModelForVision2Seq.from_pretrained = _from_pretrained_with_attn

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task_suite_name", default="libero_spatial")
parser.add_argument("--pretrained_checkpoint", default="")
parser.add_argument("--seed", default="7")
args, _ = parser.parse_known_args()
suite = benchmark.get_benchmark_dict()[args.task_suite_name]()


def state_hash(state):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(state, dtype=np.float64)).tobytes()).hexdigest()


class Tracker:
    task_id = -1
    episode_idx = -1
    expected_states = None
    init_hash = None
    steps = 0
    t0 = None


T = Tracker()


class EnvRecorder:
    """Thin proxy around the LIBERO env: records the initial state and counts env steps."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def set_init_state(self, state):
        T.episode_idx += 1
        T.init_hash = state_hash(state)
        expected = state_hash(T.expected_states[T.episode_idx])
        assert T.init_hash == expected, f"Initial state mismatch at task {T.task_id}, episode {T.episode_idx}"
        T.steps = 0
        T.t0 = time.time()
        return self._env.set_init_state(state)

    def step(self, action):
        T.steps += 1
        return self._env.step(action)


_original_get_libero_env = rle.get_libero_env
_original_save_rollout_video = rle.save_rollout_video


def get_libero_env(task, *a, **kw):
    T.task_id += 1
    T.episode_idx = -1
    assert task.name == suite.get_task(T.task_id).name, f"Task order mismatch at task {T.task_id}"
    T.expected_states = suite.get_task_init_states(T.task_id)
    env, task_description = _original_get_libero_env(task, *a, **kw)
    return EnvRecorder(env), task_description


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    record = {
        "suite": args.task_suite_name,
        "task_id": T.task_id,
        "task_description": task_description,
        "episode_idx": T.episode_idx,
        "init_state_sha1": T.init_hash,
        "success": bool(success),
        "env_steps": T.steps,
        "wall_time_s": round(time.time() - T.t0, 2),
        "global_episode": idx,
    }
    EPISODES.write(json.dumps(record) + "\n")
    EPISODES.flush()
    return _original_save_rollout_video(rollout_images, idx, success, task_description, log_file=log_file)


def run_metadata():
    import mujoco
    import robosuite
    import torch
    import transformers

    meta = {
        "argv": sys.argv[1:],
        "suite": args.task_suite_name,
        "checkpoint": args.pretrained_checkpoint,
        "seed": args.seed,
        "attn_implementation": ATTN_IMPL,
        "openvla_commit": subprocess.run(["git", "-C", OPENVLA_ROOT, "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "mujoco": mujoco.__version__, "robosuite": robosuite.__version__, "numpy": np.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        from huggingface_hub import snapshot_download
        meta["checkpoint_snapshot"] = snapshot_download(args.pretrained_checkpoint, local_files_only=True)
    except Exception as e:
        meta["checkpoint_snapshot"] = f"unresolved: {e}"
    return meta


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    episodes_path = os.path.join(OUT_DIR, "episodes.jsonl")
    assert not os.path.exists(episodes_path), f"{episodes_path} already exists; use a new OUT_DIR"
    os.chdir(OUT_DIR)
    with open("run_meta.json", "w") as f:
        json.dump(run_metadata(), f, indent=2)
    EPISODES = open(episodes_path, "a")
    rle.get_libero_env = get_libero_env
    rle.save_rollout_video = save_rollout_video
    rle.eval_libero()
    EPISODES.close()
