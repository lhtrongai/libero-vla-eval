"""Run OpenVLA's original LIBERO evaluation script unchanged and add a per-episode record.

The wrapper imports experiments/robot/libero/run_libero_eval.py and replaces only two names
that script imported: get_libero_env (to observe set_init_state and step calls) and
save_rollout_video (called once per episode with the success flag). Every episode is appended
to episodes.jsonl in OUT_DIR. The initial state passed to the environment is checked against
the LIBERO init file for (task_id, episode_idx), so paired comparisons rest on a verified match.

Each episode record also holds:
    env_seed - seed of the LIBERO env (the original script uses env.seed(0) once per task).
    fixtures - body_pos / body_quat of every fixture after the reset (cabinet, stove, wine rack, ...);
        LIBERO samples them in env.reset() and set_init_state() does not restore them.
    frame_sha1 - hash of each camera image after the settle steps, i.e. the first frame the policy sees.
run_meta.json additionally records the LIBERO package in use and the sRGB tag of its floor textures
(MuJoCo >= 3.3.3 renders tagged textures darker than the look LIBERO's training data was rendered with).

Optional control:
    ENV_SEED=k  seed the LIBERO env with k instead of 0, right after the original get_libero_env.
        Only the fixture placement changes; the fixed initial states and the policy stay the same.

Usage: OUT_DIR=/path/to/output python run_openvla_libero_logged.py <run_libero_eval.py arguments>
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_logging as el  # noqa: E402

OPENVLA_ROOT = os.environ.get("OPENVLA_ROOT", "/workspace/openvla")
OUT_DIR = os.path.abspath(os.environ["OUT_DIR"])
ENV_SEED = os.environ.get("ENV_SEED")
sys.path.insert(0, OPENVLA_ROOT)

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
import libero.libero as libero_pkg  # noqa: E402
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
parser.add_argument("--num_steps_wait", type=int, default=10)
args, _ = parser.parse_known_args()
suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
state_hash = el.state_hash


class Tracker:
    task_id = -1
    episode_idx = -1
    expected_states = None
    init_hash = None
    steps = 0
    t0 = None
    env_seed = 0
    fixtures = None
    frame_sha1 = None


T = Tracker()


class EnvRecorder:
    """Thin proxy around the LIBERO env: records the initial state, fixture poses and first frame, counts steps."""

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
        T.frame_sha1 = None
        obs = self._env.set_init_state(state)
        T.fixtures = el.fixture_poses(self._env.env)  # placed by the env.reset() just before this call
        return obs

    def step(self, action):
        T.steps += 1
        out = self._env.step(action)
        if T.steps == args.num_steps_wait:  # last settle step: its observation is the policy's first input
            obs = out[0]
            T.frame_sha1 = el.frame_hashes({k: obs[k] for k in ("agentview_image", "robot0_eye_in_hand_image")
                                            if k in obs})
        return out


_original_get_libero_env = rle.get_libero_env
_original_save_rollout_video = rle.save_rollout_video


def get_libero_env(task, *a, **kw):
    T.task_id += 1
    T.episode_idx = -1
    assert task.name == suite.get_task(T.task_id).name, f"Task order mismatch at task {T.task_id}"
    T.expected_states = suite.get_task_init_states(T.task_id)
    env, task_description = _original_get_libero_env(task, *a, **kw)  # calls env.seed(0)
    if ENV_SEED is not None:
        env.seed(int(ENV_SEED))  # replaces the seed before any reset, so only the fixture draws change
    T.env_seed = int(ENV_SEED) if ENV_SEED is not None else 0
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
        "env_seed": T.env_seed,
        "fixtures": T.fixtures,
        "frame_sha1": T.frame_sha1,
    }
    EPISODES.write(json.dumps(record) + "\n")
    EPISODES.flush()
    return _original_save_rollout_video(rollout_images, idx, success, task_description, log_file=log_file)


def run_metadata():
    libero_dir = os.path.dirname(os.path.abspath(libero_pkg.__file__))
    meta = {
        "argv": sys.argv[1:],
        "suite": args.task_suite_name,
        "checkpoint": args.pretrained_checkpoint,
        "seed": args.seed,
        "env_seed": int(ENV_SEED) if ENV_SEED is not None else 0,
        "attn_implementation": ATTN_IMPL,
        "openvla_commit": subprocess.run(["git", "-C", OPENVLA_ROOT, "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
        "versions": el.versions(),
        "libero_package": libero_dir,
        "textures": el.texture_state(os.path.join(libero_dir, "assets")),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **{k: v for k, v in el.host_info().items() if k != "time"},
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
