"""Run OpenVLA-OFT's original LIBERO evaluation script unchanged and add a per-episode record.

Same idea as run_openvla_libero_logged.py: the wrapper imports experiments/robot/libero/run_libero_eval.py
from the OFT repo and replaces only names that script imported (get_libero_env, save_rollout_video and,
with JPEG_ROUNDTRIP=off, resize_image_for_policy). Every episode is appended to episodes.jsonl in OUT_DIR;
the initial state passed to the environment is checked against the LIBERO init file for (task_id, episode_idx).
With the default initial states, --num_trials_per_task N runs initial states 0..N-1 of every task.

Optional control (see image_pipeline.py):
    JPEG_ROUNDTRIP=off    resize third-person and wrist images without the JPEG round trip.
    SAVE_POLICY_INPUTS=1  save each episode's first third-person and wrist input images as PNGs.
Each record holds jpeg_input_mad (third-person and wrist): the mean absolute difference between the first
input images with and without the JPEG round trip.

Usage: OUT_DIR=/path/to/output python run_oft_libero_logged.py <OFT run_libero_eval.py arguments>
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_logging as el  # noqa: E402
import image_pipeline as ip  # noqa: E402

OFT_ROOT = os.environ.get("OFT_ROOT", "/workspace/openvla-oft")
OUT_DIR = os.path.abspath(os.environ["OUT_DIR"])
sys.path.insert(0, OFT_ROOT)

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
import libero.libero as libero_pkg  # noqa: E402
from libero.libero import benchmark  # noqa: E402

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task_suite_name", default="libero_spatial")
parser.add_argument("--pretrained_checkpoint", default="")
parser.add_argument("--seed", default="7")
parser.add_argument("--num_steps_wait", type=int, default=10)
parser.add_argument("--center_crop", default="True")
parser.add_argument("--initial_states_path", default="DEFAULT")
args, _ = parser.parse_known_args()
assert args.initial_states_path == "DEFAULT", "the initial-state check assumes the default LIBERO init files"
suite = benchmark.get_benchmark_dict()[args.task_suite_name]()


class Tracker:
    task_id = -1
    episode_idx = -1
    expected_states = None
    init_hash = None
    steps = 0
    t0 = None
    fixtures = None
    frame_sha1 = None
    first_input_pending = False
    jpeg_input_mad = None
    policy_input_png = None


T = Tracker()


class EnvRecorder:
    """Thin proxy around the LIBERO env: records the initial state, fixture poses and first frame, counts steps."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def set_init_state(self, state):
        T.episode_idx += 1
        T.init_hash = el.state_hash(state)
        expected = el.state_hash(T.expected_states[T.episode_idx])
        assert T.init_hash == expected, f"Initial state mismatch at task {T.task_id}, episode {T.episode_idx}"
        T.steps = 0
        T.t0 = time.time()
        T.frame_sha1 = None
        T.first_input_pending = True
        T.jpeg_input_mad = None
        T.policy_input_png = None
        obs = self._env.set_init_state(state)
        T.fixtures = el.fixture_poses(self._env.env)
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
_original_resize = rle.resize_image_for_policy
_original_prepare_observation = rle.prepare_observation


def get_libero_env(task, *a, **kw):
    T.task_id += 1
    T.episode_idx = -1
    assert task.name == suite.get_task(T.task_id).name, f"Task order mismatch at task {T.task_id}"
    T.expected_states = suite.get_task_init_states(T.task_id)
    env, task_description = _original_get_libero_env(task, *a, **kw)
    return EnvRecorder(env), task_description


def resize_image_for_policy(img, resize_size):
    if ip.JPEG_ROUNDTRIP == "off":
        return ip.resize_without_jpeg(img, resize_size)
    return _original_resize(img, resize_size)


def prepare_observation(obs, resize_size):
    observation, img = _original_prepare_observation(obs, resize_size)
    if T.first_input_pending:
        T.first_input_pending = False
        raw = {"full_image": rle.get_libero_image(obs), "wrist_image": rle.get_libero_wrist_image(obs)}
        T.jpeg_input_mad = {k: ip.mean_abs_diff(_original_resize(v, resize_size), ip.resize_without_jpeg(v, resize_size))
                            for k, v in raw.items()}
        if ip.SAVE_POLICY_INPUTS:
            T.policy_input_png = {}
            for k, name in (("full_image", "agentview"), ("wrist_image", "wrist")):
                rel = f"policy_inputs/task{T.task_id:02d}_ep{T.episode_idx:02d}_{name}.png"
                ip.save_png(os.path.join(OUT_DIR, rel), observation[k])
                T.policy_input_png[k] = rel
    return observation, img


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
        "fixtures": T.fixtures,
        "frame_sha1": T.frame_sha1,
        "jpeg_input_mad": T.jpeg_input_mad,
        "policy_input_png": T.policy_input_png,
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
        "image_pipeline": ip.settings(args.center_crop),
        "oft_commit": subprocess.run(["git", "-C", OFT_ROOT, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "versions": el.versions(("tensorflow",)),
        "libero_package": libero_dir,
        "textures": el.texture_state(os.path.join(libero_dir, "assets")),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **{k: v for k, v in el.host_info().items() if k != "time"},
    }
    try:
        from huggingface_hub import snapshot_download
        meta["checkpoint_snapshot"] = snapshot_download(args.pretrained_checkpoint, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        meta["checkpoint_snapshot"] = f"unresolved: {e}"
    return meta


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    episodes_path = os.path.join(OUT_DIR, "episodes.jsonl")
    assert not os.path.exists(episodes_path), f"{episodes_path} already exists; use a new OUT_DIR"
    os.chdir(OUT_DIR)  # the original script writes rollout videos relative to the working directory
    with open("run_meta.json", "w") as f:
        json.dump(run_metadata(), f, indent=2)
    EPISODES = open(episodes_path, "a")
    rle.get_libero_env = get_libero_env
    rle.save_rollout_video = save_rollout_video
    rle.resize_image_for_policy = resize_image_for_policy
    rle.prepare_observation = prepare_observation
    rle.eval_libero()
    EPISODES.close()
