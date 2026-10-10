"""Run LeRobot's LIBERO evaluation unchanged and log what defines every episode.

Three files are written next to each other (paths from environment variables):

RESET_LOG (required), one JSON line per LiberoEnv.reset call:
    task_id, sub_env, init_state_idx, init_state_sha1 - which fixed initial state the episode starts from
        (hash function shared with the OpenVLA wrapper, so episodes pair across models).
    caller - "explicit" (an episode that is counted) or "autoreset" (from inside the vector env's step;
        that episode is discarded by lerobot-eval).
    seed - seed lerobot-eval passed to the reset; seed_used - seed given to LIBERO after SCENE_SEED_SHIFT.
    fixtures - body_pos / body_quat of every fixture after the reset (cabinet, stove, wine rack, ...).
    frame_sha1 - hash of each camera image returned by the reset, i.e. the first frame the policy sees.
    torch_rng_sha1, cuda_rng_sha1 - generator states before the episode's first noise draw.
NOISE_LOG (default: RESET_LOG name + "_noise.jsonl"), one JSON line per flow-matching noise draw:
    task_id, batch, batch_call_ix, shape, sha1 of the whole batch; for the first draw after an explicit
    reset also row_sha1 (one hash per sub-env), i.e. the initial noise of each episode.
RUN_META (default: RESET_LOG name + "_meta.json"): argv, package versions, LeRobot commit, policy revision
    and weight hash, LIBERO assets folder and the sRGB tag of its floor textures, GPU, host.

Optional controls:
    SCENE_SEED_SHIFT=k  add k to every explicit reset seed. LIBERO draws the fixture placement from the reset
        seed and SmolVLA draws its noise from --seed, so --seed=1001 with SCENE_SEED_SHIFT=-1 runs seed-1001
        policy noise on exactly the scenes of a --seed=1000 run (same body_pos/body_quat for every fixture).
    LIBERO_ASSETS_DIR=path  load LIBERO assets from this folder instead of the one hf-libero resolves,
        e.g. a copy whose floor textures had their sRGB tag removed (the look of MuJoCo < 3.3.3).

Requires synchronous vector envs and one task at a time (the hooks run in this process):
    --eval.use_async_envs=false, and --env.max_parallel_tasks left at 1.

Usage: RESET_LOG=/path/resets.jsonl python run_lerobot_libero_logged.py <lerobot-eval arguments>
"""
import atexit
import inspect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_logging as el  # noqa: E402


def cli_value(name, default=None):
    """Value of --name=value or --name value in sys.argv (draccus style), else default."""
    flag = f"--{name}"
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


RESET_LOG = os.path.abspath(os.environ["RESET_LOG"])
_base = RESET_LOG[: -len(".jsonl")] if RESET_LOG.endswith(".jsonl") else RESET_LOG
NOISE_LOG = os.path.abspath(os.environ.get("NOISE_LOG", _base + "_noise.jsonl"))
RUN_META = os.path.abspath(os.environ.get("RUN_META", _base + "_meta.json"))
SCENE_SEED_SHIFT = int(os.environ.get("SCENE_SEED_SHIFT", "0"))
ASSETS_DIR = os.environ.get("LIBERO_ASSETS_DIR")

if str(cli_value("eval.use_async_envs", "true")).lower() not in ("false", "0", "no"):
    sys.exit("run_lerobot_libero_logged.py: pass --eval.use_async_envs=false; with async envs the reset hook "
             "runs in worker processes and nothing would be logged.")
if int(cli_value("env.max_parallel_tasks", "1")) != 1:
    sys.exit("run_lerobot_libero_logged.py: keep --env.max_parallel_tasks=1; noise draws are attributed to the "
             "task whose batch was reset last.")
for path in (RESET_LOG, NOISE_LOG, RUN_META):
    if os.path.exists(path):
        sys.exit(f"run_lerobot_libero_logged.py: {path} already exists; use new paths for every run.")

import libero.libero as libero_pkg  # noqa: E402

if ASSETS_DIR:
    if not os.path.isdir(os.path.join(ASSETS_DIR, "textures")):
        sys.exit(f"run_lerobot_libero_logged.py: LIBERO_ASSETS_DIR={ASSETS_DIR} has no textures/ folder.")
    # hf-libero resolves every asset path through get_assets_path(), which returns this cache when it exists.
    libero_pkg._assets_path_cache = os.path.abspath(ASSETS_DIR)

import lerobot  # noqa: E402
import lerobot.envs.libero as libero_env  # noqa: E402
import lerobot.policies.smolvla.modeling_smolvla as smolvla_modeling  # noqa: E402

RESETS = open(RESET_LOG, "a")
NOISE = open(NOISE_LOG, "a")
STATE = {"reset_ix": 0, "explicit": 0, "autoreset": 0, "batch": -1, "task_id": None, "batch_call_ix": None,
         "noise_calls": 0}
META = {}


def write_meta(final=False):
    if final:
        META["end"] = el.host_info()["time"]
        META["counts"] = {k: STATE[k] for k in ("explicit", "autoreset", "noise_calls")}
        META["counts"]["batches"] = STATE["batch"] + 1
        META["policy"] = policy_identity(cli_value("policy.path"))
    el.write_json(RUN_META, META)


def policy_identity(path):
    """Revision (Hub commit) and weight hash of the checkpoint lerobot-eval loaded."""
    info = {"path": path, "revision": None, "local_dir": None, "model_sha1": None}
    if not path:
        return info
    local = path if os.path.isdir(path) else None
    if local is None:
        try:
            from huggingface_hub import snapshot_download
            local = snapshot_download(path, local_files_only=True)
            info["revision"] = os.path.basename(os.path.normpath(local))
        except Exception as e:  # noqa: BLE001
            info["revision"] = f"unresolved: {e}"
    info["local_dir"] = local
    weights = os.path.join(local, "model.safetensors") if local else None
    if weights and os.path.exists(weights):
        info["model_sha1"] = el.file_sha1(weights)
    return info


def logged_reset(self, seed=None, **kwargs):
    states = self._init_states if self.init_states else None
    idx = self.init_state_id % len(states) if states is not None else None
    caller = "autoreset" if any(f.function == "step" for f in inspect.stack()[1:8]) else "explicit"
    seed_used = seed + SCENE_SEED_SHIFT if seed is not None and caller == "explicit" else seed
    rng = el.torch_rng_hashes()
    out = _original_reset(self, seed=seed_used, **kwargs)
    observation = out[0]
    if caller == "explicit" and self.episode_index == 0:  # first sub-env of a new batch
        STATE["batch"] += 1
        STATE["batch_call_ix"] = 0
        STATE["task_id"] = self.task_id
    STATE[caller] += 1
    record = {
        "task_id": self.task_id,
        "sub_env": self.episode_index,
        "init_state_idx": idx,
        "init_state_sha1": el.state_hash(states[idx]) if states is not None else None,
        "caller": caller,
        "seed": seed,
        "seed_used": seed_used,
        "scene_seed_shift": SCENE_SEED_SHIFT,
        "batch": STATE["batch"],
        "reset_ix": STATE["reset_ix"],
        "fixtures": el.fixture_poses(self._env.env),
        "frame_sha1": el.frame_hashes(observation.get("pixels", {})),
        "torch_rng_sha1": rng["cpu"],
        "cuda_rng_sha1": rng["cuda"],
        "time": round(time.time(), 3),
    }
    STATE["reset_ix"] += 1
    RESETS.write(json.dumps(record) + "\n")
    RESETS.flush()
    return out


def logged_sample_noise(self, shape, device):
    noise = _original_sample_noise(self, shape, device)
    values = noise.detach().float().cpu().numpy()
    record = {
        "call_ix": STATE["noise_calls"],
        "task_id": STATE["task_id"],
        "batch": STATE["batch"],
        "batch_call_ix": STATE["batch_call_ix"],
        "shape": list(values.shape),
        "sha1": el.array_sha1(values),
    }
    if STATE["batch_call_ix"] == 0:
        record["row_sha1"] = [el.array_sha1(row) for row in values]
    if STATE["batch_call_ix"] is not None:
        STATE["batch_call_ix"] += 1
    STATE["noise_calls"] += 1
    NOISE.write(json.dumps(record) + "\n")
    NOISE.flush()
    return noise


_original_reset = libero_env.LiberoEnv.reset
_original_sample_noise = smolvla_modeling.VLAFlowMatching.sample_noise
libero_env.LiberoEnv.reset = logged_reset
smolvla_modeling.VLAFlowMatching.sample_noise = logged_sample_noise

assets_in_use = libero_pkg.get_assets_path()
lerobot_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(lerobot.__file__))))
META.update({
    "argv": sys.argv[1:],
    "start": el.host_info(),
    "versions": el.versions(),
    "packages": {name: el.package_version(name) for name in ("lerobot", "hf-libero", "huggingface-hub")},
    "lerobot": {"path": os.path.dirname(os.path.abspath(lerobot.__file__)), "commit": el.git_commit(lerobot_root),
                "uncommitted_changes": el.git_dirty(lerobot_root)},
    "libero_package": os.path.dirname(os.path.abspath(libero_pkg.__file__)),
    "textures": el.texture_state(assets_in_use),
    "seed": cli_value("seed"),
    "scene_seed_shift": SCENE_SEED_SHIFT,
    "logs": {"resets": RESET_LOG, "noise": NOISE_LOG},
})
write_meta()
atexit.register(write_meta, final=True)

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
