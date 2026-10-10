"""Run lerobot-eval on LIBERO-Plus variants: one rollout per variant, many different variants per batch.

LeRobot builds one vector env per task with `batch_size` copies of it, and copy k plays initial state k.
A LIBERO-Plus variant (for example a robot-initial-state variant of a Spatial task) is meant to be played
once, from initial state 0 of its base task, as in the LIBERO-Plus evaluation. Running it that way through
plain lerobot-eval needs batch 1, so every episode runs alone. This script instead puts `batch_size`
different variants into one vector env (sub-env k = variant k of the group, always initial state 0), so a
batch of 50 plays 50 variants at once. Everything else is lerobot-eval unchanged, including the policy and
its pre/post-processing; per-episode logging comes from run_lerobot_libero_logged.py.

Arms (ARM=...):
  as_is         the variant as LIBERO-Plus defines it.
  no_shift      the same variant with the robot perturbation removed (`_initstate_<k>` -> `_initstate_0`):
                same base task, same initial state, same reset seed, default robot. Pairs episode by episode
                with the other arms.
  debias_state  as_is, but the state given to the policy has the start-pose offset removed: with p0, R0 the
                end-effector pose at the first observation and pc, Rc the pose of the no_shift arm at its first
                observation (same base task), every observation reports p - (p0 - pc) and (R0 Rc^T)^T R.
                Images, actions and physics are untouched.
  homing        as_is, then before the policy sees anything the arm is driven back to the no_shift start pose
                (pc, Rc) by the robot's own controller (proportional end-effector commands, gripper open),
                then settled. Not counted in the policy's step budget. Residual errors are logged.

Environment variables (besides those of run_lerobot_libero_logged.py: RESET_LOG required):
  LIBERO_PLUS_ROOT   LIBERO-Plus checkout (default /workspace/LIBERO-plus). Loaded as the `libero` package even
                     when hf-libero is installed in the same environment (hf-libero would win otherwise).
  LP_CATEGORY        task_classification.json category (default "Robot Initial States").
  LP_IDS             comma-separated variant task ids (0-based) instead of a category; LP_LIMIT keeps the first n.
  ARM                as above (default as_is).
  STRIP_SUFFIX       1 (default): the instruction is the LIBERO-Plus language with its metadata suffix
                     (" view ... initstate ...") removed; 0 keeps it.
  EPISODES_LOG       per-episode results (default: RESET_LOG name + "_episodes.jsonl").
  HOMING_MAX_STEPS (60), HOMING_POS_TOL (0.002 m), HOMING_ROT_TOL (0.02 rad), HOMING_SETTLE (10).

Constraints checked at start: --eval.use_async_envs=false, --eval.n_episodes equal to --eval.batch_size,
and the number of selected variants a multiple of the batch size (each group is one batch, played once).

Usage:
  RESET_LOG=/path/resets.jsonl ARM=as_is python libero_plus/run_lerobot_variants.py \
      --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero_plus --env.task=libero_spatial \
      --eval.batch_size=50 --eval.n_episodes=50 --eval.use_async_envs=false --seed=1000 ...
"""
import atexit
import json
import os
import re
import sys
import types
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

ARM = os.environ.get("ARM", "as_is")
ARMS = ("as_is", "no_shift", "debias_state", "homing")
if ARM not in ARMS:
    sys.exit(f"run_lerobot_variants.py: ARM must be one of {ARMS}, got {ARM!r}.")
STRIP_SUFFIX = os.environ.get("STRIP_SUFFIX", "1") == "1"
CATEGORY = os.environ.get("LP_CATEGORY", "Robot Initial States")
LP_IDS = os.environ.get("LP_IDS")
LIMIT = int(os.environ.get("LP_LIMIT", "0"))
HOMING_MAX_STEPS = int(os.environ.get("HOMING_MAX_STEPS", "60"))
HOMING_POS_TOL = float(os.environ.get("HOMING_POS_TOL", "0.002"))
HOMING_ROT_TOL = float(os.environ.get("HOMING_ROT_TOL", "0.02"))
HOMING_SETTLE = int(os.environ.get("HOMING_SETTLE", "10"))


def cli_value(name, default=None):
    flag = f"--{name}"
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


batch_size = int(cli_value("eval.batch_size", "0"))
n_episodes = int(cli_value("eval.n_episodes", "0"))
if batch_size <= 0 or n_episodes != batch_size:
    sys.exit("run_lerobot_variants.py: pass --eval.batch_size=B and --eval.n_episodes=B (each group of B "
             "variants is one batch, played once).")
if cli_value("env.type") != "libero_plus":
    sys.exit("run_lerobot_variants.py: pass --env.type=libero_plus.")

# --- Load LIBERO-Plus as the `libero` package. Its top folder has no __init__.py (namespace package), so a
# regular `libero` package from hf-libero in site-packages would be imported instead even with PYTHONPATH set.
LP_ROOT = os.environ.get("LIBERO_PLUS_ROOT", "/workspace/LIBERO-plus")
if not os.path.isdir(os.path.join(LP_ROOT, "libero", "libero")):
    sys.exit(f"run_lerobot_variants.py: LIBERO_PLUS_ROOT={LP_ROOT} is not a LIBERO-Plus checkout.")
_pkg = types.ModuleType("libero")
_pkg.__path__ = [os.path.join(LP_ROOT, "libero")]
sys.modules["libero"] = _pkg
import libero.libero as libero_pkg  # noqa: E402

if libero_pkg.__file__ != os.path.join(LP_ROOT, "libero", "libero", "__init__.py"):
    sys.exit(f"run_lerobot_variants.py: libero.libero came from {libero_pkg.__file__}, not LIBERO-Plus.")
if not hasattr(libero_pkg, "get_assets_path"):  # hf-libero API used by the logging runner
    libero_pkg.get_assets_path = lambda: libero_pkg.get_libero_path("assets")

import gymnasium as gym  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import lerobot.envs.libero as libero_env  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, FreezeAfterEpisodeEnd  # noqa: E402

SUFFIX_RE = re.compile(r" view(?: -?\d+)+ initstate \d+(?: noise \d+)?$")
INITSTATE_RE = re.compile(r"_initstate_(\d+)")


def select_variants(suite_name, suite):
    if LP_IDS:
        ids = [int(x) for x in LP_IDS.split(",") if x.strip()]
    else:
        path = os.path.join(os.path.dirname(libero_pkg.__file__), "benchmark", "task_classification.json")
        classes = json.load(open(path))[suite_name]
        ids = [t["id"] - 1 for t in classes if t["category"] == CATEGORY]
    if LIMIT:
        ids = ids[:LIMIT]
    for tid in ids:
        name = suite.tasks[tid].name
        if ARM != "as_is" and not INITSTATE_RE.search(name):
            sys.exit(f"run_lerobot_variants.py: ARM={ARM} needs robot-initial-state variants; task {tid} is {name}.")
    return ids


def base_name(task_name):
    return task_name.split("_view_")[0]


def eef_pose(problem_env):
    """End-effector position and rotation matrix as the policy sees them (robot0_eef_pos / robot0_eef_quat)."""
    obs = problem_env._get_observations()
    return np.array(obs["robot0_eef_pos"], dtype=np.float64), Rotation.from_quat(obs["robot0_eef_quat"])


def robot_joints(problem_env):
    robot = problem_env.robots[0]
    return problem_env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()


CANONICAL = {}  # (suite, base task name) -> start pose of the no_shift arm
EPISODES_FILE = None
OPEN_EPISODES = {}


def write_episode(record):
    EPISODES_FILE.write(json.dumps(record) + "\n")
    EPISODES_FILE.flush()


def flush_open_episodes():
    for record in list(OPEN_EPISODES.values()):
        write_episode(record)
    OPEN_EPISODES.clear()


class VariantEnv(libero_env.LiberoEnv):
    """One LIBERO-Plus variant played from initial state 0, with the arm's intervention."""

    def __init__(self, *, variant_id, arm, **kwargs):
        super().__init__(task_id=variant_id, **kwargs)
        self.task_suite_name = kwargs["task_suite_name"]
        self.variant_id = variant_id
        self.arm = arm
        self.variant_name = self.task
        self.raw_language = self.task_description
        if STRIP_SUFFIX:
            stripped = SUFFIX_RE.sub("", self.task_description)
            if stripped == self.task_description:
                raise ValueError(f"no LIBERO-Plus suffix found in {self.task_description!r}")
            self.task_description = stripped
        # LIBERO-Plus plays every variant from initial state 0 of the base task.
        self.init_state_id = 0
        self._reset_stride = 0
        match = INITSTATE_RE.search(self.variant_name)
        self.initstate_code = int(match.group(1)) if match else None
        if arm == "no_shift":
            self._task_bddl_file = INITSTATE_RE.sub("_initstate_0", self._task_bddl_file)
        self._debias = None
        self._episode = None

    # -- canonical start pose (no_shift arm of the same base task) -------------------------------------
    def _canonical(self, seed):
        key = (self.task_suite_name, base_name(self.variant_name))
        if key not in CANONICAL:
            probe = libero_env.LiberoEnv(
                task_suite=self._suite, task_id=self.variant_id, task_suite_name=self.task_suite_name,
                camera_name=self.camera_name, init_states=True, obs_type="pixels_agent_pos",
                is_libero_plus=True, observation_height=self.observation_height,
                observation_width=self.observation_width, control_mode=self.control_mode)
            probe._task_bddl_file = INITSTATE_RE.sub("_initstate_0", probe._task_bddl_file)
            probe.init_state_id, probe._reset_stride = 0, 0
            logged._original_reset(probe, seed=seed)  # unlogged: not an episode
            pos, rot = eef_pose(probe._env.env)
            CANONICAL[key] = {"pos": pos, "rot": rot, "joints": robot_joints(probe._env.env), "seed": seed}
            probe.close()
        return CANONICAL[key]

    # -- observation hook for the de-biased state ------------------------------------------------------
    def _format_raw_obs(self, raw_obs):
        obs = super()._format_raw_obs(raw_obs)
        if self._debias is not None and "robot_state" in obs:
            eef = obs["robot_state"]["eef"]
            dp, dr = self._debias
            rot = dr.inv() * Rotation.from_quat(eef["quat"])
            eef["pos"] = (np.asarray(eef["pos"]) - dp).astype(np.asarray(eef["pos"]).dtype)
            eef["quat"] = rot.as_quat().astype(np.asarray(eef["quat"]).dtype)
            eef["mat"] = rot.as_matrix()
        return obs

    def _home(self, target):
        """Drive the end effector to the canonical pose with the arm's own delta controller."""
        problem = self._env.env
        controller = problem.robots[0].controller
        out_max = np.asarray(controller.output_max, dtype=np.float64)
        steps, raw = 0, None
        for steps in range(1, HOMING_MAX_STEPS + 1):
            pos, rot = eef_pose(problem)
            dpos = target["pos"] - pos
            drot = (target["rot"] * rot.inv()).as_rotvec()
            if np.linalg.norm(dpos) < HOMING_POS_TOL and np.linalg.norm(drot) < HOMING_ROT_TOL:
                steps -= 1
                break
            action = np.concatenate([dpos / out_max[:3], drot / out_max[3:6], [-1.0]])
            raw, _, _, _ = self._env.step(np.clip(action, -1.0, 1.0))
        for _ in range(HOMING_SETTLE):
            raw, _, _, _ = self._env.step(libero_env.get_libero_dummy_action())
        pos, rot = eef_pose(problem)
        return raw, {
            "homing_steps": steps,
            "homing_pos_err_m": float(np.linalg.norm(target["pos"] - pos)),
            "homing_rot_err_rad": float(np.linalg.norm((target["rot"] * rot.inv()).as_rotvec())),
            "homing_joint_dist_rad": float(np.linalg.norm(target["joints"] - robot_joints(problem))),
        }

    def reset(self, seed=None, **kwargs):
        explicit = bool((kwargs.get("options") or {}).get(NEW_ROLLOUT_OPTION))
        if self._episode is not None:  # the previous episode of this sub-env is over
            write_episode(OPEN_EPISODES.pop(id(self), self._episode))
            self._episode = None
        self._debias = None
        observation, info = super().reset(seed=seed, **kwargs)
        if not explicit:
            return observation, info
        problem = self._env.env
        pos0, rot0 = eef_pose(problem)
        record = {
            "variant_id": self.variant_id, "variant": self.variant_name, "base_task": base_name(self.variant_name),
            "initstate_code": self.initstate_code, "arm": self.arm, "sub_env": self.episode_index, "seed": seed,
            "instruction": self.task_description, "instruction_raw": self.raw_language,
            "start_eef_pos": pos0.round(5).tolist(), "start_joints": robot_joints(problem).round(5).tolist(),
            "success": False, "success_step": None, "steps": 0,
        }
        canon = self._canonical(seed)
        record["offset_pos_m"] = float(np.linalg.norm(pos0 - canon["pos"]))
        record["offset_rot_rad"] = float(np.linalg.norm((rot0 * canon["rot"].inv()).as_rotvec()))
        record["offset_joint_rad"] = float(np.linalg.norm(robot_joints(problem) - canon["joints"]))
        if self.arm == "debias_state":
            self._debias = (pos0 - canon["pos"], rot0 * canon["rot"].inv())
            observation = self._format_raw_obs(problem._get_observations())
        elif self.arm == "homing":
            raw, stats = self._home(canon)
            record.update(stats)
            observation = self._format_raw_obs(raw if raw is not None else problem._get_observations())
        self._episode = record
        OPEN_EPISODES[id(self)] = record
        return observation, info

    def close(self):
        if self._episode is not None:  # lerobot-eval closes each group when it is done
            write_episode(OPEN_EPISODES.pop(id(self), self._episode))
            self._episode = None
        super().close()

    def step(self, action):
        out = super().step(action)
        if self._episode is not None:
            self._episode["steps"] += 1
            if out[4].get("is_success") and not self._episode["success"]:
                self._episode["success"] = True
                self._episode["success_step"] = self._episode["steps"]
        return out


def create_variant_envs(task, n_envs, gym_kwargs=None, camera_name="agentview_image,robot0_eye_in_hand_image",
                        init_states=True, env_cls=None, control_mode="relative", episode_length=None,
                        camera_name_mapping=None, is_libero_plus=False):
    if not is_libero_plus:
        sys.exit("run_lerobot_variants.py: the env config is not LIBERO-Plus.")
    if env_cls is not gym.vector.SyncVectorEnv:
        sys.exit("run_lerobot_variants.py: needs synchronous vector envs (--eval.use_async_envs=false).")
    gym_kwargs = dict(gym_kwargs or {})
    gym_kwargs.pop("task_ids", None)
    camera_names = libero_env.parse_camera_names(camera_name)
    out = {}
    for suite_name in [s.strip() for s in str(task).split(",") if s.strip()]:
        suite = libero_env._get_suite(suite_name)
        ids = select_variants(suite_name, suite)
        if not ids or len(ids) % n_envs:
            sys.exit(f"run_lerobot_variants.py: {len(ids)} variants selected in {suite_name}, not a multiple of "
                     f"batch size {n_envs}; set LP_LIMIT or the batch size.")
        groups = {}
        for g in range(len(ids) // n_envs):
            fns = []
            for k, tid in enumerate(ids[g * n_envs:(g + 1) * n_envs]):
                def make(tid=tid, k=k):
                    env = VariantEnv(
                        variant_id=tid, arm=ARM, task_suite=suite, task_suite_name=suite_name,
                        camera_name=camera_names, init_states=init_states, episode_length=episode_length,
                        episode_index=k, n_envs=n_envs, control_mode=control_mode,
                        camera_name_mapping=camera_name_mapping, is_libero_plus=True, **gym_kwargs)
                    env._suite = suite
                    return FreezeAfterEpisodeEnd(env)
                fns.append(make)
            groups[g] = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            print(f"Built variant group {g} | suite={suite_name} | variants {ids[g * n_envs]}..{ids[(g + 1) * n_envs - 1]}"
                  f" | arm={ARM}", flush=True)
        out[suite_name] = groups
        GROUPS[suite_name] = {g: ids[g * n_envs:(g + 1) * n_envs] for g in groups}
    return out


GROUPS = {}
libero_env.create_libero_envs = create_variant_envs

# Per-episode logging of the existing runner (reset log, noise log, run meta); it patches LiberoEnv.reset,
# which VariantEnv.reset calls, so every variant episode is logged the same way as the clean cells.
import run_lerobot_libero_logged as logged  # noqa: E402

EPISODES_LOG = os.path.abspath(os.environ.get("EPISODES_LOG", logged._base + "_episodes.jsonl"))
if os.path.exists(EPISODES_LOG):
    sys.exit(f"run_lerobot_variants.py: {EPISODES_LOG} already exists; use new paths for every run.")
EPISODES_FILE = open(EPISODES_LOG, "a")
logged.META.update({
    "variants": {"arm": ARM, "category": None if LP_IDS else CATEGORY, "ids": LP_IDS, "limit": LIMIT,
                 "strip_suffix": STRIP_SUFFIX, "libero_plus_root": LP_ROOT,
                 "libero_plus_commit": logged.el.git_commit(LP_ROOT),
                 "homing": {"max_steps": HOMING_MAX_STEPS, "pos_tol": HOMING_POS_TOL, "rot_tol": HOMING_ROT_TOL,
                            "settle": HOMING_SETTLE} if ARM == "homing" else None,
                 "episodes_log": EPISODES_LOG},
})
logged.write_meta()


def _finish():
    flush_open_episodes()
    logged.META["variants"]["groups"] = GROUPS
    logged.META["variants"]["canonical"] = {
        f"{s}/{b}": {"pos": v["pos"].round(5).tolist(), "quat": v["rot"].as_quat().round(5).tolist(),
                     "joints": v["joints"].round(5).tolist(), "seed": v["seed"]}
        for (s, b), v in CANONICAL.items()}


atexit.register(_finish)  # registered after the runner's write_meta, so it runs first

if __name__ == "__main__":
    logged.main()
