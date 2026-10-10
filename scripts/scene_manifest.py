"""List the scenes lerobot-eval would create, without a policy and without a GPU.

For each task and initial state i this builds a fresh LeRobot LiberoEnv for sub-env i and resets it with
seed = start_seed + i, exactly as lerobot-eval does when one batch holds all episodes of a task
(--eval.batch_size equal to --eval.n_episodes, the paired protocol), and records the fixture poses and the
first-frame hashes. It uses lerobot-eval's own env factories with the default LIBERO env settings (360x360
cameras, 20 Hz, hard reset). A fresh env per scene matters: on some tasks (e.g. LIBERO-Spatial task 4) the
fixture placement of a seeded reset also depends on how many resets the env object did before, and in
lerobot-eval every counted episode is the first seeded reset after exactly one unseeded warm-up reset.

Use it to check a run against the scenes it should have seen (compare "fixtures" exactly), or to know how far
each fixture moves between two seeds before spending GPU time. Fixture poses come from NumPy's seeded
generator and match across machines with the same package versions (generate the manifest in the venv of the
run); frame hashes depend on the renderer and are only comparable between runs on the same kind of machine.

Usage:
    python scene_manifest.py --suite libero_spatial --start_seed 1000 --states 50 --out spatial_1000.jsonl
    python scene_manifest.py --compare a.jsonl b.jsonl      # per-fixture shift in mm between two manifests
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_logging as el  # noqa: E402


def scenes_for_task(suite_name, task_id, start_seed, states, assets_dir):
    if assets_dir:
        import libero.libero as libero_pkg
        libero_pkg._assets_path_cache = os.path.abspath(assets_dir)
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import _get_suite, _make_env_fns
    from lerobot.envs.utils import parse_camera_names

    # Same factories lerobot-eval uses (default LIBERO env settings: 360x360 cameras, 20 Hz, hard reset).
    cfg = LiberoEnvConfig(task=suite_name)
    gym_kwargs = {k: v for k, v in cfg.gym_kwargs.items() if k != "task_ids"}
    suite = _get_suite(suite_name)
    fns = _make_env_fns(suite=suite, suite_name=suite_name, task_id=task_id, n_envs=states,
                        camera_names=parse_camera_names(cfg.camera_name), episode_length=cfg.episode_length,
                        init_states=cfg.init_states, gym_kwargs=gym_kwargs, control_mode=cfg.control_mode,
                        camera_name_mapping=cfg.camera_name_mapping, is_libero_plus=cfg.is_libero_plus)
    records, t0 = [], time.time()
    for i in range(states):
        env = fns[i]()  # sub-env i of the task's batch
        idx = env.init_state_id % len(env._init_states)
        obs, _ = env.reset(seed=start_seed + i)
        records.append({
            "suite": suite_name, "task_id": task_id, "sub_env": i, "init_state_idx": idx,
            "init_state_sha1": el.state_hash(env._init_states[idx]), "seed": start_seed + i,
            "fixtures": el.fixture_poses(env._env.env), "frame_sha1": el.frame_hashes(obs["pixels"]),
        })
        env.close()
    return task_id, records, time.time() - t0


def build(args):
    assert not os.path.exists(args.out), f"{args.out} already exists"
    if args.tasks:
        tasks = [int(t) for t in args.tasks.split(",")]
    else:
        from lerobot.envs.libero import _get_suite
        tasks = list(range(len(_get_suite(args.suite).tasks)))
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(scenes_for_task, args.suite, t, args.start_seed, args.states, args.assets_dir)
                   for t in tasks]
        for fut in cf.as_completed(futures):
            task_id, records, seconds = fut.result()
            results[task_id] = records
            print(f"task {task_id}: {len(records)} scenes in {seconds:.0f} s", flush=True)
    with open(args.out, "w") as out:
        for task_id in tasks:
            for record in results[task_id]:
                out.write(json.dumps(record) + "\n")


def compare(path_a, path_b):
    def load(path):
        with open(path) as f:
            return {(r["task_id"], r["init_state_idx"]): r for r in map(json.loads, f)}

    a, b = load(path_a), load(path_b)
    shifts = {}
    for key in sorted(set(a) & set(b)):
        for name, pose in a[key]["fixtures"].items():
            other = b[key]["fixtures"][name]
            d = 1000 * float(np.linalg.norm(np.subtract(pose["pos"][:2], other["pos"][:2])))
            shifts.setdefault((key[0], name), []).append(d)
    print(f"{len(set(a) & set(b))} common (task, state) pairs")
    for (task_id, name), values in sorted(shifts.items()):
        values = np.array(values)
        print(f"task {task_id} {name}: xy shift median {np.median(values):.1f} mm, max {values.max():.1f} mm, "
              f"identical in {(values == 0).sum()}/{len(values)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite")
    p.add_argument("--start_seed", type=int, default=1000)
    p.add_argument("--states", type=int, default=50, help="episodes per task (= batch size of the run)")
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument("--assets_dir", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        build(args)
