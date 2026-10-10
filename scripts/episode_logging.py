"""Helpers shared by the per-episode logging wrappers.

Everything here only reads state; nothing changes what the simulator or the policy does.

- state_hash / array_sha1: content hashes, so two runs can be compared episode by episode.
- fixture_poses: placement of the fixed furniture (cabinet, stove, wine rack, ...). LIBERO samples it in
  env.reset() and writes it into model.body_pos / model.body_quat; set_init_state() does not restore it,
  so it depends on the env seed even with fixed initial states.
- texture_state: whether the LIBERO PNG textures carry an sRGB tag. MuJoCo 3.3.3 and later read that tag
  and render tagged textures darker than earlier versions (the LIBERO floor is tagged), so the same
  initial state looks different depending on the MuJoCo version and on the asset files.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
import time

import numpy as np

# The only LIBERO textures with an sRGB chunk (checked on the LIBERO repo and on lerobot/libero-assets).
SRGB_TAGGED_TEXTURES = (
    "textures/light-gray-floor-tile.png",
    "textures/tile_grigia_caldera_porcelain_floor.png",
    "stable_hope_objects/salad_dressing/texture_map.png",
    "stable_hope_objects/new_salad_dressing/texture_map.png",
)


def state_hash(state):
    """SHA-1 of an initial state as float64 (same function as the earlier logs, so hashes stay comparable)."""
    return hashlib.sha1(np.ascontiguousarray(np.asarray(state, dtype=np.float64)).tobytes()).hexdigest()


def array_sha1(a):
    """SHA-1 of an array's raw bytes in its own dtype (images: uint8, noise: float32)."""
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fixture_poses(problem_env):
    """{fixture name: {"pos": [x, y, z], "quat": [w, x, y, z]}} read from the live MuJoCo model.

    `problem_env` is the LIBERO problem env (OffScreenRenderEnv.env). Floats are stored with full precision,
    so poses from two runs can be compared for exact equality. Suites without fixtures give {}.
    """
    fixtures = getattr(problem_env, "fixtures_dict", None) or {}
    model = problem_env.sim.model  # read after reset: a hard reset rebuilds the simulation
    out = {}
    for name in sorted(fixtures):
        body_id = model.body_name2id(fixtures[name].root_body)
        out[name] = {
            "pos": [float(v) for v in model.body_pos[body_id]],
            "quat": [float(v) for v in model.body_quat[body_id]],
        }
    return out


def frame_hashes(images):
    """{camera key: SHA-1} for a dict of uint8 camera images."""
    return {k: array_sha1(v) for k, v in sorted(images.items())}


def png_has_srgb_chunk(path):
    with open(path, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        while True:
            header = f.read(8)
            if len(header) < 8:
                return False
            length, chunk_type = struct.unpack(">I4s", header)
            if chunk_type == b"sRGB":
                return True
            if chunk_type == b"IDAT":  # colour chunks must come before the image data
                return False
            f.seek(length + 4, 1)


def texture_state(assets_dir):
    """sRGB tag and SHA-1 of the textures that MuJoCo >= 3.3.3 renders differently."""
    state = {"assets_dir": os.path.abspath(assets_dir) if assets_dir else None, "files": {}}
    for rel in SRGB_TAGGED_TEXTURES:
        path = os.path.join(assets_dir, rel) if assets_dir else None
        if path and os.path.exists(path):
            state["files"][rel] = {"sha1": file_sha1(path), "srgb_tag": png_has_srgb_chunk(path)}
        else:
            state["files"][rel] = None
    tagged = [v["srgb_tag"] for v in state["files"].values() if v]
    state["srgb_tagged_files"] = sum(bool(t) for t in tagged)
    return state


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001 - a missing optional module must not stop a run
        return None


def versions(extra=()):
    names = ("torch", "mujoco", "robosuite", "numpy", "gymnasium", "transformers") + tuple(extra)
    return {name: module_version(name) for name in names}


def git_commit(path):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_dirty(path):
    try:
        out = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def torch_rng_hashes():
    """SHA-1 of the torch CPU and CUDA generator states (CUDA: None without a GPU)."""
    try:
        import torch
    except ImportError:
        return {"cpu": None, "cuda": None}
    cpu = array_sha1(torch.get_rng_state().numpy())
    cuda = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda = hashlib.sha1(b"".join(s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all())).hexdigest()
    return {"cpu": cpu, "cuda": cuda}


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def host_info():
    return {
        "host": platform.node(),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "gpu": gpu_name(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
