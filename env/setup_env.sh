#!/usr/bin/env bash
# ==== Environment setup: run once per new pod, then Restart Kernel ====
# Installs exact versions from the lock file captured on 27/9 (G1 passed: LIBERO-Spatial 80.5% on mujoco 3.3.7).
set -e
export DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore TF_CPP_MIN_LOG_LEVEL=3
LOCK=/workspace/env/requirements-lock.txt
apt-get update -qq > /dev/null 2>&1 || { echo "ERROR: apt-get update"; exit 1; }
apt-get install -y -qq libegl1 libgl1 libglew-dev libosmesa6-dev > /dev/null 2>&1 || { echo "ERROR: apt-get install"; exit 1; }
[ -d /workspace/LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git /workspace/LIBERO
grep -v " @ " "$LOCK" > /tmp/lock-pypi.txt   # drop entries that point to local files inside the image
echo "[info] lock file: $(wc -l < "$LOCK") packages, $(wc -l < /tmp/lock-pypi.txt) installable from PyPI"
pip install -qq --no-warn-conflicts --no-deps -r /tmp/lock-pypi.txt
pip install -qq --no-warn-conflicts --no-deps -e /workspace/LIBERO
echo "[ok] packages from lock file"

python - <<'EOF'
import importlib.util, os, shutil, yaml, mujoco
assert mujoco.__version__ == "3.3.7", f"mujoco {mujoco.__version__}: >= 3.4.0 breaks LIBERO-Spatial task 5 init states"
root = os.path.dirname(importlib.util.find_spec("robosuite").origin)

# mujoco 3.3.7 keeps the original mj_fullM signature, so robosuite's own call must stay untouched
bc = open(os.path.join(root, "controllers/base_controller.py"), newline="").read()
assert "mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)" in bc, "mj_fullM call was modified - STOP and report"

# two harmless robosuite 1.4.1 patches kept from week 7
patches = {
    "controllers/base_controller.py": [
        ('mass_matrix = np.ndarray(shape=(self.sim.model.nv, self.sim.model.nv), dtype=np.float64, order="C")',
         'mass_matrix = np.zeros((self.sim.model.nv, self.sim.model.nv), dtype=np.float64)', 1),
    ],
    "utils/binding_utils.py": [
        ('joint_type = self.jnt_type[joint_id]', 'joint_type = int(self.jnt_type[joint_id])', 2),
    ],
}
for rel, reps in patches.items():
    path = os.path.join(root, rel)
    src = open(path, newline="").read()
    for old, new, n in reps:
        if src.count(new) == n:
            continue
        assert src.count(old) == n, f"{rel}: found {src.count(old)} matches, expected {n} - STOP and report"
        src = src.replace(old, new)
    open(path, "w", newline="").write(src)
print("[ok] robosuite patched")

# private macro file: the official way to silence "No private macro file found"
mp = os.path.join(root, "macros_private.py")
if not os.path.exists(mp):
    shutil.copyfile(os.path.join(root, "macros.py"), mp)
print("[ok] macros_private")

# LIBERO config on the volume (no Y/N prompt on import)
os.makedirs("/workspace/.libero", exist_ok=True)
r = "/workspace/LIBERO/libero/libero"
yaml.dump({"benchmark_root": r, "bddl_files": f"{r}/bddl_files", "init_states": f"{r}/init_files",
           "datasets": f"{r}/../datasets", "assets": f"{r}/assets"},
          open("/workspace/.libero/config.yaml", "w"))
print("[ok] LIBERO config")
EOF

pip check
python -c "import numpy, scipy, mujoco, cv2, matplotlib, tensorflow as tf; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| mujoco', mujoco.__version__, '| cv2', cv2.__version__, '| mpl', matplotlib.__version__, '| tf', tf.__version__)"
