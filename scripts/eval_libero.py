"""Evaluate OpenVLA-7B on one LIBERO suite, following openvla/experiments/robot/libero/run_libero_eval.py.
Writes one JSON line per episode to episodes.jsonl; re-running the same command skips tasks already completed.
Verified 2026-09-27: libero_spatial, inits 0-19, MuJoCo 3.3.7 -> 161/200 = 80.5% (paper: 84.7%)."""
import os, sys, time, json, random, argparse, traceback, warnings
os.environ["MUJOCO_GL"] = "egl"
os.environ["LIBERO_CONFIG_PATH"] = "/workspace/.libero"
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/workspace/LIBERO")
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--init_start", type=int, default=0)
ap.add_argument("--init_end", type=int, default=20)      # exclusive
ap.add_argument("--out", default="/workspace/runs/libero_spatial")
args = ap.parse_args()

import numpy as np
import torch
import tensorflow as tf
import imageio
import mujoco
import robosuite
from PIL import Image
import transformers
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
transformers.logging.set_verbosity_error()

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[args.suite]
NUM_STEPS_WAIT = 10
SEED = 7
DEVICE = torch.device("cuda:0")
CKPT = f"openvla/openvla-7b-finetuned-{args.suite.replace('_', '-')}"

# ---------- Copied from openvla/experiments/robot (same code as the verified 10-episode smoke test) ----------
def set_seed_everywhere(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_env(task, resolution=256):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # official script: the seed affects object positions even with a fixed init state
    return env, task.language

def resize_image(img, resize_size):
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()

def get_libero_image(obs, resize_size=(224, 224)):
    img = obs["agentview_image"][::-1, ::-1]  # rotate 180 degrees, NOT a vertical flip
    return resize_image(img, resize_size)

def crop_and_resize(image, crop_scale, batch_size):
    expanded = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0); expanded = True
    h = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    w = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    ho, wo = (1 - h) / 2, (1 - w) / 2
    boxes = tf.stack([ho, wo, ho + h, wo + w], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(batch_size), (224, 224))
    return image[0] if expanded else image

def get_vla_action(img, task_label):
    image = Image.fromarray(img).convert("RGB")
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, 0.9, 1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    image = Image.fromarray(image.numpy()).convert("RGB")
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    return vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

def normalize_gripper_action(action, binarize=True):  # [0,1] -> [-1,1] -> sign
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action

def invert_gripper_action(action):  # dataset convention 0=close -> LIBERO +1=close
    action[..., -1] = action[..., -1] * -1.0
    return action

def run_episode(env, init_state, desc, video_path=None):
    env.reset()
    obs = env.set_init_state(init_state)
    t, done, frames, actions, err = 0, False, [], [], None
    while t < MAX_STEPS + NUM_STEPS_WAIT:
        try:
            if t < NUM_STEPS_WAIT:  # let objects settle before the policy acts
                obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                t += 1
                continue
            img = get_libero_image(obs)
            frames.append(img)
            action = get_vla_action(img, desc)
            action = invert_gripper_action(normalize_gripper_action(action, binarize=True))
            actions.append(action.copy())
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1
        except Exception:
            err = traceback.format_exc(); print(err, flush=True); break
    if video_path:
        imageio.mimwrite(video_path, frames, fps=30)
    return {"success": bool(done), "steps": t, "actions": np.array(actions), "error": err}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h

# ---------- Load model ----------
set_seed_everywhere(SEED)
t0 = time.time()
vla = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
UNNORM_KEY = args.suite
if UNNORM_KEY not in vla.norm_stats and f"{UNNORM_KEY}_no_noops" in vla.norm_stats:
    UNNORM_KEY = f"{UNNORM_KEY}_no_noops"
assert UNNORM_KEY in vla.norm_stats
print(f"Model loaded in {time.time()-t0:.0f}s | {CKPT} | unnorm_key={UNNORM_KEY} | mujoco {mujoco.__version__}", flush=True)

# ---------- Run: one env per task (as in the official script), inits init_start..init_end-1 ----------
os.makedirs(f"{args.out}/videos", exist_ok=True)
res_path = f"{args.out}/episodes.jsonl"
done_eps = {}
if os.path.exists(res_path):
    for line in open(res_path):
        r = json.loads(line); done_eps[(r["task_id"], r["init"])] = r
eps = list(range(args.init_start, args.init_end))

suite = benchmark.get_benchmark_dict()[args.suite]()
t_all = time.time()
for task_id in range(suite.n_tasks):
    if all((task_id, e) in done_eps for e in eps):
        print(f"[task {task_id}] already done, skipping", flush=True); continue
    task = suite.get_task(task_id)
    init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(init_path, weights_only=False)  # official LIBERO file; torch>=2.6 defaults to weights_only=True
    env, desc = make_env(task)
    for e in eps:
        t0 = time.time()
        r = run_episode(env, init_states[e], desc, video_path=f"{args.out}/videos/t{task_id}_i{e}.mp4")
        a = r["actions"]
        rec = {"task_id": task_id, "task": desc, "init": e, "success": r["success"], "steps": r["steps"],
               "close_ratio": float((a[:, 6] > 0).mean()) if len(a) else 0.0,
               "sec": round(time.time() - t0, 1), "error": r["error"]}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done_eps[(task_id, e)] = rec
        keys = [k for k in done_eps if k[1] in eps]
        k_ok = sum(done_eps[k]["success"] for k in keys)
        print(f"[task {task_id} | init {e}] success={rec['success']} steps={rec['steps']} {rec['sec']}s"
              f" | total {k_ok}/{len(keys)} = {100*k_ok/len(keys):.1f}%", flush=True)
    env.close()

# ---------- Summary ----------
rows = [done_eps[(t, e)] for t in range(suite.n_tasks) for e in eps]
per_task = {t: sum(r["success"] for r in rows if r["task_id"] == t) for t in range(suite.n_tasks)}
k, n = sum(per_task.values()), len(rows)
lo, hi = wilson(k, n)
json.dump({"ckpt": CKPT, "suite": args.suite, "inits": [args.init_start, args.init_end], "n": n, "success": k,
           "sr": k / n, "wilson95": [lo, hi], "per_task_success": per_task,
           "env": {"gpu": torch.cuda.get_device_name(0), "mujoco": mujoco.__version__, "torch": torch.__version__,
                   "transformers": transformers.__version__, "tensorflow": tf.__version__,
                   "numpy": np.__version__, "robosuite": robosuite.__version__},
           "hours": round((time.time() - t_all) / 3600, 2)},
          open(f"{args.out}/summary.json", "w"), indent=2)
print("\n=== RESULTS ===", flush=True)
for t in range(suite.n_tasks):
    print(f"task {t}: {per_task[t]}/{len(eps)}", flush=True)
print(f"SR {args.suite}: {100*k/n:.1f}% ({k}/{n}) | Wilson 95% CI [{100*lo:.1f}, {100*hi:.1f}]", flush=True)
