# libero-vla-eval
Evaluation harness for Vision-Language-Action (VLA) models on the LIBERO benchmark.
## What it does
- Runs a VLA policy closed-loop in LIBERO and reports the success rate with a Wilson 95% confidence interval.
- Follows the official OpenVLA evaluation protocol: identical image preprocessing (180-degree rotation, JPEG round-trip, Lanczos resize, center crop), gripper post-processing, 10 settle steps and the fixed LIBERO init states.
- Pins the full software environment in `env/requirements-lock.txt`, so every run uses identical package versions.
## Verified baseline
| Model | Suite | Episodes | Success rate | Wilson 95% CI | Paper |
|---|---|---|---|---|---|
| OpenVLA-7B (LIBERO fine-tuned) | LIBERO-Spatial | 200 (10 tasks x inits 0-19) | 80.5% | [74.5, 85.4] | 84.7% |
Environment: RTX PRO 4500 (Blackwell), torch 2.8.0+cu128, transformers 4.40.1, mujoco 3.3.7, robosuite 1.4.1.
## Known pitfalls
- mujoco >= 3.4.0 invalidates the stored init states of LIBERO-Spatial task 5 (the bowl slides off the ramekin before the policy acts). The lock file pins mujoco 3.3.7; on mujoco 3.13 the same model scores 70.0%.
- Image preprocessing must match training exactly: a vertical flip instead of a 180-degree rotation drops the success rate to 0%.
- torch >= 2.6 defaults to `weights_only=True`; LIBERO init-state files must be loaded with `weights_only=False`.
## Usage
Paths currently assume a RunPod network volume mounted at `/workspace`.
```bash
bash env/setup_env.sh            # once per new pod, then restart the Python kernel
python scripts/eval_libero.py --suite libero_spatial --init_start 0 --init_end 20 --out /workspace/runs/spatial
```
