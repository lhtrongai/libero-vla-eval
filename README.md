# libero-vla-eval
Evaluation harness for Vision-Language-Action (VLA) models on the LIBERO benchmark.
## What it does
- Runs a VLA policy closed-loop in LIBERO and reports the success rate with a Wilson 95% confidence interval.
- Follows the official OpenVLA evaluation protocol: identical image preprocessing (180-degree rotation, JPEG round-trip, Lanczos resize, center crop), gripper post-processing, 10 settle steps and the fixed LIBERO init states.
- Pins the full software environment in `env/requirements-lock.txt`, so every run uses identical package versions.
- Logs, per episode, everything that defines it (initial state, seeds, fixture placement, first frame, initial policy noise, texture look), so two runs can be compared episode by episode.
## Verified baseline
| Model | Suite | Episodes | Success rate | Wilson 95% CI | Paper |
|---|---|---|---|---|---|
| OpenVLA-7B (LIBERO fine-tuned) | LIBERO-Spatial | 200 (10 tasks x inits 0-19) | 80.5% | [74.5, 85.4] | 84.7% |
Environment: RTX PRO 4500 (Blackwell), torch 2.8.0+cu128, transformers 4.40.1, mujoco 3.3.7, robosuite 1.4.1.
## Known pitfalls
- mujoco >= 3.4.0 invalidates the stored init states of LIBERO-Spatial task 5 (the bowl slides off the ramekin before the policy acts). The lock file pins mujoco 3.3.7; on mujoco 3.13 the same model scores 70.0%.
- mujoco >= 3.3.3 reads the sRGB chunk of PNG textures, and four LIBERO textures carry one, including the floor of every scene. LIBERO-Object (robot and objects on the floor) then renders about 26% darker than the images LIBERO's training data was rendered with; LIBERO-Spatial and LIBERO-Goal change in under 0.3% of pixels. OpenVLA-7B drops on LIBERO-Object from 88.4% (paper) to about 69% with mujoco 3.3.7 (also reported in openvla issue #282). `env/make_linear_texture_assets.py` removes the chunk and restores the old rendering pixel for pixel while keeping mujoco 3.3.7; every run records which look it used in its metadata (`textures`).
- LIBERO samples the placement of fixtures (wooden cabinet, stove, wine rack) in `env.reset()`, and `set_init_state()` does not restore it, so a fixed init state is rendered with a seed-dependent fixture placement (up to about 2 cm between two seeds in our checks). LIBERO-Object has no fixtures. On some tasks (e.g. LIBERO-Spatial task 4) the placement of a seeded reset also depends on how many resets the env object did before.
- LeRobot's LIBERO env before 19 January 2026 (PR #2817) called `set_init_state()` before `env.reset()`, so the reset overwrote the fixed init state and evaluations ran on freshly sampled scenes instead.
- Image preprocessing must match training exactly: a vertical flip instead of a 180-degree rotation drops the success rate to 0%.
- torch >= 2.6 defaults to `weights_only=True`; LIBERO init-state files must be loaded with `weights_only=False`.
## Usage
Paths currently assume a RunPod network volume mounted at `/workspace`.
```bash
bash env/setup_env.sh            # once per new pod, then restart the Python kernel
python scripts/eval_libero.py --suite libero_spatial --init_start 0 --init_end 20 --out /workspace/runs/spatial
```
### Per-episode logs
OpenVLA, original evaluation script, one record per episode in `$OUT_DIR/episodes.jsonl`:
```bash
OUT_DIR=/workspace/runs/openvla_spatial python scripts/run_openvla_libero_logged.py \
  --model_family openvla --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial --center_crop True --num_trials_per_task 50 --seed 7
# ENV_SEED=1 moves only the fixtures (the original script seeds the env with 0); init states and policy are unchanged.
```
LeRobot (`lerobot-eval`, e.g. SmolVLA), one record per reset, one per policy-noise draw, plus run metadata:
```bash
RESET_LOG=/workspace/runs/smolvla_spatial/resets.jsonl python scripts/run_lerobot_libero_logged.py \
  --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=50 --eval.batch_size=50 --eval.use_async_envs=false \
  --seed=1000 --output_dir=/workspace/runs/smolvla_spatial/eval
# --seed sets both the policy noise and the scenes. SCENE_SEED_SHIFT=-1 with --seed=1001 runs seed-1001 policy
# noise on exactly the scenes of --seed=1000. LIBERO_ASSETS_DIR=<copy made by env/make_linear_texture_assets.py>
# renders with the pre-3.3.3 look.
```
Checks that need no GPU:
```bash
# scenes a lerobot-eval run with one batch per task would see (fixture poses, first-frame hashes)
python scripts/scene_manifest.py --suite libero_spatial --start_seed 1000 --states 50 --workers 6 --out spatial_1000.jsonl
# two runs (or a run and a manifest): same fixtures, same first frames, same initial noise?
python scripts/compare_episode_logs.py /workspace/runs/a/resets.jsonl /workspace/runs/b/resets.jsonl
```
