# libero-vla-eval

Evaluation harness for Vision-Language-Action (VLA) models on the LIBERO benchmark
(OpenVLA, OpenVLA-OFT, pi0).

## Status
- [x] End-to-end pipeline: OpenVLA-7B running closed-loop in LIBERO (RunPod A40, bf16)
- [ ] Reproduce OpenVLA baseline success rates on LIBERO
- [ ] Add OpenVLA-OFT and pi0 behind a common model interface

## Setup
See `requirements.txt`. robosuite 1.4.1 needs patches to run with mujoco 3.x
(see the patch cell in `notebooks/openvla_libero_rollout.ipynb`).
