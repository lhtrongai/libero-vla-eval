# Source in every terminal: OFT venv + LIBERO-Plus.
source /workspace/venvs/oft/bin/activate
source /workspace/.prompt.sh
export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=/workspace/LIBERO-plus LIBERO_CONFIG_PATH=/workspace/.libero-plus
cd /workspace/openvla-oft
