#!/usr/bin/env bash
cd /workspace/openvla-oft
run () {
  name=$1; shift
  out=/workspace/runs/lp_cam_$name
  mkdir -p "$out"
  python /workspace/libero_plus_eval/run_category.py "$@" \
    --task_suite_name libero_spatial --num_trials_per_task 1 \
    --run_id_note lp_cam_$name --local_log_dir "$out" > "$out/console.log" 2>&1
}
run oft_spatial --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial
run oft_nowrist --pretrained_checkpoint /workspace/checkpoints/oft_w/libero_spatial --num_images_in_input 1 --use_proprio True
run oft_joint   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
echo "[done] 3 camera runs"
