#!/usr/bin/env bash
cd /workspace/openvla-oft
declare -A CKPT=([persuite]=moojink/openvla-7b-oft-finetuned-libero-spatial [joint]=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10)
for cond in zh empty; do
  for regime in persuite joint; do
    out=/workspace/runs/instruction_${regime}_${cond}
    mkdir -p "$out"
    INSTRUCTION_MAP=/workspace/instructions/spatial_${cond}.json python /workspace/instructions/run_with_instructions.py \
      --pretrained_checkpoint "${CKPT[$regime]}" --task_suite_name libero_spatial --num_trials_per_task 20 \
      --run_id_note instruction_${regime}_${cond} --local_log_dir "$out" > "$out/console.log" 2>&1
  done
done
echo "[done] all 4 cells"
