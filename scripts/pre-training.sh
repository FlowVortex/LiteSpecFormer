export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 \
  src/pre_training.py \
    --team_name "FlowVortex" \
    --project_name "LiteSpecFormer-Training" \
    --experiment_name "pre-training" \
    --scenario_name "pre-training" \
    --seed 42
    