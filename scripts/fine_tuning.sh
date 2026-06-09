export CUDA_VISIBLE_DEVICES=0

python src/fine_tuning.py \
    --dataset_name_or_path data/eval/Madrid \
    --finetune_mode "lora" \
    --context_length 768 \
    --learning_rate 1e-6 \
    --num_steps 1024 \
    --batch_size 1024 \
    --split_ratio 0.8 \
    --prediction_length 16
