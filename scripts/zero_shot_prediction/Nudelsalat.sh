export CUDA_VISIBLE_DEVICES=3

for seq_length in 96 192 336 512
do
  for prediction_length in 12 24 48 96 192
  do
    python src/zero_shot_prediction.py \
      --dataset_name_or_path "data/eval/Nudelsalat" \
      --seq_length $seq_length \
      --prediction_length $prediction_length \
      --batch_size 4
  done
done