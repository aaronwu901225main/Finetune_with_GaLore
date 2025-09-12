#一般 GaLore AdamW（bf16、單卡）
# python finetune_galore.py \
#  --model_name_or_path meta-llama/Llama-2-7b-hf \
#  --dataset_path /data/my_sft/train.jsonl \
#  --output_dir /outputs/llama2-galore-sft \
#  --max_steps 2000 \
#  --per_device_train_batch_size 2 \
#  --gradient_accumulation_steps 8 \
#  --block_size 1024 \
#  --optimizer galore_adamw \
#  --rank 128 \
#  --galore_scale 0.25 \
#  --update_proj_gap 200 \
#  --dtype bfloat16 \
#  --eval_every 200 \
#  --save_every 200

# 8-bit GaLore（更省記憶體）
#python finetune_galore.py \
#  --model_name_or_path mistralai/Mistral-7B-v0.1 \
#  --dataset_path /data/my_sft \
#  --output_dir /outputs/mistral-galore8 \
#  --optimizer galore_adamw8bit \
#  --rank 256 \
#  --update_proj_gap 500 \
#  --galore_scale 0.25 \
#  --dtype bfloat16

# Per-layer 8-bit（單 GPU、建議關掉 AMP，腳本會自動關）
python finetune_galore.py \
  --model_name_or_path meta-llama/Llama-3-8B \
  --dataset_path /data/my_sft \
  --output_dir /outputs/llama3-perlayer \
  --optimizer galore_adamw8bit_per_layer \
  --rank 1024 \
  --update_proj_gap 500 \
  --galore_scale 0.25 \
  --dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16

# 從 checkpoint 繼續
#python finetune_galore.py \
#  --model_name_or_path /outputs/llama2-galore-sft \
#  --dataset_path /data/my_sft \
#  --output_dir /outputs/llama2-galore-sft \
#  --resume_from /outputs/llama2-galore-sft/checkpoint-step_2000
