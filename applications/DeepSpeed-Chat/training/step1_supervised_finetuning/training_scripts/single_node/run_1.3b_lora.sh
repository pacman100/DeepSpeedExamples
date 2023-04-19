#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

deepspeed main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-1.3b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 3 \
   --lora_r 64 \
   --use_lora \
   --lora_alpha 128 \
   --lora_target_modules "q_proj,v_proj" \
   --lora_dropout 0.05 \
   --lora_bias "none" \
   --gradient_checkpointing \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log
