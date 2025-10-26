#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=1
python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_encoder \
  --message_decoder \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --save \
  --seed 8 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  --experiment_name MAGIC_FakeQuant_GRF_benchmark_8 \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=0
python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_encoder \
  --message_decoder \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --save \
  --seed 12 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  --experiment_name MAGIC_FakeQuant_GRF_benchmark_12 \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=1
python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_encoder \
  --message_decoder \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --save \
  --seed 18 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  --experiment_name MAGIC_FakeQuant_GRF_benchmark_18 \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=0
python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_encoder \
  --message_decoder \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --save \
  --seed 35 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  --experiment_name MAGIC_FakeQuant_GRF_benchmark_35 \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=1
python -u main.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_encoder \
  --message_decoder \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --save \
  --seed 41 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  --experiment_name MAGIC_FakeQuant_GRF_benchmark_41 \
  | tee train_grf.log &
