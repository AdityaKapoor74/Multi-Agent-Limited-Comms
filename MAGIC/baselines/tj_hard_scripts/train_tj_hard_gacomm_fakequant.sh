#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=3
python -u run_baselines.py \
  --env_name traffic_junction \
  --gacomm \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 4000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 8 \
  --experiment_name GACOMM_FakeQuant_TJ_hard_bench_8 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=0
python -u run_baselines.py \
  --env_name traffic_junction \
  --gacomm \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 4000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 12 \
  --experiment_name GACOMM_FakeQuant_TJ_hard_bench_12 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=1
python -u run_baselines.py \
  --env_name traffic_junction \
  --gacomm \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 4000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 18 \
  --experiment_name GACOMM_FakeQuant_TJ_hard_bench_18 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=2
python -u run_baselines.py \
  --env_name traffic_junction \
  --gacomm \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 4000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 35 \
  --experiment_name GACOMM_FakeQuant_TJ_hard_bench_35 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=3
python -u run_baselines.py \
  --env_name traffic_junction \
  --gacomm \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 4000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 41 \
  --experiment_name GACOMM_FakeQuant_TJ_hard_bench_41 \
  --use_wandb \
  --use_fake_quantization \
  --quant_bits 8 \
  | tee train_tj_hard.log &
