#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=1
python3 -u run_baselines.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.01 \
  --lrate 0.001 \
  --max_steps 80 \
  --recurrent \
  --save \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --use_comms_channel \
  --comms_penalty 0.0001 \
  --num_messages 20 \
  --seed 8 \
  --experiment_name IC3Net_COMMS_GRF_bench_1e-4_delta_20_8 \
  --use_wandb \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=2
python3 -u run_baselines.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.01 \
  --lrate 0.001 \
  --max_steps 80 \
  --recurrent \
  --save \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --use_comms_channel \
  --comms_penalty 0.0001 \
  --num_messages 20 \
  --seed 12 \
  --experiment_name IC3Net_COMMS_GRF_bench_1e-4_delta_20_12 \
  --use_wandb \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=3
python3 -u run_baselines.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.01 \
  --lrate 0.001 \
  --max_steps 80 \
  --recurrent \
  --save \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --use_comms_channel \
  --comms_penalty 0.0001 \
  --num_messages 20 \
  --seed 18 \
  --experiment_name IC3Net_COMMS_GRF_bench_1e-4_delta_20_18 \
  --use_wandb \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=0
python3 -u run_baselines.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.01 \
  --lrate 0.001 \
  --max_steps 80 \
  --recurrent \
  --save \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --use_comms_channel \
  --comms_penalty 0.0001 \
  --num_messages 20 \
  --seed 35 \
  --experiment_name IC3Net_COMMS_GRF_bench_1e-4_delta_20_35 \
  --use_wandb \
  | tee train_grf.log &

export CUDA_VISIBLE_DEVICES=1
python3 -u run_baselines.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 1200 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.01 \
  --lrate 0.001 \
  --max_steps 80 \
  --recurrent \
  --save \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --use_comms_channel \
  --comms_penalty 0.0001 \
  --num_messages 20 \
  --seed 41 \
  --experiment_name IC3Net_COMMS_GRF_bench_1e-4_delta_20_41 \
  --use_wandb \
  | tee train_grf.log &
