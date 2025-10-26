#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=1
python3 -u run_baselines.py \
  --env_name predator_prey \
  --ic3net \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent \
  --save \
  --use_comms_channel \
  --comms_penalty 0.001 \
  --seed 8 \
  --use_wandb \
  --experiment_name IC3Net_COMMS_PP_hard_bench_1e-3_8 \
  | tee train_pp_hard.log &

export CUDA_VISIBLE_DEVICES=2
python3 -u run_baselines.py \
  --env_name predator_prey \
  --ic3net \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent \
  --save \
  --use_comms_channel \
  --comms_penalty 0.001 \
  --seed 12 \
  --use_wandb \
  --experiment_name IC3Net_COMMS_PP_hard_bench_1e-3_12 \
  | tee train_pp_hard.log &

export CUDA_VISIBLE_DEVICES=3
python3 -u run_baselines.py \
  --env_name predator_prey \
  --ic3net \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent \
  --save \
  --use_comms_channel \
  --comms_penalty 0.001 \
  --seed 18 \
  --use_wandb \
  --experiment_name IC3Net_COMMS_PP_hard_bench_1e-3_18 \
  | tee train_pp_hard.log &

export CUDA_VISIBLE_DEVICES=0
python3 -u run_baselines.py \
  --env_name predator_prey \
  --ic3net \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent \
  --save \
  --use_comms_channel \
  --comms_penalty 0.001 \
  --seed 35 \
  --use_wandb \
  --experiment_name IC3Net_COMMS_PP_hard_bench_1e-3_35 \
  | tee train_pp_hard.log &

export CUDA_VISIBLE_DEVICES=1
python3 -u run_baselines.py \
  --env_name predator_prey \
  --ic3net \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent \
  --save \
  --use_comms_channel \
  --comms_penalty 0.001 \
  --seed 41 \
  --use_wandb \
  --experiment_name IC3Net_COMMS_PP_hard_bench_1e-3_41 \
  | tee train_pp_hard.log &
