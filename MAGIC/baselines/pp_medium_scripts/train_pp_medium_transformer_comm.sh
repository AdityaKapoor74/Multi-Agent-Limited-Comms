#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=1
python -u run_baselines.py \
  --env_name predator_prey \
  --transformer_comm \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.0001 \
  --recurrent \
  --save \
  --seed 8 \
  --use_wandb \
  --experiment_name TransformerComm_PP_medium_bench_8 \
  | tee train_pp_medium.log &

export CUDA_VISIBLE_DEVICES=2
python -u run_baselines.py \
  --env_name predator_prey \
  --transformer_comm \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.0001 \
  --recurrent \
  --save \
  --seed 12 \
  --use_wandb \
  --experiment_name TransformerComm_PP_medium_bench_12 \
  | tee train_pp_medium.log &

export CUDA_VISIBLE_DEVICES=3
python -u run_baselines.py \
  --env_name predator_prey \
  --transformer_comm \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.0001 \
  --recurrent \
  --save \
  --seed 18 \
  --use_wandb \
  --experiment_name TransformerComm_PP_medium_bench_18 \
  | tee train_pp_medium.log &

export CUDA_VISIBLE_DEVICES=0
python -u run_baselines.py \
  --env_name predator_prey \
  --transformer_comm \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.0001 \
  --recurrent \
  --save \
  --seed 35 \
  --use_wandb \
  --experiment_name TransformerComm_PP_medium_bench_35 \
  | tee train_pp_medium.log &

export CUDA_VISIBLE_DEVICES=1
python -u run_baselines.py \
  --env_name predator_prey \
  --transformer_comm \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.0001 \
  --recurrent \
  --save \
  --seed 41 \
  --use_wandb \
  --experiment_name TransformerComm_PP_medium_bench_41 \
  | tee train_pp_medium.log &
