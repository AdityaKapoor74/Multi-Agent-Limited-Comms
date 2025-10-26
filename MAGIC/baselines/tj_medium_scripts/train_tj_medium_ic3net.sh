#!/bin/bash
export OMP_NUM_THREADS=1

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --add_rate_min 0.2 \
  --add_rate_max 0.2 \
  --difficulty medium \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 3000 \
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
  --experiment_name IC3Net_TJ_medium_bench_8 \
  --use_wandb \
  | tee train_tj_medium.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --add_rate_min 0.2 \
  --add_rate_max 0.2 \
  --difficulty medium \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 3000 \
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
  --experiment_name IC3Net_TJ_medium_bench_12 \
  --use_wandb \
  | tee train_tj_medium.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --add_rate_min 0.2 \
  --add_rate_max 0.2 \
  --difficulty medium \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 3000 \
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
  --experiment_name IC3Net_TJ_medium_bench_18 \
  --use_wandb \
  | tee train_tj_medium.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --add_rate_min 0.2 \
  --add_rate_max 0.2 \
  --difficulty medium \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 3000 \
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
  --experiment_name IC3Net_TJ_medium_bench_35 \
  --use_wandb \
  | tee train_tj_medium.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --add_rate_min 0.2 \
  --add_rate_max 0.2 \
  --difficulty medium \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 3000 \
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
  --experiment_name IC3Net_TJ_medium_bench_41 \
  --use_wandb \
  | tee train_tj_medium.log &
