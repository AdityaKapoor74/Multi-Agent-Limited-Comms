#!/bin/bash
export OMP_NUM_THREADS=1

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
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
  --use_comms_channel \
  --comms_penalty 0.0005 \
  --seed 8 \
  --experiment_name IC3Net_COMMS_TJ_hard_bench_5e-4_8 \
  --use_wandb \
  | tee train_tj_hard.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
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
  --use_comms_channel \
  --comms_penalty 0.0005 \
  --seed 12 \
  --experiment_name IC3Net_COMMS_TJ_hard_bench_5e-4_12 \
  --use_wandb \
  | tee train_tj_hard.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
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
  --use_comms_channel \
  --comms_penalty 0.0005 \
  --seed 18 \
  --experiment_name IC3Net_COMMS_TJ_hard_bench_5e-4_18 \
  --use_wandb \
  | tee train_tj_hard.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
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
  --use_comms_channel \
  --comms_penalty 0.0005 \
  --seed 35 \
  --experiment_name IC3Net_COMMS_TJ_hard_bench_5e-4_35 \
  --use_wandb \
  | tee train_tj_hard.log &

python3 -u run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
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
  --use_comms_channel \
  --comms_penalty 0.0005 \
  --seed 41 \
  --experiment_name IC3Net_COMMS_TJ_hard_bench_5e-4_41 \
  --use_wandb \
  | tee train_tj_hard.log &
