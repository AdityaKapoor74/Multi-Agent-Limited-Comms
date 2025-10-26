#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=1
python -u main.py \
  --env_name traffic_junction \
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
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 1 \
  --self_loop_type2 1 \
  --first_graph_complete \
  --second_graph_complete \
  --message_decoder \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 8 \
  --use_comms_channel \
  --comms_penalty 0.005 \
  --use_comet \
  --use_wandb \
  --experiment_name MAGIC_comms_TJ_hard_bench_5e-3_8 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=2
python -u main.py \
  --env_name traffic_junction \
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
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 1 \
  --self_loop_type2 1 \
  --first_graph_complete \
  --second_graph_complete \
  --message_decoder \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 12 \
  --use_comms_channel \
  --comms_penalty 0.005 \
  --use_wandb \
  --experiment_name MAGIC_comms_TJ_hard_bench_5e-3_12 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=3
python -u main.py \
  --env_name traffic_junction \
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
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 1 \
  --self_loop_type2 1 \
  --first_graph_complete \
  --second_graph_complete \
  --message_decoder \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 18 \
  --use_comms_channel \
  --comms_penalty 0.005 \
  --use_wandb \
  --experiment_name MAGIC_comms_TJ_hard_bench_5e-3_18 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=0
python -u main.py \
  --env_name traffic_junction \
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
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 1 \
  --self_loop_type2 1 \
  --first_graph_complete \
  --second_graph_complete \
  --message_decoder \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 35 \
  --use_comms_channel \
  --comms_penalty 0.005 \
  --use_wandb \
  --experiment_name MAGIC_comms_TJ_hard_bench_5e-3_35 \
  | tee train_tj_hard.log &

export CUDA_VISIBLE_DEVICES=1
python -u main.py \
  --env_name traffic_junction \
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
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 1 \
  --self_loop_type2 1 \
  --first_graph_complete \
  --second_graph_complete \
  --message_decoder \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 41 \
  --use_comms_channel \
  --comms_penalty 0.005 \
  --use_wandb \
  --experiment_name MAGIC_comms_TJ_hard_bench_5e-3_41 \
  | tee train_tj_hard.log &
