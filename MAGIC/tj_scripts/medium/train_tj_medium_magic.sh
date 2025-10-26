#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name traffic_junction \
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
  --use_wandb \
  --experiment_name MAGIC_TJ_medium_benchmark_8 \
  | tee train_tj_medium.log &

python -u main.py \
  --env_name traffic_junction \
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
  --use_wandb \
  --experiment_name MAGIC_TJ_medium_benchmark_12 \
  | tee train_tj_medium.log &


python -u main.py \
  --env_name traffic_junction \
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
  --use_wandb \
  --experiment_name MAGIC_TJ_medium_benchmark_18 \
  | tee train_tj_medium.log &

python -u main.py \
  --env_name traffic_junction \
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
  --use_wandb \
  --experiment_name MAGIC_TJ_medium_benchmark_35 \
  | tee train_tj_medium.log &

python -u main.py \
  --env_name traffic_junction \
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
  --use_wandb \
  --experiment_name MAGIC_TJ_medium_benchmark_41 \
  | tee train_tj_medium.log &
