#!/bin/bash
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0
python -u mappo_v2.py \
  --env_name traffic_junction \
  --nagents 10 \
  --dim 14 \
  --max_steps 40 \
  --difficulty medium \
  --vision 1 \
  --recurrent \
  --add_rate_min 0.2 \
  --add_rate_max 0.2 \
  --curr_start 0 \
  --curr_end 0 \
  --vocab_type bool \
  --hid_size 64 \
  --num_heads 1 \
  --attn_dropout 0.1 \
  --ff_dropout 0.1 \
  --proj_dropout 0.1 \
  --policy_lr 1e-4 \
  --policy_weight_decay 0.0 \
  --value_lr 1e-4 \
  --value_weight_decay 0.0 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --policy_clip_param 0.05 \
  --value_clip_param 0.05 \
  --value_coef 0.001 \
  --entropy_coef 0.01 \
  --max_grad_norm 10.0 \
  --ppo_epoch 5 \
  --num_mini_batch 1 \
  --use_comms_channel \
  --comm_loss_coef 1e-4 \
  --max_episodes 30000 \
  --log_interval 10 \
  --eval_interval 100 \
  --save_interval 500 \
  --update_every_n_episodes 10 \
  --sequence_length 10 \
  --device cuda \
  --seed 8 \
  --wandb_name "TransformerComm_Comms_TJ_medium" \
  --save_dir "./models/traffic_junction_medium" \
  | tee train_mappo_tj_medium_batch.log


# #!/bin/bash
# export OMP_NUM_THREADS=1

# export CUDA_VISIBLE_DEVICES=0
# python -u run_baselines.py \
#   --env_name traffic_junction \
#   --transformer_comm \
#   --nagents 10 \
#   --dim 14 \
#   --max_steps 40 \
#   --add_rate_min 0.2 \
#   --add_rate_max 0.2 \
#   --difficulty medium \
#   --vision 1 \
#   --nprocesses 1 \
#   --num_epochs 3000 \
#   --epoch_size 10 \
#   --hid_size 128 \
#   --detach_gap 10 \
#   --lrate 0.0001 \
#   --value_coeff 0.01 \
#   --recurrent \
#   --use_comms_channel \
#   --comms_penalty 0.0001 \
#   --curr_start 0 \
#   --curr_end 0 \
#   --save \
#   --seed 8 \
#   --experiment_name TransformerComm_Comms_TJ_medium_bench_8 \
#   --use_wandb \
#   --ppo_clip_param 0.2 \
#   --ppo_epoch 10 \
#   --num_mini_batch 4 \
#   --use_gae \
#   --gae_lambda 0.95 \
#   --normalize_advantages \
#   --entropy_coefficient 0.01 \
#   --max_grad_norm 0.5 \
#   --gamma 0.99 \
#   --temperature 1.0 \
#   --num_heads 4 \
#   --device cuda \
#   --attn_dropout 0.1 \
#   --ff_dropout 0.1 \
#   --proj_dropout 0.1 \
#   | tee train_tj_medium_ppo.log &

# # export CUDA_VISIBLE_DEVICES=0
# # python -u run_baselines.py \
# #   --env_name traffic_junction \
# #   --transformer_comm \
# #   --nagents 10 \
# #   --dim 14 \
# #   --max_steps 40 \
# #   --add_rate_min 0.2 \
# #   --add_rate_max 0.2 \
# #   --difficulty medium \
# #   --vision 1 \
# #   --nprocesses 1 \
# #   --num_epochs 3000 \
# #   --epoch_size 10 \
# #   --hid_size 128 \
# #   --detach_gap 10 \
# #   --lrate 0.0001 \
# #   --value_coeff 0.01 \
# #   --recurrent \
# #   --use_comms_channel \
# #   --comms_penalty 0.0001 \
# #   --curr_start 0 \
# #   --curr_end 0 \
# #   --save \
# #   --seed 12 \
# #   --experiment_name TransformerComm_Comms_TJ_medium_bench_12 \
# #   --use_wandb \
# #   | tee train_tj_medium.log &

# # export CUDA_VISIBLE_DEVICES=1
# # python -u run_baselines.py \
# #   --env_name traffic_junction \
# #   --transformer_comm \
# #   --nagents 10 \
# #   --dim 14 \
# #   --max_steps 40 \
# #   --add_rate_min 0.2 \
# #   --add_rate_max 0.2 \
# #   --difficulty medium \
# #   --vision 1 \
# #   --nprocesses 1 \
# #   --num_epochs 3000 \
# #   --epoch_size 10 \
# #   --hid_size 128 \
# #   --detach_gap 10 \
# #   --lrate 0.0001 \
# #   --value_coeff 0.01 \
# #   --recurrent \
# #   --use_comms_channel \
# #   --comms_penalty 0.0001 \
# #   --curr_start 0 \
# #   --curr_end 0 \
# #   --save \
# #   --seed 18 \
# #   --experiment_name TransformerComm_Comms_TJ_medium_bench_18 \
# #   --use_wandb \
# #   | tee train_tj_medium.log &

# # export CUDA_VISIBLE_DEVICES=0
# # python -u run_baselines.py \
# #   --env_name traffic_junction \
# #   --transformer_comm \
# #   --nagents 10 \
# #   --dim 14 \
# #   --max_steps 40 \
# #   --add_rate_min 0.2 \
# #   --add_rate_max 0.2 \
# #   --difficulty medium \
# #   --vision 1 \
# #   --nprocesses 1 \
# #   --num_epochs 3000 \
# #   --epoch_size 10 \
# #   --hid_size 128 \
# #   --detach_gap 10 \
# #   --lrate 0.0001 \
# #   --value_coeff 0.01 \
# #   --recurrent \
# #   --use_comms_channel \
# #   --comms_penalty 0.0001 \
# #   --curr_start 0 \
# #   --curr_end 0 \
# #   --save \
# #   --seed 35 \
# #   --experiment_name TransformerComm_Comms_TJ_medium_bench_35 \
# #   --use_wandb \
# #   | tee train_tj_medium.log &

# # export CUDA_VISIBLE_DEVICES=1
# # python -u run_baselines.py \
# #   --env_name traffic_junction \
# #   --transformer_comm \
# #   --nagents 10 \
# #   --dim 14 \
# #   --max_steps 40 \
# #   --add_rate_min 0.2 \
# #   --add_rate_max 0.2 \
# #   --difficulty medium \
# #   --vision 1 \
# #   --nprocesses 1 \
# #   --num_epochs 3000 \
# #   --epoch_size 10 \
# #   --hid_size 128 \
# #   --detach_gap 10 \
# #   --lrate 0.0001 \
# #   --value_coeff 0.01 \
# #   --recurrent \
# #   --use_comms_channel \
# #   --comms_penalty 0.0001 \
# #   --curr_start 0 \
# #   --curr_end 0 \
# #   --save \
# #   --seed 41 \
# #   --experiment_name TransformerComm_Comms_TJ_medium_bench_41 \
# #   --use_wandb \
# #   | tee train_tj_medium.log &