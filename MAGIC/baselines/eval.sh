#!/bin/bash
export OMP_NUM_THREADS=1

python3 -u eval.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 100 \
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
  --seed 8 \
  --model_path model.pt > t1.out &

  python3 -u eval.py \
  --env_name grf \
  --ic3net \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 100 \
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
  --seed 8 \
  --model_path model2.pt \
  --use_comms_channel > t2.out &