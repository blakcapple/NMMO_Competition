#!/bin/sh
env="NeuralMMO"
algo="mappo"
exp="base"
seed_max=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ijcai2022-nmmo 
echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 
    python train_nmmo.py \
            --env_name ${env} \
            --algorithm_name ${algo} \
            --experiment_name ${exp} \
            --seed ${seed} \
            --n_training_threads 127 \
            --n_rollout_threads 14 \
            --num_mini_batch 1 \
            --episode_length 50 \
            --num_env_steps 100000000 \
            --ppo_epoch 5 \
            --use_eval \
            --use_recurrent_policy \
            --n_eval_rollout_threads 1 \
            --eval_episodes 10 \
            --save_interval 10 \
            --use_wandb
            --load 
            # --use_value_norm
done