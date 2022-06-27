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
    python train_distributed_ppo.py \
            --env_name ${env} \
            --algorithm_name ${algo} \
            --experiment_name ${exp} \
            --seed ${seed} \
            --n_training_threads 127 \
            --n_rollout_threads 8 \
            --num_mini_batch 1 \
            --episode_length 100 \
            --num_env_steps 10000000 \
            --ppo_epoch 5 \
            --use_eval \
            --use_recurrent_policy \
            --n_eval_rollout_threads 1 \
            --eval_episodes 10 \
            --save_interval 50 \
            --team_spirit 0 \
            --action_type 'both' \
            --use_wandb \
            --warmup_step 100 \
            --load
done