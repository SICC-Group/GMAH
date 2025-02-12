#!/bin/sh
env="hmpe"
exp="debug"
name="shiyuchen"
seed=33
total_step=1000000

CUDA_VISIBLE_DEVICES=0

python trash-grid/hmpe/train_ppo.py --env_name ${env} --exp_name ${exp} --alg_name "IPPO" \
--seed ${seed} --num_episodes_step ${total_step}

python trash-grid/hmpe/train_ppo.py --env_name ${env} --exp_name ${exp} --alg_name "MAPPO" \
--seed ${seed} --num_episodes_step ${total_step} --use_global_all_local_state 

python trash-grid/hmpe/train_low.py --env_name ${env} --exp_name ${exp} \
--seed ${seed} --num_episodes_step 400000

python trash-grid/hmpe/train_high.py --env_name ${env} --exp_name ${exp} \
--seed ${seed} --num_episodes_step ${total_step} --fix_interval 15

python offpolicy/scripts/train/train_hmpe.py --env_name ${env} --algorithm_name "mqmix" \
--experiment_name ${exp} --seed ${seed} --buffer_size 100000 --lr 5e-4 --batch_size 2048 \
--use_soft_update --hard_update_interval 20000 --num_env_steps ${total_step} --log_interval 6000 \
--eval_interval 10000 --user_name ${name} --use_global_all_local_state --gain 1 --use_wandb \
--train_interval 300 --use_reward_normalization

python offpolicy/scripts/train/train_hmpe.py --env_name ${env} --algorithm_name "mqtran" \
--experiment_name ${exp} --seed ${seed} --buffer_size 50000 --lr 5e-4 --batch_size 2048 \
--use_soft_update --hard_update_interval 20000 --num_env_steps ${total_step} --log_interval 6000 \
--eval_interval 10000 --user_name ${name} --use_global_all_local_state --gain 1 --use_wandb \
--train_interval 300 --use_reward_normalization