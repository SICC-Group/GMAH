#!/bin/sh
env="StarCraft2"
map="2s3z"
exp="debug"
name="shiyuchen"
seed=33
total_step=1000000

CUDA_VISIBLE_DEVICES=0

python trash-grid/hmpe/train_ppo.py --env_name ${env} --exp_name ${exp} --alg_name "IPPO" \
--seed ${seed} --num_episodes_step ${total_step} --map_name ${map}

python trash-grid/hmpe/train_ppo.py --env_name ${env} --exp_name ${exp} --alg_name "MAPPO" \
--seed ${seed} --num_episodes_step ${total_step} --map_name ${map} --use_global_all_local_state  

python trash-grid/hmpe/train_low.py --env_name ${env} --exp_name ${exp} \
--seed ${seed} --num_episodes_step 400000 --map_name ${map}

python trash-grid/hmpe/train_high.py --env_name ${env} --exp_name ${exp} \
--seed ${seed} --num_episodes_step ${total_step} --map_name ${map} --fix_interval 5

python offpolicy/scripts/train/train_smac.py --env_name ${env} --algorithm_name "mqmix" \
--experiment_name ${exp} --seed ${seed} --map_name ${map} --buffer_size 200000 --lr 5e-4 \
--batch_size 1920 --use_soft_update --hard_update_interval 20000 --num_env_steps ${total_step} \
--log_interval 6000 --eval_interval 10000 --user_name ${name} --use_global_all_local_state --gain 1 \
--use_wandb

python offpolicy/scripts/train/train_smac.py --env_name ${env} --algorithm_name "mqtran" \
--experiment_name ${exp} --seed ${seed} --map_name ${map} --buffer_size 100000 --lr 5e-4 \
--batch_size 480 --use_soft_update --hard_update_interval 10000 --num_env_steps ${total_step} \
--log_interval 3000 --eval_interval 10000 --user_name ${name} --use_global_all_local_state --gain 1 \
--use_wandb