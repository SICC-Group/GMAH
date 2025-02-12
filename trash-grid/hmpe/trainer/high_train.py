from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import collections
import random
import pandas as pd
import os
from env.hmpe import hmpe
from utils.utils import Normalization
from utils.utils import compute_advantage
from utils.utils import parsetoJson, get_cent_act_dim, get_dim_from_space, make_onehot

missons = ['FindTrash', 'PickUp_Small', 'PickUp_Big', 'PutTrash', 'PickUpTrash']
prob = [0.25, 0.25, 0.25, 0.25]
missonv2 = ['FindTrash','PickTrash','SplitBig','PutTrash']
probv2 = [0.25, 0.25, 0.25, 0.25]

# No_Action = -1
# Up = 1-1
# Down = 2-1
# Left = 3-1
# Right = 4-1
# Pickup = 5-1
# Putdown = 6-1
# Split = 7

# Forward = 0
# Left = 1
# Right = 2
# Pickup = 3
# Split = 4
# Putdown = 5
Up = 0
Down = 1
Left = 2
Right = 3
Pickup = 4
Putdown = 5
Split = 6

a_str = ['Up', 'Down', 'Left', 'Right', 'Pickup', 'Putdown', 'Split']

def high_train_smac_policy(env, low_policy, high_policy, num_episodes_step, logger=None, num_agents = 3, buffer_size = 32, max_cycles = 128, run_dir = None, fix_interval = 15, test_episodes=10, n_rollout_threads = 1, log_interval = 6000, eval_interval = 20000):

    num_envs = n_rollout_threads
    obs_dim  = get_dim_from_space(env.observation_space[0])
    share_obs_dim = get_dim_from_space(env.share_observation_space[0])
    action_dim = get_dim_from_space(env.action_space[0])
    num_goal = len(env._goal_space)
    ppo_epoch = 10

    total_env_steps = 0
    last_log_t = 0
    last_eval_t = 0
    episode_rewards = []
    win = []
    death_enemy = []

    while total_env_steps < num_episodes_step:
        obs = np.zeros((buffer_size,max_cycles+1,num_agents,obs_dim),dtype=np.float32)
        states = np.zeros((buffer_size,max_cycles+1,num_agents,share_obs_dim),dtype=np.float32)
        avail_acts = np.ones((buffer_size,max_cycles+1,num_agents,num_goal),dtype=np.float32)
        actions = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.int32)
        actions_probs = np.zeros((buffer_size,max_cycles,num_agents,num_goal),dtype=np.float32)
        actions_log_probs = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        rewards = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        dones = np.ones((buffer_size,max_cycles,num_agents,1),dtype=np.bool_)
        subgoals = np.zeros((buffer_size,max_cycles+1,num_agents,num_goal),dtype=np.float32)
        values = np.zeros((buffer_size,max_cycles+1,num_agents,1),dtype=np.float32)
        returns = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)


        i = int(total_env_steps)//max_cycles
        high_policy.lr_decay(int(total_env_steps)//max_cycles, int(num_episodes_step)//max_cycles)

        with tqdm(total=buffer_size, desc='Iteration %d' % i) as pbar:
            for i_episode in range(buffer_size):
                getgoals = np.zeros((num_agents,1))
                for agent in range(num_agents):
                    getgoals[agent] = 0
                    # if i_episode % 10 == 0:
                obss, share_obs, avail_action = env.reset()
                # goal_list = random.choices(list(range(num_goal)), weights=probv2, k=env.num_agents)
                # for idx, g in zip(range(num_agents), goal_list):
                #     env.setgoals(idx, 0)
                # goals = env.goals
                t = 0
                h_i = np.zeros((num_agents),dtype=np.int32)
                high_reward = np.zeros((num_agents,1))
                kl_high_policy = np.zeros((num_agents))
                goals = np.zeros((num_agents, num_goal))
                high_avail_acts = 1 - np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0)
                #np.eye(num_goal)[goals]
                subgoals[i_episode,0] = goals
                while t < max_cycles:
                    high_avail_acts = 1 - np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0)
                    if t == 0 or sign.any():
                        probs, action, log_probs,_ = high_policy.take_action(obss[0], goals, high_avail_acts)
                        value = high_policy.get_values(np.repeat(share_obs,num_agents,axis=0), goals)
                        #value = high_policy.get_values(obss[0], goals)
                        for idx in range(num_agents):
                            if t == 0 or sign[idx]:
                                obs[i_episode,h_i[idx],idx] = obss[0,idx]
                                states[i_episode,h_i[idx],idx] = share_obs[0]
                                avail_acts[i_episode,h_i[idx],idx] = high_avail_acts[idx]
                                actions[i_episode,h_i[idx],idx] = action[idx].unsqueeze(-1).cpu().numpy()
                                actions_log_probs[i_episode,h_i[idx],idx] = log_probs[idx].unsqueeze(-1).cpu().numpy()
                                values[i_episode,h_i[idx]] = value[idx].unsqueeze(-1).cpu().numpy()
                                goals[idx] = np.eye(num_goal)[action[idx].cpu().numpy()]

                                env.goals[idx] = int(action[idx].cpu().numpy())
                                subgoals[i_episode,h_i[idx]+1,idx] = goals[idx]
                                actions_probs[i_episode,h_i[idx],idx] = probs[idx].cpu().numpy()
                    #_, low_action,_,_ = low_policy.take_action(obss[0], goals, avail_action[0])
                    _, low_action = low_policy.eval_action(obss[0], goals, avail_action[0])
                    low_acts = low_action.cpu().numpy()
                    env_acts = np.split(make_onehot(low_acts, action_dim), num_envs)
                    next_obss, next_share_obss, reward, done, infos, next_avail_action = env.step(env_acts)
                    #print(low_acts)
                    # print(t)
                    # print("goals",goals)
                    # print("reward",reward)
                    high_reward += reward[0]
                    
                    #subgoals[i_episode,t] = goals[:,np.newaxis]
                    getgoals[:] = env.get_goals()
                    total_env_steps += 1
                    t += 1
                    
                    pro_probs,_,_,_ = high_policy.take_action(obss[0], goals, high_avail_acts)
                    obss = next_obss
                    share_obs = next_share_obss
                    avail_action = next_avail_action

                    probs,_,_,_ = high_policy.take_action(obss[0], goals, 1-np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0))
                    for idx in range(num_agents):
                        kl_high_policy[idx] = F.kl_div(pro_probs[idx].log().cpu(),probs[idx].cpu(),reduction='sum').numpy()
                    sign = (kl_high_policy > 10) | (getgoals[:,0]==1) | (t % fix_interval == 0)
                    #(kl_high_policy > 10) | getgoal[:,0] | (t % fix_interval == 0)
                    for idx in range(num_agents):
                        if sign[idx]:
                            rewards[i_episode,h_i[idx],idx] = high_reward[idx]
                            dones[i_episode,h_i[idx],idx] = done[0,idx]
                            high_reward[idx] = 0
                            h_i[idx] += 1
                    if done.all():
                        break

                high_avail_acts = 1 - np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0)
                _,action,_,_ = high_policy.take_action(obss[0], goals, high_avail_acts)
                value = high_policy.get_values(np.repeat(share_obs,num_agents,axis=0), goals)
                #value = high_policy.get_values(obss[0], goals)
                for idx in range(num_agents):
                    obs[i_episode,h_i[idx],idx] = obss[0,idx]
                    states[i_episode,h_i[idx],idx] = share_obs[0]
                    avail_acts[i_episode,h_i[idx],idx] = high_avail_acts[idx]
                    values[i_episode,h_i[idx]] = value[idx].unsqueeze(-1).cpu().numpy()

                episode_rewards.append(rewards[i_episode].sum())
                if "won" in infos[0][0].keys():
                    if infos[0][0]['won']:
                        win.append(1)
                    else:
                        win.append(0)

                death_enemy.append(env.get_enemies_death().sum())

                returns[i_episode] = compute_advantage(high_policy.gamma, high_policy.lmbda,values[i_episode],rewards[i_episode],dones[i_episode],high_policy.value_normalizer)
                if (i_episode+1) % 5 == 0:
                    pbar.set_postfix({'episode': '%d' % (buffer_size * i + i_episode+1), 'return': '%.3f' % (np.sum(rewards[:i_episode+1])/(i_episode+1))})
                pbar.update(1)

                if (total_env_steps - last_eval_t)/eval_interval >= 1:
                    high_eval_smac_policy(env, low_policy, high_policy, total_env_steps , num_agents, max_cycles, run_dir, fix_interval, test_episodes, n_rollout_threads)
                    last_eval_t = total_env_steps

            for _ in range(ppo_epoch):
                actor_loss, critic_loss = high_policy.update(obs, states, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts)
            
            if logger:
                logger.add_scalar('actor-loss', actor_loss, i * 50 + i_episode)
                # logger.add_scalar('reward', returns[agent], i * 50 + i_episode)
                logger.add_scalar('critic-loss', critic_loss, i * 50 + i_episode)
                logger.add_scalar(f'rew', np.sum(returns[i_episode]), (i * 50 + i_episode))
            
        if (total_env_steps - last_log_t)/log_interval >= 1:
            data_env = []
            data_env.append(total_env_steps)
            data_env.append(np.mean(episode_rewards))
            data_env.append(np.mean(win))
            data_env.append(np.mean(death_enemy))
            progress_filename = os.path.join(run_dir,'progress.csv')
            df = pd.DataFrame([data_env])
            df.to_csv(progress_filename,mode='a',header=False,index=False) 
            episode_rewards = []
            win = []
            death_enemy = []
            last_log_t = total_env_steps

    return None

def high_eval_smac_policy(env, low_policy, high_policy, i_x, num_agents = 3, max_cycles = 128, run_dir = None, fix_interval = 15, test_episodes=10, n_rollout_threads = 1):

    num_envs = n_rollout_threads
    obs_dim  = get_dim_from_space(env.observation_space[0])
    share_obs_dim = get_dim_from_space(env.share_observation_space[0])
    action_dim = get_dim_from_space(env.action_space[0])
    num_goal = len(env._goal_space)
    ppo_epoch = 10

    actions_probs = np.zeros((test_episodes,max_cycles,num_agents,num_goal),dtype=np.float32)
    actions_log_probs = np.zeros((test_episodes,max_cycles,num_agents,1),dtype=np.float32)
    rewards = np.zeros((test_episodes,max_cycles,num_agents,1),dtype=np.float32)

    win = np.zeros((test_episodes))
    death_enemy = np.zeros((test_episodes))
    
    for i_episode in range(test_episodes):
        getgoals = np.zeros((num_agents,1))
        for agent in range(num_agents):
            getgoals[agent] = 0
            # if i_episode % 10 == 0:
        obss, share_obs, avail_action = env.reset()

        t = 0
        h_i = np.zeros((num_agents),dtype=np.int32)
        high_reward = np.zeros((num_agents,1))
        kl_high_policy = np.zeros((num_agents))
        goals = np.zeros((num_agents, num_goal))
        high_avail_acts = 1 - np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0)
        #np.eye(num_goal)[goals]
        while t < max_cycles:
            high_avail_acts = 1 - np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0)
            if t == 0 or sign.any():
                probs, action = high_policy.eval_action(obss[0], goals, high_avail_acts)
                value = high_policy.get_values(np.repeat(share_obs,num_agents,axis=0), goals)
                #value = high_policy.get_values(obss[0], goals)
                for idx in range(num_agents):
                    if t == 0 or sign[idx]:
                        goals[idx] = np.eye(num_goal)[action[idx].cpu().numpy()]

                        env.goals[idx] = int(action[idx].cpu().numpy())
                        actions_probs[i_episode,h_i[idx],idx] = probs[idx].cpu().numpy()
            #_, low_action,_,_ = low_policy.take_action(obss[0], goals, avail_action[0])

            _, low_action = low_policy.eval_action(obss[0], goals, avail_action[0])

            low_acts = low_action.cpu().numpy()
            env_acts = np.split(make_onehot(low_acts, action_dim), num_envs)
            next_obss, next_share_obss, reward, done, infos, next_avail_action = env.step(env_acts)

            high_reward += reward[0]
            

            getgoals[:] = env.get_goals()
            t += 1
            
            pro_probs,_ = high_policy.eval_action(obss[0], goals, high_avail_acts)
            obss = next_obss
            share_obs = next_share_obss
            avail_action = next_avail_action

            probs,_ = high_policy.eval_action(obss[0], goals, 1-np.repeat(env.get_enemies_death()[np.newaxis,:],num_agents,axis=0))
            for idx in range(num_agents):
                kl_high_policy[idx] = F.kl_div(pro_probs[idx].log().cpu(),probs[idx].cpu(),reduction='sum').numpy()
            sign = (kl_high_policy > 10) | (getgoals[:,0]==1) | (t % fix_interval == 0)
            #(kl_high_policy > 10) | getgoal[:,0] | (t % fix_interval == 0)
            for idx in range(num_agents):
                if sign[idx]:
                    rewards[i_episode,h_i[idx],idx] = high_reward[idx]
                    high_reward[idx] = 0
                    h_i[idx] += 1
            if done.all():
                break
        #env.save_replay()
        if "won" in infos[0][0].keys():
            if infos[0][0]['won']:
                win[i_episode] = 1

        death_enemy[i_episode] = env.get_enemies_death().sum()
            

    data_env = []
    data_env.append(i_x)
    data_env.append(np.sum(rewards)/test_episodes)
    data_env.append(np.sum(win)/test_episodes)
    data_env.append(np.sum(death_enemy)/test_episodes)
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame([data_env])
    df.to_csv(progress_filename,mode='a',header=False,index=False) 

    return None

def high_train_policy(env, low_policy, high_policy, num_episodes_step, logger=None, num_agents = 3, buffer_size = 32, max_cycles = 128, run_dir = None, fix_interval = 15, test_episodes=10, n_rollout_threads = 1, log_interval = 6000, eval_interval = 20000):

    num_envs = n_rollout_threads
    obs_dim  = env.obs_dim
    share_obs_dim = env.state_dim
    action_dim = env.action_space
    num_goal = len(env._goal_space)
    ppo_epoch = 10

    total_env_steps = 0
    last_log_t = 0
    last_eval_t = 0
    episode_rewards = []
    num_put_trash = []

    while total_env_steps < num_episodes_step:
        obs = np.zeros((buffer_size,max_cycles+1,num_agents,obs_dim),dtype=np.float32)
        states = np.zeros((buffer_size,max_cycles+1,num_agents,share_obs_dim),dtype=np.float32)
        avail_acts = np.ones((buffer_size,max_cycles+1,num_agents,num_goal),dtype=np.float32)
        actions = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.int32)
        actions_probs = np.zeros((buffer_size,max_cycles,num_agents,num_goal),dtype=np.float32)
        actions_log_probs = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        rewards = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        dones = np.ones((buffer_size,max_cycles,num_agents,1),dtype=np.bool_)
        subgoals = np.zeros((buffer_size,max_cycles+1,num_agents,num_goal),dtype=np.float32)
        values = np.zeros((buffer_size,max_cycles+1,num_agents,1),dtype=np.float32)
        returns = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)


        i = int(total_env_steps)//max_cycles
        high_policy.lr_decay(int(total_env_steps)//max_cycles, int(num_episodes_step)//max_cycles)

        with tqdm(total=buffer_size, desc='Iteration %d' % i) as pbar:
            for i_episode in range(buffer_size):
                getgoals = np.zeros((num_agents,1))
                for agent in range(num_agents):
                    getgoals[agent] = 0
                    # if i_episode % 10 == 0:
                obss, share_obs, avail_action = env.reset()
                # goal_list = random.choices(list(range(num_goal)), weights=probv2, k=env.num_agents)
                # for idx, g in zip(range(num_agents), goal_list):
                #     env.setgoals(idx, 0)
                # goals = env.goals
                t = 0
                h_i = np.zeros((num_agents),dtype=np.int32)
                high_reward = np.zeros((num_agents,1))
                kl_high_policy = np.zeros((num_agents))
                goals = np.zeros((num_agents, num_goal))
                high_avail_acts = env.get_high_action_mask()
                #np.eye(num_goal)[goals]
                subgoals[i_episode,0] = goals
                while t < max_cycles:
                    high_avail_acts = env.get_high_action_mask()
                    if t == 0 or sign.any():
                        probs, action, log_probs,_ = high_policy.take_action(obss[0], goals, high_avail_acts)
                        value = high_policy.get_values(np.repeat(share_obs,num_agents,axis=0), goals)
                        #value = high_policy.get_values(obss[0], goals)
                        for idx in range(num_agents):
                            if t == 0 or sign[idx]:
                                obs[i_episode,h_i[idx],idx] = obss[0,idx]
                                states[i_episode,h_i[idx],idx] = share_obs[0]
                                avail_acts[i_episode,h_i[idx],idx] = high_avail_acts[idx]
                                actions[i_episode,h_i[idx],idx] = action[idx].unsqueeze(-1).cpu().numpy()
                                actions_log_probs[i_episode,h_i[idx],idx] = log_probs[idx].unsqueeze(-1).cpu().numpy()
                                values[i_episode,h_i[idx]] = value[idx].unsqueeze(-1).cpu().numpy()
                                goals[idx] = np.eye(num_goal)[action[idx].cpu().numpy()]

                                env.goals[idx] = int(action[idx].cpu().numpy())
                                subgoals[i_episode,h_i[idx]+1,idx] = goals[idx]
                                actions_probs[i_episode,h_i[idx],idx] = probs[idx].cpu().numpy()
                    
                    #_, low_action,_,_ = low_policy.take_action(obss[0], goals, avail_action[0])
                    _, low_action = low_policy.eval_action(obss[0], goals, avail_action[0])
                    low_acts = low_action.cpu().numpy()
                    env_acts = np.split(make_onehot(low_acts, action_dim), num_envs)
                    next_obss, next_share_obss, reward, done, infos, next_avail_action = env.step(env_acts[0].argmax(axis=-1))
                    #print(low_acts)
                    # print(t)
                    # print("goals",goals)
                    # print("reward",reward)
                    high_reward += reward[0]
                    
                    #subgoals[i_episode,t] = goals[:,np.newaxis]
                    getgoals[:] = env.get_goals()
                    total_env_steps += 1
                    t += 1
                    
                    pro_probs,_,_,_ = high_policy.take_action(obss[0], goals, high_avail_acts)
                    obss = next_obss
                    share_obs = next_share_obss
                    avail_action = next_avail_action

                    probs,_,_,_ = high_policy.take_action(obss[0], goals, env.get_high_action_mask())
                    for idx in range(num_agents):
                        kl_high_policy[idx] = F.kl_div(pro_probs[idx].log().cpu(),probs[idx].cpu(),reduction='sum').numpy()
                    sign = (kl_high_policy > 10) | (getgoals[:,0]==1) | (t % fix_interval == 0)
                    #(kl_high_policy > 10) | getgoal[:,0] | (t % fix_interval == 0)
                    for idx in range(num_agents):
                        if sign[idx]:
                            rewards[i_episode,h_i[idx],idx] = high_reward[idx]
                            dones[i_episode,h_i[idx],idx] = done[0,idx]
                            high_reward[idx] = 0
                            h_i[idx] += 1
                    if done.all():
                        break
                
                env.close()

                high_avail_acts = env.get_high_action_mask()
                _,action,_,_ = high_policy.take_action(obss[0], goals, high_avail_acts)
                value = high_policy.get_values(np.repeat(share_obs,num_agents,axis=0), goals)
                #value = high_policy.get_values(obss[0], goals)
                for idx in range(num_agents):
                    obs[i_episode,h_i[idx],idx] = obss[0,idx]
                    states[i_episode,h_i[idx],idx] = share_obs[0]
                    avail_acts[i_episode,h_i[idx],idx] = high_avail_acts[idx]
                    values[i_episode,h_i[idx]] = value[idx].unsqueeze(-1).cpu().numpy()

                episode_rewards.append(rewards[i_episode].sum())
                num_put_trash.append(6-env.remain)

                returns[i_episode] = compute_advantage(high_policy.gamma, high_policy.lmbda,values[i_episode],rewards[i_episode],dones[i_episode],high_policy.value_normalizer)
                if (i_episode+1) % 5 == 0:
                    pbar.set_postfix({'episode': '%d' % (buffer_size * i + i_episode+1), 'return': '%.3f' % (np.sum(rewards[:i_episode+1])/(i_episode+1))})
                pbar.update(1)

                if (total_env_steps - last_eval_t)/eval_interval >= 1:
                    high_eval_policy(env, low_policy, high_policy, total_env_steps , num_agents, max_cycles, run_dir, fix_interval, test_episodes, n_rollout_threads)
                    last_eval_t = total_env_steps

            for _ in range(ppo_epoch):
                actor_loss, critic_loss = high_policy.update(obs, states, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts)
            
            if logger:
                logger.add_scalar('actor-loss', actor_loss, i * 50 + i_episode)
                # logger.add_scalar('reward', returns[agent], i * 50 + i_episode)
                logger.add_scalar('critic-loss', critic_loss, i * 50 + i_episode)
                logger.add_scalar(f'rew', np.sum(returns[i_episode]), (i * 50 + i_episode))
            
        if (total_env_steps - last_log_t)/log_interval >= 1:
            data_env = []
            data_env.append(total_env_steps)
            data_env.append(np.mean(episode_rewards))
            data_env.append(np.mean(num_put_trash))
            progress_filename = os.path.join(run_dir,'progress.csv')
            df = pd.DataFrame([data_env])
            df.to_csv(progress_filename,mode='a',header=False,index=False) 
            episode_rewards = []
            num_put_trash = []
            last_log_t = total_env_steps
    

    return None

def high_eval_policy(env, low_policy, high_policy, i_x , num_agents = 3, max_cycles = 128, run_dir = None, fix_interval = 15, test_episodes = 10, n_rollout_threads = 1):


    num_envs = n_rollout_threads
    obs_dim  = env.obs_dim
    share_obs_dim = env.state_dim
    action_dim = env.action_space
    num_goal = len(env._goal_space)
    ppo_epoch = 10

    actions_probs = np.zeros((test_episodes,max_cycles,num_agents,num_goal),dtype=np.float32)
    actions_log_probs = np.zeros((test_episodes,max_cycles,num_agents,1),dtype=np.float32)
    rewards = np.zeros((test_episodes,max_cycles,num_agents,1),dtype=np.float32)

    num_put_trash = np.zeros((test_episodes))
    
    for i_episode in range(test_episodes):
        getgoals = np.zeros((num_agents,1))
        for agent in range(num_agents):
            getgoals[agent] = 0
            # if i_episode % 10 == 0:
        obss, share_obs, avail_action = env.reset()

        t = 0
        h_i = np.zeros((num_agents),dtype=np.int32)
        high_reward = np.zeros((num_agents,1))
        kl_high_policy = np.zeros((num_agents))
        goals = np.zeros((num_agents, num_goal))
        high_avail_acts = env.get_high_action_mask()
        #np.eye(num_goal)[goals]
        while t < max_cycles:
            high_avail_acts = env.get_high_action_mask()
            if t == 0 or sign.any():
                probs, action = high_policy.eval_action(obss[0], goals, high_avail_acts)
                value = high_policy.get_values(np.repeat(share_obs,num_agents,axis=0), goals)
                #value = high_policy.get_values(obss[0], goals)
                for idx in range(num_agents):
                    if t == 0 or sign[idx]:
                        goals[idx] = np.eye(num_goal)[action[idx].cpu().numpy()]

                        env.goals[idx] = int(action[idx].cpu().numpy())
                        actions_probs[i_episode,h_i[idx],idx] = probs[idx].cpu().numpy()
            #_, low_action,_,_ = low_policy.take_action(obss[0], goals, avail_action[0])
            # print(t)
            # print("goals=",goals.argmax(-1))
            _, low_action = low_policy.eval_action(obss[0], goals, avail_action[0])
            # print("action",low_action)
            # print("remain",env.remain)
            low_acts = low_action.cpu().numpy()
            env_acts = np.split(make_onehot(low_acts, action_dim), num_envs)
            next_obss, next_share_obss, reward, done, infos, next_avail_action = env.step(env_acts[0].argmax(axis=-1))

            high_reward += reward[0]
            

            getgoals[:] = env.get_goals()
            t += 1
            
            pro_probs,_ = high_policy.eval_action(obss[0], goals, high_avail_acts)
            obss = next_obss
            share_obs = next_share_obss
            avail_action = next_avail_action

            probs,_ = high_policy.eval_action(obss[0], goals, env.get_high_action_mask())
            for idx in range(num_agents):
                kl_high_policy[idx] = F.kl_div(pro_probs[idx].log().cpu(),probs[idx].cpu(),reduction='sum').numpy()
            sign = (kl_high_policy > 10) | (getgoals[:,0]==1) | (t % fix_interval == 0)
            #(kl_high_policy > 10) | getgoal[:,0] | (t % fix_interval == 0)
            for idx in range(num_agents):
                if sign[idx]:
                    rewards[i_episode,h_i[idx],idx] = high_reward[idx]
                    high_reward[idx] = 0
                    h_i[idx] += 1
            if done.all():
                break
        #import pdb;pdb.set_trace()
        num_put_trash[i_episode] = 6 - env.remain
            

    data_env = []
    data_env.append(i_x)
    data_env.append(np.sum(rewards)/test_episodes)
    data_env.append(np.sum(num_put_trash)/test_episodes)
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame([data_env])
    df.to_csv(progress_filename,mode='a',header=False,index=False) 

    return None