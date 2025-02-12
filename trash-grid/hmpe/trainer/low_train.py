from tqdm import tqdm
import numpy as np
import torch
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

def low_train_smac_policy(env, policy, num_episodes_step, logger=None, num_agents = 3, buffer_size = 32, max_cycles = 128, run_dir = None, test_episodes = 10, n_rollout_threads = 1, use_rnn = True, hidden_size = 64, log_interval = 6000, eval_interval = 20000, env_name = None):

    num_envs = n_rollout_threads
    obs_dim  = get_dim_from_space(env.observation_space[0])
    action_dim = get_dim_from_space(env.action_space[0])
    num_goal = len(env._goal_space)
    ppo_epoch = 10
    #probv2 = [1.0/num_goal] * num_goal

    total_env_steps = 0
    last_log_t = 0
    last_eval_t = 0
    episode_rewards = []
    win = []
    death_enemy = []
    num_put_trash = []

    while total_env_steps < num_episodes_step:
        obs = np.zeros((buffer_size,max_cycles+1,num_agents,obs_dim),dtype=np.float32)
        rnn_states = np.zeros((buffer_size,max_cycles+1,num_agents,hidden_size),dtype=np.float32)
        rnn_states_critic = np.zeros((buffer_size,max_cycles+1,num_agents,hidden_size),dtype=np.float32)
        avail_acts = np.zeros((buffer_size,max_cycles+1,num_agents,action_dim),dtype=np.float32)
        actions = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.int32)
        actions_log_probs = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        rewards = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        dones = np.ones((buffer_size,max_cycles,num_agents,1),dtype=np.bool_)
        subgoals = np.zeros((buffer_size,max_cycles+1,num_agents,num_goal),dtype=np.float32)
        values = np.zeros((buffer_size,max_cycles+1,num_agents,1),dtype=np.float32)
        returns = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)

        i = int(total_env_steps)//max_cycles
        policy.lr_decay(int(total_env_steps)//max_cycles, int(num_episodes_step)//max_cycles)

        with tqdm(total=buffer_size, desc='Iteration %d' % (int(total_env_steps)//max_cycles)) as pbar:
            for i_episode in range(buffer_size):
                getgoals = np.zeros((num_agents,1))
                stop = {}
                for agent in range(num_agents):
                    getgoals[agent] = 0
                    stop[agent] = 0
                    # if i_episode % 10 == 0:
                obss, _, avail_action = env.reset()
                #goal_list = random.choices(list(range(num_goal)), weights=probv2, k=env.num_agents)
                for agent in range(num_agents):
                    env.setgoals(agent)
                goals = np.eye(num_goal)[env.goals]
                rnn_states_batch = np.zeros((num_agents, hidden_size))
                rnn_critic = np.zeros((num_agents, hidden_size))
                done = False
                t = 0
                while t < max_cycles:
                    
                    if use_rnn:
                        _,action,log_probs,_,rnn_states_batch = policy.take_action(obss[0], goals, avail_action[0], rnn_states_batch)
                        value,rnn_critic = policy.get_values(obss[0], goals, rnn_critic)
                    else:
                        _,action,log_probs,_ = policy.take_action(obss[0], goals, avail_action[0])
                        value = policy.get_values(obss[0], goals)
                    acts = action.cpu().numpy()
                    env_acts = np.split(make_onehot(acts, action_dim), num_envs)
                    next_obss, _, reward, done, infos, next_avail_action = env.step(env_acts)
                    #next_obss, reward, getgoal, done, _ = env.step(action)
                    getgoal = env.get_goals()
                    #env.get_goals()
                    #(reward != 0)
                    if use_rnn:
                        rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
                        rnn_critic = rnn_critic if isinstance(rnn_critic, np.ndarray) else rnn_critic.cpu().detach().numpy()
                        rnn_states[i_episode,t] = rnn_states_batch
                        rnn_states_critic[i_episode,t] = rnn_critic
                        
                    obs[i_episode,t] = obss[0]
                    avail_acts[i_episode,t] = avail_action[0]
                    subgoals[i_episode,t] = goals
                    actions[i_episode,t] = acts[:,np.newaxis]
                    actions_log_probs[i_episode,t] = log_probs.unsqueeze(-1).cpu().numpy()
                    rewards[i_episode,t] = reward[0]
                    dones[i_episode,t] = (done[0] | (getgoal==1))
                    values[i_episode,t] = value.unsqueeze(-1).cpu().numpy()
                    getgoals[:] = getgoal

                    for agent in range(num_agents):
                        if getgoals[agent]:
                            env.setgoals(agent)
                            goals = np.eye(num_goal)[env.goals]

                    t += 1
                    total_env_steps += 1
                    obss = next_obss
                    avail_action = next_avail_action 
                    if done.all():
                        break

                if use_rnn:
                    _,action,log_probs,_,rnn_states_batch = policy.take_action(obss[0], goals, avail_action[0], rnn_states_batch)
                    value,rnn_critic = policy.get_values(obss[0], goals, rnn_critic)
                else:
                    _,action,log_probs,_ = policy.take_action(obss[0], goals, avail_action[0])
                    value = policy.get_values(obss[0], goals)
                if use_rnn:
                    rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
                    rnn_critic = rnn_critic if isinstance(rnn_critic, np.ndarray) else rnn_critic.cpu().detach().numpy()    
                    rnn_states[i_episode,t] = rnn_states_batch
                    rnn_states_critic[i_episode,t] = rnn_critic
                
                obs[i_episode,t] = obss[0]
                avail_acts[i_episode,t] = avail_action[0]
                subgoals[i_episode,t] = goals
                values[i_episode,t] = value.unsqueeze(-1).cpu().numpy()
                
                episode_rewards.append(rewards[i_episode].sum())

                if env_name == 'hmpe':
                    num_put_trash.append(6-env.remain)
                else:
                    if "won" in infos[0][0].keys():
                        if infos[0][0]['won']:
                            win.append(1)
                        else:
                            win.append(0)
                    death_enemy.append(env.get_enemies_death().sum())
                
                returns[i_episode] = compute_advantage(policy.gamma, policy.lmbda,values[i_episode],rewards[i_episode],dones[i_episode],policy.value_normalizer)
                if (i_episode+1) % 5 == 0:
                    pbar.set_postfix({'episode': '%d' % (buffer_size * i + i_episode+1), 'return': '%.3f' % (np.sum(rewards[:i_episode+1])/(i_episode+1))})
                pbar.update(1)
            
            if use_rnn:
                for _ in range(ppo_epoch):
                    actor_loss, critic_loss = policy.update(obs, None, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts, rnn_states, rnn_states_critic)
            else:
                for _ in range(ppo_epoch):
                    actor_loss, critic_loss = policy.update(obs, None, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts)
                
            if logger:
                logger.add_scalar('actor-loss', actor_loss, i * 50 + i_episode)
                # logger.add_scalar('reward', returns[agent], i * 50 + i_episode)
                logger.add_scalar('critic-loss', critic_loss, i * 50 + i_episode)
                logger.add_scalar(f'rew', np.sum(returns[i_episode]), (i * 50 + i_episode))
        if (total_env_steps - last_log_t)/log_interval >= 1:
            data_env = []
            data_env.append(total_env_steps)
            data_env.append(np.mean(episode_rewards))
            if env_name == 'hmpe':
                data_env.append(np.mean(num_put_trash))
            else:
                data_env.append(np.mean(win))
                data_env.append(np.mean(death_enemy))
            progress_filename = os.path.join(run_dir,'progress.csv')
            df = pd.DataFrame([data_env])
            df.to_csv(progress_filename,mode='a',header=False,index=False) 
            episode_rewards = []
            win = []
            death_enemy = []
            last_log_t = total_env_steps
    
        if (total_env_steps - last_eval_t)/eval_interval >= 1:
            low_eval_smac_policy(env, policy, total_env_steps, num_agents, max_cycles, run_dir, test_episodes, n_rollout_threads, hidden_size, use_rnn, env_name)
            last_eval_t = total_env_steps

    return None

def low_eval_smac_policy(env, policy, i_x, num_agents = 3, max_cycles = 128, run_dir = None, test_episodes = 10, n_rollout_threads = 1, hidden_size = 64, use_rnn = True, env_name = None):

    num_envs = n_rollout_threads
    obs_dim  = get_dim_from_space(env.observation_space[0])
    action_dim = get_dim_from_space(env.action_space[0])
    num_goal = len(env._goal_space)

    rewards = np.zeros((test_episodes,max_cycles,num_agents,1),dtype=np.float32)
    win = np.zeros((test_episodes))
    death_enemy = np.zeros((test_episodes))
    num_put_trash = np.zeros((test_episodes))


    for i_episode in range(test_episodes):
        getgoals = np.zeros((num_agents,1))
        stop = {}
        for agent in range(num_agents):
            getgoals[agent] = 0
            stop[agent] = 0
            # if i_episode % 10 == 0:
        obss, _, avail_action = env.reset()
        #goal_list = random.choices(list(range(num_goal)), weights=probv2, k=env.num_agents)
        for agent in range(num_agents):
            env.setgoals(agent)
        goals = np.eye(num_goal)[env.goals]
        rnn_states_batch = np.zeros((num_agents, hidden_size))
        done = False
        t = 0
        while t < max_cycles:
            
            if use_rnn:
                _,action,rnn_states_batch = policy.eval_action(obss[0], goals, avail_action[0], rnn_states_batch)
                rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            else:
                _,action = policy.eval_action(obss[0], goals, avail_action[0])
            acts = action.cpu().numpy()
            env_acts = np.split(make_onehot(acts, action_dim), num_envs)
            next_obss, _, reward, done, infos, next_avail_action = env.step(env_acts)
            #next_obss, reward, getgoal, done, _ = env.step(action)
            getgoal = env.get_goals()
            rewards[i_episode,t] = reward[0]
            getgoals[:] = getgoal

            for agent in range(num_agents):
                if getgoals[agent]:
                    env.setgoals(agent)
                    goals = np.eye(num_goal)[env.goals]

            t += 1
            obss = next_obss
            avail_action = next_avail_action 
            if done.all():
                break
        
        if env_name == 'hmpe':
            num_put_trash[i_episode] = 6 - env.remain
        else:
            if "won" in infos[0][0].keys():
                if infos[0][0]['won']:
                    win[i_episode] = 1
            death_enemy[i_episode] = env.get_enemies_death().sum()

        
        
    data_env = []
    data_env.append(i_x)
    data_env.append(np.sum(rewards)/test_episodes)
    if env_name == 'hmpe':
        data_env.append(np.sum(num_put_trash)/test_episodes)
    else:
        data_env.append(np.sum(win)/test_episodes)
        data_env.append(np.sum(death_enemy)/test_episodes)
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame([data_env])
    df.to_csv(progress_filename,mode='a',header=False,index=False) 

    return None



def low_train_policy(env, policy, num_episodes_step, logger=None, num_agents = 3, buffer_size = 32, max_cycles = 128, run_dir = None, test_episodes = 10, n_rollout_threads = 1, use_rnn = True, hidden_size = 64, log_interval = 6000, eval_interval = 20000, env_name = None):

    num_envs = n_rollout_threads
    obs_dim  = env.obs_dim
    action_dim = env.action_space
    num_goal = len(env._goal_space)
    ppo_epoch = 10
    #probv2 = [1.0/num_goal] * num_goal

    total_env_steps = 0
    last_log_t = 0
    last_eval_t = 0
    episode_rewards = []
    num_put_trash = []

    while total_env_steps < num_episodes_step:
        obs = np.zeros((buffer_size,max_cycles+1,num_agents,obs_dim),dtype=np.float32)
        rnn_states = np.zeros((buffer_size,max_cycles+1,num_agents,hidden_size),dtype=np.float32)
        rnn_states_critic = np.zeros((buffer_size,max_cycles+1,num_agents,hidden_size),dtype=np.float32)
        avail_acts = np.zeros((buffer_size,max_cycles+1,num_agents,action_dim),dtype=np.float32)
        actions = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.int32)
        actions_log_probs = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        rewards = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)
        dones = np.ones((buffer_size,max_cycles,num_agents,1),dtype=np.bool_)
        subgoals = np.zeros((buffer_size,max_cycles+1,num_agents,num_goal),dtype=np.float32)
        values = np.zeros((buffer_size,max_cycles+1,num_agents,1),dtype=np.float32)
        returns = np.zeros((buffer_size,max_cycles,num_agents,1),dtype=np.float32)

        i = int(total_env_steps)//max_cycles
        policy.lr_decay(int(total_env_steps)//max_cycles, int(num_episodes_step)//max_cycles)

        with tqdm(total=buffer_size, desc='Iteration %d' % (int(total_env_steps)//max_cycles)) as pbar:
            for i_episode in range(buffer_size):
                getgoals = np.zeros((num_agents,1))
                stop = {}
                for agent in range(num_agents):
                    getgoals[agent] = 0
                    stop[agent] = 0
                    # if i_episode % 10 == 0:
                obss, _, avail_action = env.reset()
                #goal_list = random.choices(list(range(num_goal)), weights=probv2, k=env.num_agents)
                for agent in range(num_agents):
                    env.setgoals(agent)
                goals = np.eye(num_goal)[env.goals]
                rnn_states_batch = np.zeros((num_agents, hidden_size))
                rnn_critic = np.zeros((num_agents, hidden_size))
                done = False
                t = 0
                while t < max_cycles:
                    
                    if use_rnn:
                        _,action,log_probs,_,rnn_states_batch = policy.take_action(obss[0], goals, avail_action[0], rnn_states_batch)
                        value,rnn_critic = policy.get_values(obss[0], goals, rnn_critic)
                    else:
                        _,action,log_probs,_ = policy.take_action(obss[0], goals, avail_action[0])
                        value = policy.get_values(obss[0], goals)
                    acts = action.cpu().numpy()
                    env_acts = np.split(make_onehot(acts, action_dim), num_envs)
                    next_obss, _, reward, done, infos, next_avail_action = env.step(env_acts[0].argmax(axis=-1))
                    #next_obss, reward, getgoal, done, _ = env.step(action)
                    getgoal = env.get_goals()
                    #env.get_goals()
                    #(reward != 0)
                    if use_rnn:
                        rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
                        rnn_critic = rnn_critic if isinstance(rnn_critic, np.ndarray) else rnn_critic.cpu().detach().numpy()
                        rnn_states[i_episode,t] = rnn_states_batch
                        rnn_states_critic[i_episode,t] = rnn_critic
                        
                    obs[i_episode,t] = obss[0]
                    avail_acts[i_episode,t] = avail_action[0]
                    subgoals[i_episode,t] = goals
                    actions[i_episode,t] = acts[:,np.newaxis]
                    actions_log_probs[i_episode,t] = log_probs.unsqueeze(-1).cpu().numpy()
                    rewards[i_episode,t] = reward[0]
                    dones[i_episode,t] = (done[0] | (getgoal==1))
                    values[i_episode,t] = value.unsqueeze(-1).cpu().numpy()
                    getgoals[:] = getgoal

                    for agent in range(num_agents):
                        if getgoals[agent]:
                            env.setgoals(agent)
                            goals = np.eye(num_goal)[env.goals]

                    t += 1
                    total_env_steps += 1
                    obss = next_obss
                    avail_action = next_avail_action 
                    if done.all():
                        break

                if use_rnn:
                    _,action,log_probs,_,rnn_states_batch = policy.take_action(obss[0], goals, avail_action[0], rnn_states_batch)
                    value,rnn_critic = policy.get_values(obss[0], goals, rnn_critic)
                else:
                    _,action,log_probs,_ = policy.take_action(obss[0], goals, avail_action[0])
                    value = policy.get_values(obss[0], goals)
                if use_rnn:
                    rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
                    rnn_critic = rnn_critic if isinstance(rnn_critic, np.ndarray) else rnn_critic.cpu().detach().numpy()    
                    rnn_states[i_episode,t] = rnn_states_batch
                    rnn_states_critic[i_episode,t] = rnn_critic
                
                obs[i_episode,t] = obss[0]
                avail_acts[i_episode,t] = avail_action[0]
                subgoals[i_episode,t] = goals
                values[i_episode,t] = value.unsqueeze(-1).cpu().numpy()
                
                episode_rewards.append(rewards[i_episode].sum())
                num_put_trash.append(6-env.remain)

                returns[i_episode] = compute_advantage(policy.gamma, policy.lmbda,values[i_episode],rewards[i_episode],dones[i_episode],policy.value_normalizer)
                if (i_episode+1) % 5 == 0:
                    pbar.set_postfix({'episode': '%d' % (buffer_size * i + i_episode+1), 'return': '%.3f' % (np.sum(rewards[:i_episode+1])/(i_episode+1))})
                pbar.update(1)
            
            if use_rnn:
                for _ in range(ppo_epoch):
                    actor_loss, critic_loss = policy.update(obs, None, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts, rnn_states, rnn_states_critic)
            else:
                for _ in range(ppo_epoch):
                    actor_loss, critic_loss = policy.update(obs, None, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts)
                
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
    
        if (total_env_steps - last_eval_t)/eval_interval >= 1:
            low_eval_policy(env, policy, total_env_steps, num_agents, max_cycles, run_dir, test_episodes, n_rollout_threads, hidden_size, use_rnn, env_name)
            last_eval_t = total_env_steps

    return None

def low_eval_policy(env, policy, i_x , num_agents = 3, max_cycles = 128, run_dir = None,test_episodes = 10, n_rollout_threads = 1, hidden_size = 64, use_rnn = True, env_name = None):

    num_envs = n_rollout_threads
    obs_dim  = env.obs_dim
    action_dim = env.action_space
    num_goal = len(env._goal_space)

    rewards = np.zeros((test_episodes,max_cycles,num_agents,1),dtype=np.float32)
    num_put_trash = np.zeros((test_episodes))


    for i_episode in range(test_episodes):
        getgoals = np.zeros((num_agents,1))
        stop = {}
        for agent in range(num_agents):
            getgoals[agent] = 0
            stop[agent] = 0
            # if i_episode % 10 == 0:
        obss, _, avail_action = env.reset()
        #goal_list = random.choices(list(range(num_goal)), weights=probv2, k=env.num_agents)
        for agent in range(num_agents):
            env.setgoals(agent)
        goals = np.eye(num_goal)[env.goals]
        rnn_states_batch = np.zeros((num_agents, hidden_size))
        done = False
        t = 0
        while t < max_cycles:
            
            if use_rnn:
                _,action,rnn_states_batch = policy.eval_action(obss[0], goals, avail_action[0], rnn_states_batch)
                rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            else:
                _,action = policy.eval_action(obss[0], goals, avail_action[0])
            acts = action.cpu().numpy()
            env_acts = np.split(make_onehot(acts, action_dim), num_envs)
            next_obss, _, reward, done, infos, next_avail_action = env.step(env_acts[0].argmax(axis=-1))
            #next_obss, reward, getgoal, done, _ = env.step(action)
            getgoal = env.get_goals()
            rewards[i_episode,t] = reward[0]
            getgoals[:] = getgoal

            for agent in range(num_agents):
                if getgoals[agent]:
                    env.setgoals(agent)
                    goals = np.eye(num_goal)[env.goals]

            t += 1
            obss = next_obss
            avail_action = next_avail_action 
            if done.all():
                break
        
        num_put_trash[i_episode] = 6 - env.remain

    data_env = []
    data_env.append(i_x)
    data_env.append(np.sum(rewards)/test_episodes)
    data_env.append(np.sum(num_put_trash)/test_episodes)
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame([data_env])
    df.to_csv(progress_filename,mode='a',header=False,index=False) 

    return None



