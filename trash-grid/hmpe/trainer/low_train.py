from tqdm import tqdm
import numpy as np
import torch
import collections
import random

from env.hmpe import hmpe
from utils.utils import Normalization

missons = ['FindTrash', 'PickUp_Small', 'PickUp_Big', 'PutTrash', 'PickUpTrash']
prob = [0.25, 0.25, 0.25, 0.25]
missonv2 = ['FindTrash','PickTrash','SplitBig','PutTrash']
probv2 = [0.1, 0.3, 0.3, 0.3]

# No_Action = -1
# Up = 1-1
# Down = 2-1
# Left = 3-1
# Right = 4-1
# Pickup = 5-1
# Putdown = 6-1
# Split = 7
Forward = 0
Left = 1
Right = 2
Pickup = 3
Split = 4
Putdown = 5

a_str = ['Up', 'Down', 'Left', 'Right', 'Pickup', 'Putdown']

def low_train_policy(env, policy, num_episodes, logger=None):
    return_list = []
    results = {'agent_0': [], 'agent_1': [], 'agent_2': []}
    loss = {'actor': [], 'critic': []}
    returns = {'agent_0': 0, 'agent_1': 0, 'agent_2': 0}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                t_data = {}
                getgoals = {}
                stop = {}
                for agent in env.n_agents:
                    t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
                    getgoals[agent] = 0
                    stop[agent] = 0
                    # if i_episode % 10 == 0:
                    returns[agent] = 0
                obss, _ = env.reset()
                goal_list = random.choices(list(range(5)), weights=prob, k=env.num_agents)
                for a, g in zip(env.n_agents, goal_list):
                    env.setgoals(a, g)
                goals = env.goals
                done = False
                while not done:
                    for agent in env.n_agents:
                        # TODO
                        if stop[agent]:
                            env.step(agent, 0)
                            # print('stop')
                            continue
                        action = policy.take_action(obss[agent], goals[agent])
                        # print('action: ', action)
                        state, reward, getgoal, done, _ = env.step(agent, action)
                        t_data[agent]['obs'].append(obss[agent])
                        t_data[agent]['goals'].append(goals[agent])
                        t_data[agent]['actions'].append(action)
                        t_data[agent]['rewards'].append(reward)

                        t_data[agent]['dones'].append(done)

                        getgoals[agent] = getgoal
                        returns[agent] += reward
                    next_obss = env.updateobs()
                    for agent in env.n_agents:
                        # t_data[agent]['next_obs'].append(next_obss[agent])
                        if stop[agent] == 0:
                            t_data[agent]['next_obs'].append(next_obss[agent])
                            # if getgoals[agent] == 1:
                            stop[agent] = getgoals[agent]
                                # actor_loss, critic_loss = policy.update(t_data[agent])
                                # t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
                                # if logger:
                                #     logger.add_scalar('actor-loss', actor_loss, i * 50 + i_episode)
                                #     # logger.add_scalar('reward', returns[agent], i * 50 + i_episode)
                                #     logger.add_scalar('critic-loss', critic_loss, i * 50 + i_episode)
                    done = done or (stop['agent_0'] and stop['agent_1'] and stop['agent_2'])
                    obss = next_obss
                if i_episode % 50 == 0:
                    for agent in env.n_agents:
                        print(f"reward: {len(t_data[agent]['rewards'])}\n",t_data[agent]['rewards'])
                        print('goal: ', t_data[agent]['goals'])
                        print('dones: ', len(t_data[agent]['dones']))
                for agent in env.n_agents:
                    # if not stop[agent]:
                    actor_loss, critic_loss = policy.update(t_data[agent])
                    t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}

                    if logger:
                        logger.add_scalar('actor-loss', actor_loss, i * 50 + i_episode)
                        # logger.add_scalar('reward', returns[agent], i * 50 + i_episode)
                        logger.add_scalar('critic-loss', critic_loss, i * 50 + i_episode)
                        logger.add_scalar(f'rew/{agent}', returns[agent], (i * 50 + i_episode))

                return_list.append(sum([returns[a] for a in env.n_agents]))


                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return loss, results

def mlow_train_policy(env: hmpe, policy, num_episodes, logger=None):
    loss = {'actor': [], 'critic': []}
    returns = {}
    norm = Normalization(env.obs_space)
    with open(logger.log_dir + '/out.txt', 'w') as f:
        for iter in range(num_episodes//4):
            with tqdm(total=4, desc='Iteration %d' % iter) as pbar:
                act_loss = critic_loss = {}
                rews = []
                for a in env.n_agents:
                    # rews[a] = 0
                    act_loss[a] = 0
                    critic_loss[a] = 0
                for i_episode in range(4):
                    t_data = {}
                    getgoals = {}
                    stop = {}
                    epi_return = 0
                    for agent in env.n_agents:
                        t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
                        getgoals[agent] = -1
                        stop[agent] = 0
                        returns[agent] = [0] * 4
                    obss, _ = env.reset()
                    goal_list = random.choices(list(range(4)), weights=prob, k=env.num_agents)
                    for a, g in zip(env.n_agents, goal_list):
                        env.setgoals(a, 0)
                    goals = env.goals
                    done = False
                    while not done:
                        for i, agent in enumerate(env.n_agents):
                            # TODO
                            if stop[agent]:
                                env.step(agent, 0-1)
                                # print('stop')
                                continue
                            # action = policy[i].take_action(norm(obss[agent]), goals[agent])
                            action = policy[i].take_action((obss[agent][0].reshape(1, 8, 8)), goals[agent])
                            state, reward, getgoal, done, _ = env.step(agent, action)
                            # t_data[agent]['obs'].append(norm(obss[agent]))
                            t_data[agent]['obs'].append((obss[agent][0].reshape(1, 8, 8)))
                            t_data[agent]['goals'].append(goals[agent])
                            t_data[agent]['actions'].append(action)
                            t_data[agent]['rewards'].append(reward)
                            if getgoal != -1:
                                t_data[agent]['dones'].append(1)
                                stop[agent] = 1
                            else:
                                t_data[agent]['dones'].append(done)
                                stop[agent] = done
                            next_obss = env.updateobs()
                            t_data[agent]['next_obs'].append(next_obss[agent][0].reshape(1,8,8))
                            getgoals[agent] = getgoal
                            returns[agent][goals[agent]] += reward
                            epi_return += reward
                        # next_obss = env.updateobs()
                        # for agent in env.n_agents:
                        #     if stop[agent] == 0:
                        #         # t_data[agent]['next_obs'].append(norm(next_obss[agent]))
                        #         t_data[agent]['next_obs'].append((next_obss[agent]))
                        #         if getgoals[agent] != -1:
                        #             stop[agent] = 1
                        done = done or (stop['agent_0'] and stop['agent_1'] and stop['agent_2'])
                        # obss = next_obss
                        rews.append(epi_return)
                    if i_episode == 0:
                        for agent in env.n_agents:
                            if sum(returns[agent]) > 0:
                                f.write(f"orin_goal: {missons[t_data[agent]['goals'][0]]}------real_goal: [{getgoals[agent]}]{missons[getgoals[agent]]}\n")
                                for i in range(len(t_data[agent]['obs'])):
                                    f.write(str(t_data[agent]['obs'][i]) + '\n')
                                    f.write('action: ' + str(t_data[agent]['actions'][i]) + str(a_str[t_data[agent]['actions'][i]]) + '\n')
                                    f.write('reward' + str(t_data[agent]['rewards'][i]) + '\n')
                    for i, agent in enumerate(env.n_agents):
                        # relabel
                        # if goals[agent] != getgoals[agent] and getgoals[agent] != -1:
                        #     # print(f"pre_goal:{goals[agent]},,,,,,,nxt_goal: {getgoals[agent]}")
                        #     for j, g in enumerate(t_data[agent]['goals']):
                        #         t_data[agent]['goals'][j] = getgoals[agent]
                        actor_loss, critic_loss = policy[i].update(t_data[agent])
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes/10 * iter + i_episode+1)})
                    pbar.update(1)

                if logger:
                    for a in env.n_agents:
                        logger.add_scalar(f'loss/{a}_actor', actor_loss, iter)
                        logger.add_scalar(f'loss/{a}_critic', critic_loss, iter)
                        for j in range(4):
                            logger.add_scalar(f'rew/{a}_{missons[j]}', returns[a][j], iter)
                    logger.add_scalar(f'reward', np.mean(rews), iter)

                    # t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
    return loss

def v2low_train(env: hmpe, policy, num_episodes, logger=None):
    loss = {'actor': [], 'critic': []}
    returns = {}
    norm = Normalization(env.observation_space)
    with open(logger.log_dir + '/out.txt', 'w') as f:
        for iter in range(num_episodes//4):
            with tqdm(total=4, desc='Iteration %d' % iter) as pbar:
                act_loss = critic_loss = {}
                rews = []
                for a in env.n_agents:
                    # rews[a] = 0
                    act_loss[a] = 0
                    critic_loss[a] = 0
                for i_episode in range(4):
                    t_data = {}
                    getgoals = {}
                    stop = {}
                    epi_return = 0
                    for agent in env.n_agents:
                        t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
                        getgoals[agent] = -1
                        stop[agent] = 0
                        returns[agent] = [0] * 4
                    obss, _ = env.reset()
                    goal_list = random.choices(list(range(4)), weights=prob, k=env.num_agents)
                    for a, g in zip(env.n_agents, goal_list):
                        env.setgoals(a, 0)
                    goals = env.goals
                    done = False
                    while not done:
                        for i, agent in enumerate(env.n_agents):
                            # TODO
                            if stop[agent]:
                                env.step(agent, 0-1)
                                # print('stop')
                                continue
                            # action = policy[i].take_action(norm(obss[agent]), goals[agent])
                            action = policy[i].take_action((obss[agent][0].reshape(1, 8, 8)), goals[agent])
                            state, reward, getgoal, done, _ = env.step(agent, action)
                            # t_data[agent]['obs'].append(norm(obss[agent]))
                            t_data[agent]['obs'].append((obss[agent][0].reshape(1, 8, 8)))
                            t_data[agent]['goals'].append(goals[agent])
                            t_data[agent]['actions'].append(action)
                            t_data[agent]['rewards'].append(reward)
                            if getgoal != -1:
                                t_data[agent]['dones'].append(1)
                                stop[agent] = 1
                            else:
                                t_data[agent]['dones'].append(done)
                                stop[agent] = done
                            next_obss = env.updateobs()
                            t_data[agent]['next_obs'].append(next_obss[agent][0].reshape(1,8,8))
                            getgoals[agent] = getgoal
                            returns[agent][goals[agent]] += reward
                            epi_return += reward
                        # next_obss = env.updateobs()
                        # for agent in env.n_agents:
                        #     if stop[agent] == 0:
                        #         # t_data[agent]['next_obs'].append(norm(next_obss[agent]))
                        #         t_data[agent]['next_obs'].append((next_obss[agent]))
                        #         if getgoals[agent] != -1:
                        #             stop[agent] = 1
                        done = done or (stop['agent_0'] and stop['agent_1'] and stop['agent_2'])
                        # obss = next_obss
                        rews.append(epi_return)
                    if i_episode == 0:
                        for agent in env.n_agents:
                            if sum(returns[agent]) > 0:
                                f.write(f"orin_goal: {missons[t_data[agent]['goals'][0]]}------real_goal: [{getgoals[agent]}]{missons[getgoals[agent]]}\n")
                                for i in range(len(t_data[agent]['obs'])):
                                    f.write(str(t_data[agent]['obs'][i]) + '\n')
                                    f.write('action: ' + str(t_data[agent]['actions'][i]) + str(a_str[t_data[agent]['actions'][i]]) + '\n')
                                    f.write('reward' + str(t_data[agent]['rewards'][i]) + '\n')
                    for i, agent in enumerate(env.n_agents):
                        # relabel
                        # if goals[agent] != getgoals[agent] and getgoals[agent] != -1:
                        #     # print(f"pre_goal:{goals[agent]},,,,,,,nxt_goal: {getgoals[agent]}")
                        #     for j, g in enumerate(t_data[agent]['goals']):
                        #         t_data[agent]['goals'][j] = getgoals[agent]
                        actor_loss, critic_loss = policy[i].update(t_data[agent])
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes/10 * iter + i_episode+1)})
                    pbar.update(1)

                if logger:
                    for a in env.n_agents:
                        logger.add_scalar(f'loss/{a}_actor', actor_loss, iter)
                        logger.add_scalar(f'loss/{a}_critic', critic_loss, iter)
                        for j in range(4):
                            logger.add_scalar(f'rew/{a}_{missons[j]}', returns[a][j], iter)
                    logger.add_scalar(f'reward', np.mean(rews), iter)

                    # t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
    return loss