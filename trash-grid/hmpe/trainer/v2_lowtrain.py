from tqdm import tqdm
import numpy as np
import torch
import collections
import random

from env.hmpe import hmpe
from utils.utils import Normalization

misson = ['FindTrash','PickTrash','SplitBig','PutTrash']
prob = [0.1, 0.3, 0.3, 0.3]

Forward = 0
Left = 1
Right = 2
Pickup = 3
Split = 4
Putdown = 5

astr = ['Forward', 'Left', 'Right', 'Pickup', 'Split', 'Putdown']

def v2low_train(env: hmpe, policy, num_episodes, logger=None):
    loss = {'actor': [], 'critic': []}
    norm = Normalization(env.observation_space+env.goal_space)
    with open(logger.log_dir + '/out.txt', 'w') as f:
        for iter in range(num_episodes//4):
            with tqdm(total=4, desc='Iteration %d' % iter) as pbar:
                rews = []
                for i_episode in range(4):
                    t_data = {}
                    getgoals = {}
                    stop = {}
                    epi_return = 0
                    for agent in env.n_agents:
                        t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': [], 'goals': []}
                        getgoals[agent] = -1
                        stop[agent] = -1
                    state, obss = env.reset()
                    if i_episode == 0:
                        f.write(f'Iter: {iter}\nInit-State:\n{state}\n')
                        f.write(f'Goals:PickTrash\tSplitBig\tPutTrash\n')
                    done = False
                    while not done:
                        for i, agent in enumerate(env.n_agents):
                            if stop[agent] != -1:
                                continue
                            # print(obss[agent].shape)
                            action = policy.take_action(norm(obss[agent]))
                            state, reward, getgoal, done, _ = env.step(agent, action)
                            # t_data[agent]['obs'].append(norm(obss[agent]))
                            t_data[agent]['obs'].append(obss[agent])
                            # t_data[agent]['goals'].append(goals[agent])
                            t_data[agent]['actions'].append(action)
                            t_data[agent]['rewards'].append(reward)
                            if getgoal != -1:
                                t_data[agent]['dones'].append(1)
                                stop[agent] = 1
                            else:
                                t_data[agent]['dones'].append(done)
                                stop[agent] = done
                            next_obss = env.updateobs()
                            t_data[agent]['next_obs'].append(norm(next_obss[agent]))
                            # getgoals[agent] = getgoal
                            epi_return += reward
                            if i_episode == 0:
                                f.write(f'{agent}_action: {astr[action]}\n')
                                f.write(f"state:\n{state}\n")
                                f.write(f"obs: {obss[agent]}\n")
                                f.write(f"reward: {reward}\n")
                        done = done or (stop['agent_0'] and stop['agent_1'] and stop['agent_2'])
                        next_obss = env.updateobs()
                        obss = next_obss
                    rews.append(epi_return)

                    for i, agent in enumerate(env.n_agents):
                        # relabel
                        # if goals[agent] != getgoals[agent] and getgoals[agent] != -1:
                        #     # print(f"pre_goal:{goals[agent]},,,,,,,nxt_goal: {getgoals[agent]}")
                        #     for j, g in enumerate(t_data[agent]['goals']):
                        #         t_data[agent]['goals'][j] = getgoals[agent]
                        actor_loss, critic_loss = policy.update(t_data[agent])
                        loss['actor'].append(actor_loss)
                        loss['critic'].append(critic_loss)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (4 * iter + i_episode+1)})
                    pbar.update(1)

                if logger:
                    logger.add_scalar('actor-loss', np.mean(loss['actor']), iter)
                    logger.add_scalar('reward', np.mean(rews), iter)
                    logger.add_scalar('critic-loss', np.mean(loss['critic']), iter)
                loss['actor'] = loss['critic'] = []
    return loss