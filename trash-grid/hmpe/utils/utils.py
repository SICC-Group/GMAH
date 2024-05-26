from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import argparse
import os
import json

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, policy, num_episodes, logger=None):
    return_list = []
    results = {'agent_0': [], 'agent_1': [], 'agent_2': []}
    loss = {'actor': [], 'critic': []}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                t_data = {}
                for agent in env.n_agents:
                    t_data[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': []}
                # num_agent = env.num_agents
                obss, _ = env.reset()
                done = False
                while not done:
                    for agent in env.n_agents:
                        action = policy.take_action(obss[agent])
                        next_state, reward, termination, truncation, _ = env.step(agent, action)
                        done = termination | truncation
                        t_data[agent]['obs'].append(obss[agent])
                        t_data[agent]['actions'].append(action)
                        t_data[agent]['rewards'].append(reward)
                        # t_data[agent]['dones'].append(done)
                        episode_return += reward
                    next_obss = env.updateobs()
                    for agent in env.n_agents:
                        t_data[agent]['next_obs'].append(next_obss[agent])
                        t_data[agent]['dones'].append(done)
                    obss = next_obss

                if i_episode == 0:
                    print(len(t_data['agent_0']['obs']))
                    print(t_data['agent_0']['rewards'])
                    print(t_data['agent_0']['actions'])
                    print(t_data['agent_0']['dones'])
                return_list.append(episode_return)
                for agent in env.n_agents:
                    results[agent].append(t_data[agent])
                    actor_loss, critic_loss = policy.update(t_data[agent])
                    loss['actor'].append(actor_loss)
                    loss['critic'].append(critic_loss)
                if logger:
                    logger.add_scalar('actor-loss', actor_loss, i * 50 + i_episode)
                    logger.add_scalar('reward', episode_return, i * 50 + i_episode)
                    logger.add_scalar('critic-loss', critic_loss, i * 50 + i_episode)

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return loss, results

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'obs': b_s, 'actions': b_a, 'next_obs': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def parsetoJson(parse: argparse.Namespace, logdir=None):
    parse = vars(parse)
    if logdir:
        with open(os.path.join(logdir, 'params.json'), 'w') as f:
            json.dump(parse, f, indent=4, ensure_ascii=False)