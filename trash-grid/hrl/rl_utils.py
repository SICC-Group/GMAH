from tqdm import tqdm
import numpy as np
import torch
import collections
import random

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

def train_on_policy_agent(env, agent_policy, num_episodes, logger=None):
    return_list = []
    results = {'agent_0': [], 'agent_1': [], 'agent_2': []}
    loss = {'actor': [], 'critic': []}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {}
                for agent in env.agents:
                    transition_dict[agent] = {'obs': [], 'actions': [], 'next_obs': [], 'rewards': [], 'dones': []}
                n_agent = len(env.agents)
                env.reset()
                obs, reward, _, _, _ = env.last()
                done = False
                # while not done:
                for agent in env.agent_iter():
                    action = agent_policy.take_action(obs)
                    env.step(action)
                    next_obs, reward, termination, truncation, _ = env.last()
                    # print(reward)
                    done = termination | truncation
                    transition_dict[agent]['obs'].append(obs)
                    transition_dict[agent]['actions'].append(action)
                    transition_dict[agent]['next_obs'].append(next_obs)
                    transition_dict[agent]['rewards'].append(reward)
                    transition_dict[agent]['dones'].append(done)
                    obs = next_obs
                    if agent == env.agents[0]:
                        episode_return += reward
                    if done:
                        break
                return_list.append(episode_return)
                for agent in env.agents:
                    results[agent].append(transition_dict[agent])
                    actor_loss, critic_loss = agent_policy.update(transition_dict[agent])
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
                