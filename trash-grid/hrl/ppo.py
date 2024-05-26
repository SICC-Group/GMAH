import os
import argparse
from typing import Optional, Tuple

from pettingzoo.mpe import simple_hmpe_v3

import torch
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
import rl_utils

from torch.utils.tensorboard import SummaryWriter


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 正交初始化
        # for m in self.actor.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.orthogonal_(m.weight)
        #         torch.nn.init.zeros_(m.bias)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # print(transition_dict)
        obs = torch.tensor(transition_dict['obs'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_obs = torch.tensor(transition_dict['next_obs'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_obs) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(obs)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(obs).gather(1,
                                                            actions)).detach()
        
        _actor_loss = []
        _critic_loss = []

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(obs).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(obs), td_target.detach()))
            
            # _actor_loss.append(actor_loss.detach())
            # _critic_loss.append(critic_loss.detach())
            _actor_loss.append(actor_loss)
            _critic_loss.append(critic_loss)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # print(actor_loss)
        # print(_actor_loss, _critic_loss)
        return torch.mean(torch.tensor(_actor_loss)), torch.mean(torch.tensor(_critic_loss))
        # return np.mean(_actor_loss, _critic_loss)

def test_policy(policy, test_env, test_episode=1):
    policy.eval()
    test_env.reset(seed=4)
    result = []
    for i in range(test_episode):
        for agent in env.agent_iter():
            obs, rew, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = policy.take_action(obs)
            env.step(action)
            next_obs = test_env.observe(agent)
            result.append({'obs': obs, 'next_obs': next_obs, 'rew': rew})
    env.close()
    return result

if __name__ == '__main__':

    # =========Logger===============
    log_path = os.path.join(os.getcwd(), 'hmpe', 'ppo')
    logger = SummaryWriter(log_path)

    render_mode = 'rgb_array' # 'human' | 'rgb_array'

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 4000
    hidden_dim = 128
    max_cycles = 60
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    seed = 144
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = simple_hmpe_v3.env(max_cycles=max_cycles, render_mode=render_mode)
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # env._seed(seed)

    state_dim = env.observation_space('agent_0').shape[0]
    action_dim = env.action_space('agent_0').n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    
    # Train
    return_list, loss = rl_utils.train_on_policy_agent(env, agent, num_episodes, logger)

    actor_model_save_path = os.path.join(log_path, 'actor_policy.pth')
    critic_model_save_path = os.path.join(log_path, 'critic_policy.pth')

    torch.save(agent.actor.state_dict(), actor_model_save_path)
    torch.save(agent.critic.state_dict(), critic_model_save_path)

    # test_env = simple_hmpe_v3.env(max_cycles=max_cycles, render_mode=render_mode)
    # result = test_policy(agent, test_env, 2)