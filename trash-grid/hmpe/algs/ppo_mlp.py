import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import compute_advantage

from model.acnet import ActorNet, CriticNet, PolicyNet, ValueNet

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, obs_shape, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, use_text = False, goal_space = 4, embedding_size=8):
        self.actor = PolicyNet(obs_shape, hidden_dim=32, action_dim=action_dim).to(device)
        self.critic = ValueNet(obs_shape, hidden_dim=32).to(device)
        
        self.use_text = use_text
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

    def take_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(obs.to(self.device))
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

        # print(goals.shape)
        # print(goals)
        dones = dones.squeeze()
        rewards = rewards.squeeze()
        td_target = rewards + self.gamma * self.critic(next_obs) * (1 -
                                                                       dones)
        # print('td_target', td_target.shape)
        td_delta = td_target - self.critic(obs)
        advantage = compute_advantage(self.gamma, self.lmbda,
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
            # print(td_target.shape)
            values = self.critic(obs)
            # print(values.shape)
            critic_loss = torch.mean(
                F.mse_loss(values, td_target.detach()))
            
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

def test_policy(env, policy, test_env, test_episode=1):
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