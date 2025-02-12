import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import compute_advantage, update_linear_schedule, huber_loss
from utils.valuenorm import ValueNorm

from model.acnet import ActorRnnNet, CriticRnnNet

class PPO_rnn:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, obs_dim, share_obs_dim, hidden_dim,action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, use_text = False, goal_space = 4, use_obs = True ,embedding_size=8):
        self.actor = ActorRnnNet(obs_dim, hidden_dim, action_dim, use_text, goal_space=goal_space, embedding_size=embedding_size).to(device)
        if use_obs:
            self.critic = CriticRnnNet(obs_dim, hidden_dim, use_text, goal_space=goal_space, embedding_size=embedding_size).to(device)
        else:
            self.critic = CriticRnnNet(share_obs_dim, hidden_dim, use_text, goal_space=goal_space, embedding_size=embedding_size).to(device)
        self.use_text = use_text
        # 正交初始化
        # for m in self.actor.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.orthogonal_(m.weight)
        #         torch.nn.init.zeros_(m.bias)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,eps=1e-5,weight_decay=0)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,eps=1e-5,weight_decay=0)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.value_normalizer = ValueNorm(1).to(self.device)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        
    def take_action(self, obs, goal=0, avail_act = None, rnn_obs = None):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        rnn_obs = torch.tensor(rnn_obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, rnn_states = self.actor(obs, torch.LongTensor(goal).to(self.device), rnn_obs)
            logits_copy = logits.clone()
            if avail_act is not None:
                logits_copy[avail_act==0] = -1e10
            
            action_dist = torch.distributions.Categorical(logits = logits_copy)
            action = action_dist.sample()
            log_probs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
        return action_dist.probs,action,log_probs,entropy,rnn_states
    
    def eval_action(self, obs, goal=0, avail_act = None, rnn_obs = None):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        rnn_obs = torch.tensor(rnn_obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, rnn_states = self.actor(obs, torch.LongTensor(goal).to(self.device), rnn_obs)
            logits_copy = logits.clone()
            if avail_act is not None:
                logits_copy[avail_act==0] = -1e10
                action_dist = torch.distributions.Categorical(logits = logits_copy)
            action = action_dist.probs.argmax(dim=-1)
        return action_dist.probs,action,rnn_states
    
    def get_values(self, obs, goal=0, rnn_critic=None):
        obs = torch.from_numpy(obs).float().to(self.device)
        rnn_critic = torch.from_numpy(rnn_critic).float().to(self.device)
        # goal = torch.tensor([goal], dtype=torch.float).to(self.device)
        with torch.no_grad():
            values, new_rnn_critic = self.critic(obs, torch.tensor(goal).long().to(self.device), rnn_critic)
        return values, new_rnn_critic


    def update(self, obs, states, actions, actions_log_probs, subgoals, rewards, dones, values, returns, avail_acts= None, rnn_states = None, rnn_states_critic = None):
        # print(transition_dict)
        data_chunks = 4
        bs = obs.shape[0] * (obs.shape[1]-1) * obs.shape[2]
        buffer_size = obs.shape[0]
        num_agent = obs.shape[2]
        obs_dim = obs.shape[3]
        obs = torch.from_numpy(obs[:,:-1]).float().reshape(bs,1,obs_dim).to(self.device)
        rnn_states = torch.from_numpy(rnn_states[:,:-1]).float().reshape(bs,1,self.hidden_dim).to(self.device)
        rnn_states_critic = torch.from_numpy(rnn_states_critic[:,:-1]).float().reshape(bs,1,self.hidden_dim).to(self.device)
        if states is not None:
            states = torch.from_numpy(states[:,:-1]).float().reshape(bs,1,-1).to(self.device)
        else:
            states = torch.zeros((bs,1,1)).to(self.device)
        if avail_acts is not None:
            avail_acts = torch.from_numpy(avail_acts[:,:-1]).float().reshape(bs,-1).to(self.device)
        actions = torch.from_numpy(actions).long().reshape(bs,-1).to(self.device)
        actions_log_probs = torch.from_numpy(actions_log_probs).float().reshape(bs,-1).to(self.device)
        rewards = torch.from_numpy(rewards).float().reshape(bs,-1).to(self.device)
        dones_tmp = np.concatenate((np.zeros((buffer_size, 1, num_agent, 1), dtype=np.int32), dones[:,:-1]),axis=1)
        
        adv = returns.copy() - self.value_normalizer.denormalize(values[:,:-1])
        advantage_copy = adv.copy()
        advantage_copy[dones==1.0] = np.nan
        mean_advantage = np.nanmean(advantage_copy)
        std_advantage = np.nanstd(advantage_copy)
        advantages = (adv - mean_advantage) / (std_advantage + 1e-10)

        dones = torch.from_numpy(dones_tmp).int().reshape(bs,-1).to(self.device)
        values = torch.from_numpy(values[:,:-1]).float().reshape(bs,-1).to(self.device) 
        returns = torch.from_numpy(returns).float().reshape(bs,-1).to(self.device)   
        advantages = torch.from_numpy(advantages).float().reshape(bs,-1).to(self.device)   

        if self.use_text:
            goals = torch.from_numpy(subgoals[:,:-1]).long().reshape(bs,-1).to(self.device)
        else:
            goals = torch.zeros(values.shape, dtype=torch.int64).to(self.device)

        rand = torch.randperm(bs).numpy()
        
        length = bs // data_chunks
        sampler = [rand[i*length:(i+1)*length] for i in range(data_chunks)]
        _actor_loss = []
        _critic_loss = []
        for i in range(data_chunks):
            idx = sampler[i]
            if avail_acts is not None:
                actor_loss, critic_loss = self.train_loss(obs[idx], states[idx], actions[idx], actions_log_probs[idx], goals[idx], 
                            rewards[idx], dones[idx], values[idx], returns[idx], advantages[idx], rnn_states[idx], rnn_states_critic[idx], avail_acts[idx])
            else:
                actor_loss, critic_loss = self.train_loss(obs[idx], states[idx], actions[idx], actions_log_probs[idx], goals[idx], 
                            rewards[idx], dones[idx], values[idx], returns[idx], advantages[idx], rnn_states[idx], rnn_states_critic[idx])
            _actor_loss.append(actor_loss)
            _critic_loss.append(critic_loss)
            # print(actor_loss)
        # print(_actor_loss, _critic_loss)
        return torch.mean(torch.tensor(_actor_loss)), torch.mean(torch.tensor(_critic_loss))
        # return np.mean(_actor_loss, _critic_loss)
    
    def train_loss(self, obs, states, actions, old_log_probs ,goals, rewards, dones, pres_values, returns, advantages, rnn_states, rnn_states_critic, avail_acts = None):

       
        logits, _ = self.actor(obs[:,0], goals, rnn_states[:,0])
        #probs = self.actor(obs, goals.squeeze(-1))
        eps_logits = torch.ones_like(logits).to(self.device) * -1e10
        if avail_acts is not None:
            logits_mask = torch.where(avail_acts==0, eps_logits, logits)
            action_dist = torch.distributions.Categorical(logits=logits_mask)
        else:
            action_dist = torch.distributions.Categorical(logits=logits)
        entropy = action_dist.entropy().unsqueeze(-1)
        log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps,
                            1 + self.eps) * advantages # 截断
        actor_loss = -torch.sum(torch.min(surr1, surr2)*(1-dones))/(1-dones).sum() - 0.001 * torch.sum(entropy*(1-dones))/(1-dones).sum()  # PPO损失函数

        if states.shape[-1] != 1:
            values,_ = self.critic(states[:,0], goals, rnn_states_critic[:,0])[0].unsqueeze(-1)
        else:
            values = self.critic(obs[:,0], goals, rnn_states_critic[:,0])[0].unsqueeze(-1)
        #values = self.critic(obs, goals.unsqueeze(1)).unsqueeze(-1)
        # print(values.shape)
        critic_loss = self.cal_value_loss(values, pres_values,returns,1-dones)  
        #critic_loss = torch.sum(F.mse_loss(values, returns.detach())*(1-dones))/(1-dones).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss, critic_loss

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.eps,self.eps)

        self.value_normalizer.update(return_batch)
        error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = self.value_normalizer.normalize(return_batch) - values


        value_loss_clipped = huber_loss(error_clipped, 10.0)
        value_loss_original = huber_loss(error_original, 10.0)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()

        return value_loss

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