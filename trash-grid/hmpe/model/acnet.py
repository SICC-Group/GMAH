import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import init


class ActorNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, use_text = False, goal_space=5, embedding_size = 8):
        super(ActorNet, self).__init__()

        def init_last(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=0.01)
        
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=nn.init.calculate_gain('relu'))
        
        self.use_text = use_text
        self.goal_space = goal_space
        if use_text:
            obs_dim += goal_space
        self.feature_norm = nn.LayerNorm(obs_dim) 
        self.fc =  nn.Sequential(
            init_(torch.nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.img_emb_size = hidden_dim
        # if use_text:
        #     self.word_embedding_size = embedding_size
        #     self.img_emb_size += self.word_embedding_size
        #     self.word_embedding = nn.Embedding(goal_space, self.word_embedding_size)

        self.fnn = init_last(nn.Linear(self.img_emb_size, action_dim))


    def forward(self, obs, text):
        bs = obs.shape[0]

        if self.use_text:
            obs_joint = torch.cat((obs, text), dim=-1)
            #obs_joint = torch.cat((obs, F.one_hot(text, num_classes=self.goal_space)), dim=-1)
        else:
            obs_joint = obs
        x = self.feature_norm(obs_joint)
        x : torch.Tensor = self.fc(x)
        x = x.reshape(bs, -1)
        # if self.use_text:
        #     emb = self.word_embedding(text)
        #     #emb = F.one_hot(text, num_classes=4)
        #     x = torch.cat((x, emb), dim=1)
        x = self.fnn(x)
        return x

class CriticNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, use_text=False, goal_space=5, embedding_size=8):
        super(CriticNet, self).__init__()
        def init_last(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=1)
        
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=nn.init.calculate_gain('relu'))
        
        self.use_text = use_text
        self.goal_space = goal_space
        if use_text:
            obs_dim += goal_space
        self.feature_norm = nn.LayerNorm(obs_dim) 
        self.fc =  nn.Sequential(
            init_(torch.nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # self.img_conv = nn.Sequential(
        #     nn.Conv2d(input_channel, 8, (2, 2)),
        #     # nn.ReLU(),
        #     # nn.LeakyReLU(),
        #     nn.Tanh(),
        #     # nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(8, 16, (2, 2)),
        #     # nn.ReLU(),
        #     nn.Tanh(),
        #     # nn.LeakyReLU(),
        #     # nn.Conv2d(32, 64, (2, 2)),
        #     # # nn.ReLU()
        #     # nn.Tanh(),
        #     # nn.LeakyReLU(),
        # )

        self.img_emb_size = hidden_dim
        # if use_text:
        #     self.word_embedding_size = embedding_size
        #     self.img_emb_size += self.word_embedding_size
        #     self.word_embedding = nn.Embedding(goal_space, self.word_embedding_size)

        self.fnn = init_last(nn.Linear(self.img_emb_size , 1))

    def forward(self, obs, text):
        bs = obs.shape[0]
        if self.use_text:
            obs_joint = torch.cat((obs, text), dim=-1)
        else:
            obs_joint = obs
        x = self.feature_norm(obs_joint)
        x : torch.Tensor = self.fc(x)
        x = x.reshape(bs, -1)
        # if self.use_text:
        #     emb = self.word_embedding(text)
        #     #emb = F.one_hot(text, num_classes=4)
        #     x = torch.cat((x, emb), dim=1)
        x = self.fnn(x)
        return x.squeeze(1)

class ActorRnnNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, use_text = False, goal_space=5, embedding_size = 8):
        super(ActorRnnNet, self).__init__()

        def init_last(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=0.01)
        
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=nn.init.calculate_gain('relu'))
        
        self.use_text = use_text
        self.goal_space = goal_space
        if use_text:
            obs_dim += goal_space
        self.feature_norm = nn.LayerNorm(obs_dim) 
        self.fc =  nn.Sequential(
            init_(torch.nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.rnn_norm = nn.LayerNorm(hidden_dim)

        self.img_emb_size = hidden_dim
        self.fnn = init_last(nn.Linear(self.img_emb_size, action_dim))


    def forward(self, obs, text, rnn_states):
        bs = obs.shape[0]

        if self.use_text:
            obs_joint = torch.cat((obs, text), dim=-1)
            #obs_joint = torch.cat((obs, F.one_hot(text, num_classes=self.goal_space)), dim=-1)
        else:
            obs_joint = obs
        x = self.feature_norm(obs_joint)
        x : torch.Tensor = self.fc(x)
        x = x.reshape(bs, -1)

        if len(x.shape) == 2:
            x = x[None]
        if len(rnn_states.shape) == 2:
            rnn_states = rnn_states[None]
        
        self.rnn.flatten_parameters()
        x, hid = self.rnn(x, rnn_states)
        x = self.rnn_norm(x)
        
        x = self.fnn(x)
        return x[0], hid[0]


class CriticRnnNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, use_text=False, goal_space=5, embedding_size=8):
        super(CriticRnnNet, self).__init__()
        def init_last(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=1)
        
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),gain=nn.init.calculate_gain('relu'))
        
        self.use_text = use_text
        self.goal_space = goal_space
        if use_text:
            obs_dim += goal_space
        self.feature_norm = nn.LayerNorm(obs_dim) 
        self.fc =  nn.Sequential(
            init_(torch.nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.rnn_norm = nn.LayerNorm(hidden_dim)

        self.img_emb_size = hidden_dim


        self.fnn = init_last(nn.Linear(self.img_emb_size , 1))

    def forward(self, obs, text, rnn_states):
        bs = obs.shape[0]
        if self.use_text:
            obs_joint = torch.cat((obs, text), dim=-1)
        else:
            obs_joint = obs
        x = self.feature_norm(obs_joint)
        x : torch.Tensor = self.fc(x)
        x = x.reshape(bs, -1)

        if len(x.shape) == 2:
            x = x[None]
        if len(rnn_states.shape) == 2:
            rnn_states = rnn_states[None]

        self.rnn.flatten_parameters()
        x, hid = self.rnn(x, rnn_states)
        x = self.rnn_norm(x)

        x = self.fnn(x)

        return x[0].squeeze(1), hid[0]
























class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = torch.nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)