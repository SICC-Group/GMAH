import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNet(torch.nn.Module):
    def __init__(self, obs_shape, action_dim, use_text = False, goal_space=5, embedding_size = 8):
        super(ActorNet, self).__init__()

        self.img_shape = obs_shape
        n = self.image_h = obs_shape[1]
        m = self.image_w = obs_shape[2]
        input_channel = obs_shape[0]
        self.use_text = use_text
        self.img_conv = nn.Sequential(
            nn.Conv2d(input_channel, 16, (2, 2)),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Tanh(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.Conv2d(32, 64, (2, 2)),
            # # nn.ReLU()
            # nn.Tanh(),
            # nn.LeakyReLU(),
        )

        # self.img_emb_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.img_emb_size = 6*6*32

        if use_text:
            self.word_embedding_size = embedding_size
            self.img_emb_size += self.word_embedding_size
            self.word_embedding = nn.Embedding(goal_space, self.word_embedding_size)

        self.fnn = nn.Sequential(
            nn.Linear(self.img_emb_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs, text):
        # print()
        bs = obs.shape[0]
        # print(bs)
        x : torch.Tensor = self.img_conv(obs)
        x = x.reshape(bs, -1)
        # print(x.shape)
        if self.use_text:
            # emb = self.word_embedding(text)
            emb = F.one_hot(text, num_classes=4)
            x = torch.cat((x, emb), dim=1)
        x = self.fnn(x)
        return F.softmax(x, dim=1)

class CriticNet(torch.nn.Module):
    def __init__(self, obs_shape, use_text=False, goal_space=5, embedding_size=8):
        super(CriticNet, self).__init__()

        self.img_shape = obs_shape
        self.use_text=use_text
        n = self.image_h = obs_shape[1]
        m = self.image_w = obs_shape[2]
        input_channel = obs_shape[0]

        self.img_conv = nn.Sequential(
            nn.Conv2d(input_channel, 16, (2, 2)),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Conv2d(32, 64, (2, 2)),
            # nn.Tanh(),
            # nn.ReLU()
            # nn.LeakyReLU(),
        )

        # self.img_emb_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.img_emb_size = 6*6*32

        if use_text:
            self.word_embedding_size = embedding_size
            self.img_emb_size += self.word_embedding_size
            self.word_embedding = nn.Embedding(goal_space, self.word_embedding_size)

        self.fnn = nn.Sequential(
            nn.Linear(self.img_emb_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, text):
        bs = obs.shape[0]
        x : torch.Tensor = self.img_conv(obs)
        x = x.reshape(bs, -1)
        if self.use_text:
            # emb = self.word_embedding(text)
            emb = F.one_hot(text, num_classes=4)
            x = torch.cat((x, emb), dim=1)
        x = self.fnn(x)
        return x.squeeze(1)


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