import torch as th
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from sortedcontainers import SortedList
import random
import numpy as np


class BehaviourNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        hidden_dims=[32, 64, 64],
        activation=F.relu,
        lr=0.001,
    ):
        super(BehaviourNetwork, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(state_dim + 2, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], num_actions))

        for l in self.layers:
            l.weight = init.xavier_normal_(l.weight)

        self.activation = activation
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, state, horizon, reward):
        x = th.cat([state, horizon, reward], dim=-1)
        for l in self.layers[:-1]:
            state = self.activation(l(state))
        return self.layers[-1](state)

    def choose_action(self, state, horizon, reward):
        probs = (
            self.forward(th.tensor(list(state) + [horizon] + [reward])).cpu().detach()
        )
        m = Categorical(probs)
        return int(m.sample().numpy())

    def update(self, states, horizons, rewards, actions):
        self.train()
        predicted_action_probs = self.forward(states, horizons, rewards)
        self.optimizer.zero_grad()
        self.loss = F.cross_entropy(predicted_action_probs, actions)
        self.loss.backward()
        self.optimizer.step()

        return self.loss


class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = SortedList(key=lambda x: -x[2])
        self.max_size = max_size

    def add(self, state, horizon, reward, action):
        while len(self.buffer) >= self.max_size:
            self.buffer.pop()
        self.buffer.add([state, horizon, reward, action])

    def pop(self):
        self.buffer.pop()

    def sample(self, size: int):
        return random.sample(self.buffer, min(size, len(self.buffer)))

    def size(self):
        return len(self.buffer)

