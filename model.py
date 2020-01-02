import torch as th
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from sortedcontainers import SortedList, SortedDict
import random
import numpy as np
import random


class BehaviourNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        hidden_dims=[32, 64, 64],
        activation=F.relu,
        lr=0.001,
        device=th.device("cpu"),
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

        self.device = device

    def forward(self, state, horizon, reward):
        x = th.cat([state, horizon, reward], dim=-1)
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)

    def choose_action(self, state, horizon, reward):
        probs = F.softmax(self.forward(state, horizon, reward), dim=-1)
        m = Categorical(probs)
        return int(m.sample().cpu().numpy())

    def update(self, states, horizons, rewards, actions):
        self.train()
        predicted_action_probs = self.forward(states, horizons, rewards)
        self.optimizer.zero_grad()
        self.loss = F.cross_entropy(predicted_action_probs, actions)
        self.loss.backward()
        self.optimizer.step()

        return self.loss.cpu().detach()

    def save(self, path):
        th.save({"model": self.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path):
        data = th.load(path, map_location=self.device)
        self.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["opt"])


class EpisodeBuffer:
    def __init__(self, max_size: int = 1000, seed=0):
        self.buffer = SortedList(key=lambda x: -x[1])
        self.max_size = max_size
        self.size = 0
        self.eps_size = 0
        self.rg = np.random.RandomState(seed)
        self.ct = 0
        self.pop_ct = 0

    def add(self, episode, total_reward):
        while self.eps_size >= self.max_size:
            for x in self.buffer:
                if x[-1] == self.pop_ct: break
            self.buffer.remove(x)
            self.size -= x[2]
            self.eps_size -= 1
            self.pop_ct += 1
        data = [episode, total_reward, len(episode), self.ct]
        self.buffer.add(data)
        self.size += len(episode)
        self.eps_size += 1
        self.ct += 1

    def sample(self, size):
        size = min(size, self.size)
        ep_sizes = np.array([x[1] for x in self.buffer])
        probs = np.exp(ep_sizes)
        probs /= probs.sum()
        #probs = probs / probs.sum()
        batch = []
        ep_idxs = self.rg.choice(range(len(self.buffer)), size)  #, p=probs)
        for i in ep_idxs:
            ep = self.buffer[i][0]
            idx = self.rg.randint(len(ep))
            batch.append(
                [ep[idx][0], ep[idx][-2], ep[idx][-1], ep[idx][1]]
            )  # State, Horizon, Desired_Reward, Action
        return batch

l = SortedDict({}, )