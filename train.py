from model import ReplayBuffer, BehaviourNetwork
import numpy as np
import gym
import yaml
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter


ENV_NAME = "LunarLander-v2"


def update_network(
    net: BehaviourNetwork, optimizer, buffer: ReplayBuffer, batch_size: int = 32
):
    batch = buffer.sample(batch_size)
    states, horizons, rewards, actions = [], [], [], []
    for s, h, r, a in batch:
        states.append(s)
        horizons.append([h])
        rewards.append([r])
        actions.append(a)
    states = np.array(states, dtype=np.float32)
    horizons = np.array(horizons, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    net.update(states, horizons, rewards, actions)


def sample_trajectory(
    env, tot_reward, time_steps, bnet: BehaviourNetwork, render=False, rand=False
):
    bnet.eval()
    state = env.reset()
    done = False
    rollout = []
    while not done and time_steps > 0:
        if rand:
            action = env.action_space.sample()
        else:
            action = bnet.choose_action(state, float(time_steps), float(tot_reward))
        s1, r, done, _ = env.step(action)
        rollout.append([state.copy(), time_steps, tot_reward, action])
        state = s1
        tot_reward -= r
        time_steps -= 1
        if render:
            env.render()
    return rollout


with open("config.yml", "r") as fl:
    config = yaml.load(fl)

env = gym.make(config["env_name"])

bnet = BehaviourNetwork(
    env.observation_space.shape[0],
    env.action_space.n,
    config["hidden_dims"],
    activation=F.relu if config["activation"] == "relu" else F.tanh,
    lr=config["learning_rate"],
)

buffer = ReplayBuffer(max_size=config["buffer_size"])


log_dir = os.path.join("logs", config["log_file"])
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

for _ in range(config['num_warmup']):
    trajs = sample_trajectory(env, 1000, 100000, bnet, rand=True)
    for s,h,r,a in trajs:
        buffer.add(s,h,r,a)

#TODO: Keep Track of samples from each episode and their returns 