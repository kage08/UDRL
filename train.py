from model import BehaviourNetwork, EpisodeBuffer
import numpy as np
import gym
import yaml
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import torch as th
import argparse


parser = argparse.ArgumentParser(description="ƃuıuɹɐǝן ʇuǝɯǝɔɹoɟuıǝɹ")
parser.add_argument("--config", dest="conf_file", type=str, default="config.yml")
args = parser.parse_args()


with open(args.conf_file, "r") as fl:
    config = yaml.load(fl, Loader=yaml.FullLoader)

rg = np.random.RandomState(config["seed"])

device = (
    th.device("cuda")
    if th.cuda.is_available() and config["use_cuda"]
    else th.device("cpu")
)


def update_network(net: BehaviourNetwork, buffer: EpisodeBuffer, batch_size: int = 32):
    batch = buffer.sample(batch_size)
    states, horizons, rewards, actions = [], [], [], []
    for s, h, r, a in batch:
        states.append(s)
        horizons.append([h/ config["time_norm_factor"]])
        rewards.append([r/ config["reward_norm_factor"]])
        actions.append(a)
    states = th.tensor(np.array(states, dtype=np.float32)).to(device)
    horizons = th.tensor(np.array(horizons, dtype=np.float32)).to(device)
    rewards = th.tensor(np.array(rewards, dtype=np.float32)).to(device)
    actions = th.tensor(np.array(actions, dtype=np.long)).to(device)

    loss = net.update(states, horizons, rewards, actions)
    return float(loss.cpu().numpy())


def sample_trajectory(
    env, tot_reward, time_steps, bnet, render=False, rand=False, config=config, evaluate=False,
):
    if not rand:
        bnet.eval()
    state = env.reset()
    done = False
    rollout = []
    rws = 0
    t = 0
    while not done:
        if rand or (not evaluate and rg.rand() < config["epsilon"]):
            action = env.action_space.sample()
        else:
            action = bnet.choose_action(
                th.tensor(state, dtype=th.float32).to(device),
                th.tensor([float(time_steps)/ config["time_norm_factor"]]).to(device),
                th.tensor([float(tot_reward)/ config["reward_norm_factor"]]).to(device),
            )
        s1, r, done, _ = env.step(action)
        # rollout.append([state.copy(), action, r, time_steps, tot_reward])
        rollout.append([state.copy(), action, r])
        state = s1
        tot_reward -= r
        time_steps -= 1
        if render:
            env.render()
        rws += r
        t += 1
        config["epsilon"] = max(config["epsilon_min"], (1-config["epsilon_decay"])*config["epsilon"])

    tot_rws = rws
    for i, x in enumerate(rollout):
        x.append((t - i) )
        x.append(rws)
        rws -= x[2]

    return rollout, tot_rws


def get_explr_commands(buffer: EpisodeBuffer, num_sample_last: int = 100):
    r = []
    lens = []
    for _, reward, l, _ in buffer.buffer[:num_sample_last]:
        r.append(reward)
        lens.append(l)
    mean_r = np.mean(r)
    std_r = np.std(r)
    mean_l = np.mean(lens)

    def foo():
        return mean_r + rg.rand() * std_r, mean_l

    return foo, mean_r, mean_l


env = gym.make(config["env_name"])

bnet = BehaviourNetwork(
    env.observation_space.shape[0],
    env.action_space.n,
    config["hidden_dims"],
    activation=th.relu if config["activation"] == "relu" else th.tanh,
    lr=config["learning_rate"],
    device=device,
).to(device)

buffer = EpisodeBuffer(max_size=config["buffer_size"])


log_dir = os.path.join("logs", config["log_file"])
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

rws = []
trs = []
for _ in range(config["num_warmup"]):
    trj, rw = sample_trajectory(env, 100, 100, bnet, rand=True)
    buffer.add(trj, rw)
    rws.append(rw)
    trs.append(trj)
mean_l = np.mean([len(t) for t in trs])
mean_rw = np.mean(rws)
print(f"Random average reward: {mean_rw}")
writer.add_scalars("Rewards", {"Actual Reward": mean_rw, "Target Reward": mean_rw}, 0)
writer.add_scalars("Times", {"Actual Times": mean_l, "Target Times": mean_l}, 0)


ep = 1
best_rw = mean_rw
losses = []
for _ in range(config["num_steps_update"]):
    update_network(bnet, buffer, config["batch_size"])
    loss = update_network(bnet, buffer, config["batch_size"])
    losses.append(loss)
loss = np.mean(losses)
writer.add_scalar("Behaviour Network Loss", loss, ep)


while ep <= config["max_episodes"]:
    if ep % config["num_episodes_update"] == 0:
        losses = []
        for i in range(config["num_steps_update"]):
            loss = update_network(bnet, buffer, config["batch_size"])
            losses.append(loss)
        loss = np.mean(losses)
        writer.add_scalar("Behaviour Network Loss", loss, ep)
        print(f"Episode {ep}: Loss {loss}")

    explr_foo, mean_rew, mean_len = get_explr_commands(
        buffer, config["num_sample_last"]
    )
    r, l = explr_foo()

    trj, rw = sample_trajectory(env, r, l, bnet)
    buffer.add(trj, rw)

    if ep % config["eval_every"] == 0:
        r, l = explr_foo()
        rws = []
        trs = []
        for _ in range(config["eval_num_episodes"]):
            tr, rw = sample_trajectory(env, r, l, bnet, evaluate=True)
            trs.append(tr)
            rws.append(rw)
        #tr, rw = sample_trajectory(env, r, l, bnet, evaluate=True, render=True)
        mean_l = np.mean([len(t) for t in trs])
        mean_rw = np.mean(rws)
        print(f"Episode: {ep}: Mean Reward: {mean_rw:.3f}")
        writer.add_scalars(
            "Rewards", {"Actual Reward": mean_rw, "Target Reward": mean_rew}, ep
        )
        writer.add_scalars(
            "Times", {"Actual Times": mean_l, "Target Times": mean_len}, ep
        )

        if mean_rw > best_rw:
            file_name = os.path.join(log_dir, f"model_ep_{ep}.pth")
            bnet.save(file_name)

    ep += 1
