import os
import math
import random
import argparse
import csv
import json
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import trange

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(seed: int, render: bool = False):
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ----------------------------
# Replay Buffer
# ----------------------------
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

# ----------------------------
# Q-Network
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------
# DQN Agent
# ----------------------------
@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    min_buffer: int = 1_000
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    tau: float = 1.0
    grad_clip_norm: float = 10.0

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, cfg: DQNConfig):
        self.device = device
        self.cfg = cfg
        self.policy = QNetwork(state_dim, action_dim).to(device)
        self.target = QNetwork(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber

        self.steps_done = 0
        self.action_dim = action_dim

    def epsilon(self):
        eps = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
              max(0.0, (self.cfg.epsilon_decay_steps - self.steps_done)) / self.cfg.epsilon_decay_steps
        return eps

    def act(self, state, eval_mode=False):
        if (not eval_mode) and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.policy(to_tensor(state, self.device).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def update(self, replay: ReplayBuffer):
        if len(replay) < self.cfg.min_buffer:
            return None

        batch = replay.sample(self.cfg.batch_size)
        states = to_tensor(np.stack(batch.state), self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = to_tensor(np.array(batch.reward), self.device).unsqueeze(1)
        next_states = to_tensor(np.stack(batch.next_state), self.device)
        dones = torch.as_tensor(np.array(batch.done, dtype=np.float32), device=self.device).unsqueeze(1)

        q_values = self.policy(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target = rewards + (1.0 - dones) * self.cfg.gamma * next_q

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def maybe_update_target(self):
        if self.cfg.tau >= 1.0:
            self.target.load_state_dict(self.policy.state_dict())
        else:
            with torch.no_grad():
                for tp, sp in zip(self.target.parameters(), self.policy.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * sp.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.steps_done = ckpt.get("steps_done", 0)

# ----------------------------
# Training / Evaluation
# ----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Train] device: {device}")

    env = make_env(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        target_update_every=args.target_update_every,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay_steps,
        tau=1.0,
        grad_clip_norm=10.0
    )

    agent = DQNAgent(state_dim, action_dim, device, cfg)
    replay = ReplayBuffer(cfg.buffer_size)

    best_avg = -float("inf")
    rewards_hist = []
    global_step = 0

    for ep in trange(args.episodes, desc="Training"):
        state, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        done = False
        loss_accum, updates = 0.0, 0

        while not done:
            action = agent.act(state, eval_mode=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            agent.steps_done += 1
            global_step += 1

            loss = agent.update(replay)
            if loss is not None:
                loss_accum += loss
                updates += 1

            if global_step % cfg.target_update_every == 0:
                agent.maybe_update_target()

        rewards_hist.append(ep_reward)

        if len(rewards_hist) >= 20:
            avg20 = np.mean(rewards_hist[-20:])
            if avg20 > best_avg:
                best_avg = avg20
                agent.save(os.path.join(args.run_dir, "dqn_cartpole.pt"))

        if args.log_every and (ep + 1) % args.log_every == 0:
            avg = np.mean(rewards_hist[-args.log_every:])
            mean_loss = (loss_accum / max(1, updates)) if updates else 0.0
            print(f"[Ep {ep+1}] Reward(avg {args.log_every}): {avg:.1f} | Loss: {mean_loss:.4f} | eps: {agent.epsilon():.3f}")

    env.close()

    # Save rewards to CSV
    rewards_csv = os.path.join(args.run_dir, "rewards.csv")
    with open(rewards_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards_hist, 1):
            w.writerow([i, r])
    print(f"Saved per-episode rewards to {rewards_csv}")

    # Save hparams
    with open(os.path.join(args.run_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Best avg(20): {best_avg:.1f}. Model saved to {os.path.join(args.run_dir, 'dqn_cartpole.pt')}")

def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Eval] device: {device}")

    env = make_env(args.seed, render=args.render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, device, DQNConfig())

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    agent.load(args.checkpoint, map_location=device)

    scores = []
    for ep in range(args.eval_episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done, total = False, 0.0
        while not done:
            action = agent.act(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        scores.append(total)
        print(f"[Eval Ep {ep+1}] Return: {total:.1f}")

    env.close()
    print(f"Average return over {args.eval_episodes} eval episodes: {np.mean(scores):.1f}")

# ----------------------------
# Entry
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DQN on CartPole (GPU Baseline)")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--min_buffer", type=int, default=1_000)
    p.add_argument("--target_update_every", type=int, default=1000)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_steps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--run_dir", type=str, default="runs_gpu_dqn",
                   help="where to save checkpoints/plots")
    p.add_argument("--log_every", type=int, default=20)

    # Eval
    p.add_argument("--eval", action="store_true")
    p.add_argument("--checkpoint", type=str, default="runs_gpu_dqn/dqn_cartpole.pt")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    if args.eval:
        evaluate(args)
    else:
        train(args)
