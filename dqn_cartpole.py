import os
import random
import argparse
from collections import deque, namedtuple
from dataclasses import dataclass
import csv
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import trange

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(seed: int, render: bool = False, for_video: bool = False):
    # render_mode:
    # - "human" opens a live window (use with --render)
    # - "rgb_array" enables frame capture for RecordVideo (--video_dir)
    if for_video:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    elif render:
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
    target_update_every: int = 1000  # steps (used when tau >= 1)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    tau: float = 1.0  # <1.0 => soft target update; >=1.0 => hard copy every target_update_every
    grad_clip_norm: float = 10.0
    updates_per_step: int = 1
    double_dqn: bool = False

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
        # linear decay from eps_start -> eps_end over epsilon_decay_steps
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

        # Q(s,a)
        q_values = self.policy(states).gather(1, actions)

        # --- Target calculation ---
        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN: action selection by policy, evaluation by target
                next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions)
            else:
                # Standard DQN: max over target
                next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target = rewards + (1.0 - dones) * self.cfg.gamma * next_q

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def maybe_update_target(self, global_step: int):
        if self.cfg.tau < 1.0:
            # soft update every step
            with torch.no_grad():
                for tp, sp in zip(self.target.parameters(), self.policy.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * sp.data)
        else:
            # hard copy every target_update_every steps
            if global_step % self.cfg.target_update_every == 0:
                self.target.load_state_dict(self.policy.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "steps_done": self.steps_done,
            "cfg": self.cfg.__dict__,
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
    # deterministic enough for demos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        tau=args.tau,
        grad_clip_norm=10.0,
        updates_per_step=args.updates_per_step,
        double_dqn=args.double
    )

    agent = DQNAgent(state_dim, action_dim, device, cfg)
    replay = ReplayBuffer(cfg.buffer_size)

    # save hparams for reproducibility
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_avg = -float("inf")
    global_step = 0
    rewards_hist = []

    # optional rewards.csv (episode, reward) for plotting
    rewards_csv_path = os.path.join(args.run_dir, "rewards.csv")
    write_header = not os.path.exists(rewards_csv_path)

    for ep in trange(args.episodes, desc="Training"):
        state, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        done = False
        loss_accum = 0.0
        updates = 0

        while not done:
            action = agent.act(state, eval_mode=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            agent.steps_done += 1
            global_step += 1

            # potentially do multiple SGD steps per env step
            for _ in range(cfg.updates_per_step):
                loss = agent.update(replay)
                if loss is not None:
                    loss_accum += loss
                    updates += 1

            # target update policy (soft every step, or hard every N steps)
            agent.maybe_update_target(global_step)

        rewards_hist.append(ep_reward)

        # persist per-episode reward
        with open(rewards_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["episode", "reward"])
                write_header = False
            w.writerow([ep + 1, float(ep_reward)])

        # rolling average over last 20 episodes
        if len(rewards_hist) >= 20:
            avg20 = float(np.mean(rewards_hist[-20:]))
            if avg20 > best_avg:
                best_avg = avg20
                agent.save(os.path.join(args.run_dir, "dqn_cartpole.pt"))

        if args.log_every and (ep + 1) % args.log_every == 0:
            avg = float(np.mean(rewards_hist[-args.log_every:]))
            mean_loss = (loss_accum / max(1, updates)) if updates else 0.0
            print(f"[Ep {ep+1}] Reward(avg {args.log_every}): {avg:.1f} | Loss: {mean_loss:.4f} | eps: {agent.epsilon():.3f}")

    env.close()
    print(f"Best avg(20): {best_avg:.1f}. Model saved to {os.path.join(args.run_dir, 'dqn_cartpole.pt')}")

def evaluate(args):
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Eval] device: {device}")

    # build env (rgb_array when saving video; human when rendering)
    for_video = bool(args.video_dir)
    env = make_env(args.seed, render=args.render, for_video=for_video)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda ep: True)

    # agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Use defaults here; the loaded weights determine behavior
    agent = DQNAgent(state_dim, action_dim, device, DQNConfig())

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    agent.load(args.checkpoint, map_location=device)

    episodes = args.eval_episodes
    scores = []
    traj_saved = False

    for ep in range(episodes):
        # per-episode buffers (only used if saving trajectory)
        states, actions, rewards, q_left, q_right = [], [], [], [], []

        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        while not done:
            if args.render:
                env.render()

            with torch.no_grad():
                q = agent.policy(to_tensor(state, device).unsqueeze(0)).squeeze(0).cpu().numpy()

            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if args.save_trajectory and not traj_saved:
                states.append(state)
                actions.append(int(action))
                rewards.append(float(reward))
                q_left.append(float(q[0]))
                q_right.append(float(q[1]))

            state = next_state
            total += reward

        scores.append(total)
        print(f"[Eval Ep {ep+1}] Return: {total:.1f}")

        if args.save_trajectory and not traj_saved:
            npz_path = args.traj_path
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez(npz_path,
                     states=np.array(states),
                     actions=np.array(actions),
                     rewards=np.array(rewards),
                     q_left=np.array(q_left),
                     q_right=np.array(q_right))
            print(f"Saved trajectory to {npz_path}")
            traj_saved = True

    env.close()
    print(f"Average return over {episodes} eval episodes: {np.mean(scores):.1f}")

# ----------------------------
# Entry
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Deep Q-Learning on CartPole (PyTorch + Gymnasium)")
    # Train
    p.add_argument("--episodes", type=int, default=400, help="training episodes")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--min_buffer", type=int, default=1_000)
    p.add_argument("--target_update_every", type=int, default=1000)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_steps", type=int, default=20_000)
    p.add_argument("--tau", type=float, default=1.0, help="soft target update rate; <1 => soft update every step")
    p.add_argument("--updates_per_step", type=int, default=1, help="number of SGD updates per env step")
    p.add_argument("--double", action="store_true", help="use Double DQN targets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    p.add_argument("--run_dir", type=str, default="runs", help="where to save checkpoints/plots")
    p.add_argument("--log_every", type=int, default=20)

    # Eval / Demo
    p.add_argument("--eval", action="store_true", help="run evaluation instead of training")
    p.add_argument("--checkpoint", type=str, default="runs/dqn_cartpole.pt")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--render", action="store_true", help="open a live window (human render mode)")
    p.add_argument("--video_dir", type=str, default="", help="folder to save MP4s during eval (uses RecordVideo)")

    # Trajectory capture (for plots / animation)
    p.add_argument("--save_trajectory", action="store_true", help="save first eval episode trajectory to NPZ")
    p.add_argument("--traj_path", type=str, default="runs/trajectory.npz", help="path to save trajectory NPZ")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    if args.eval:
        evaluate(args)
    else:
        train(args)
