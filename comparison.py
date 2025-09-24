import os, pandas as pd, matplotlib.pyplot as plt
import csv

paths = {
    "CPU Baseline": "runs_cpu_baseline/rewards.csv",
    "GPU DQN":      "runs_gpu_dqn/rewards.csv",
    "GPU DDQN":     "runs_gpu_ddqn/rewards.csv",
}

loaded = []
min_len = None
for label, p in paths.items():
    if not os.path.exists(p):
        print(f"[skip] {label}: {p} not found")
        continue
    df = pd.read_csv(p)
    df.columns = df.columns.str.lower()  # episode, reward
    loaded.append((label, df))
    min_len = len(df) if min_len is None else min(min_len, len(df))

if not loaded:
    raise SystemExit("No rewards.csv files found; nothing to compare.")

plt.figure(figsize=(12, 7))
for label, df in loaded:
    d = df.iloc[:min_len].copy()
    d["smooth20"] = d["reward"].rolling(20).mean()
    plt.plot(d["episode"], d["smooth20"], linewidth=2, label=f"{label} (avg20)")

plt.title("CartPole — Equal-Length Comparison (avg20)")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png", dpi=200)
print("Saved plot -> comparison.png")
plt.show()
