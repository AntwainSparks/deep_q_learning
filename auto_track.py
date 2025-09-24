import os, json, pandas as pd, matplotlib.pyplot as plt
import csv

BASE = "."
RUN_DIRS = [
    "runs_cpu_baseline",
    "runs_gpu_dqn",
    "runs_gpu_ddqn",
]

def label_for(run_dir: str) -> str:
    # Prefer a human label from hparams.json if available
    hp = os.path.join(run_dir, "hparams.json")
    if os.path.exists(hp):
        try:
            with open(hp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "label" in data and data["label"]:
                return str(data["label"])
        except Exception:
            pass
    # Fallback: prettify folder name
    return run_dir.replace("runs_", "").replace("_", " ").title()

series = []
min_len = None

for run in RUN_DIRS:
    csv_path = os.path.join(BASE, run, "rewards.csv")
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found")
        continue
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()  # episode, reward
    if not {"episode", "reward"}.issubset(df.columns):
        print(f"[skip] {csv_path} lacks required columns")
        continue
    series.append((label_for(run), df))
    min_len = len(df) if min_len is None else min(min_len, len(df))

if not series:
    raise SystemExit("No rewards.csv files found in expected run directories.")

plt.figure(figsize=(12, 7))
for label, df in series:
    d = df.iloc[:min_len].copy()
    d["smooth"] = d["reward"].rolling(20).mean()
    plt.plot(d["episode"], d["smooth"], linewidth=2, label=f"{label} (avg20)")

plt.title("CartPole Training — CPU Baseline vs GPU DQN vs GPU DDQN")
plt.xlabel("Episode")
plt.ylabel("Reward (avg20)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("auto_track_comparison.png", dpi=200)
print("Saved plot -> auto_track_comparison.png")
plt.show()
