import pathlib
import re
import matplotlib.pyplot as plt

# Paths to eval logs
pairs = [
    ("CPU Baseline", "runs_cpu_baseline/eval.txt"),
    ("GPU DQN", "runs_gpu_dqn/eval.txt"),
    ("GPU DDQN", "runs_gpu_ddqn/eval.txt"),
]

def read_log(path):
    """Try multiple encodings to read a log file safely."""
    raw = pathlib.Path(path).read_bytes()
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")

vals = {}
for label, path in pairs:
    txt = read_log(path)
    m = re.search(r"Average return over .*?:\s*([\d\.]+)", txt)
    if not m:
        raise RuntimeError(f"Could not find average return in {path}")
    vals[label] = float(m.group(1))

# Plot bar chart
plt.figure(figsize=(6,4))
plt.bar(vals.keys(), vals.values(), color=["red","blue","green"])
plt.ylabel("Average Return")
plt.title("Evaluation Comparison: CPU vs GPU")
for i, (label, val) in enumerate(vals.items()):
    plt.text(i, val + 5, f"{val:.1f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("eval_comparison.png")
plt.show()
