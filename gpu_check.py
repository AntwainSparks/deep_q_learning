import torch

ckpt = torch.load("runs/dqn_cartpole.pt", map_location="cpu")
print("Keys in checkpoint:", ckpt.keys())

# Inspect the first tensor
first_tensor = next(iter(ckpt["policy"].values()))
print("Tensor device:", first_tensor.device)
