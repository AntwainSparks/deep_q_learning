import torch
import platform
import psutil

print("=== System Technical Specs ===")

# CPU
print(f"CPU: {platform.processor()} ({psutil.cpu_count(logical=True)} cores)")

# RAM
mem = psutil.virtual_memory()
print(f"RAM: {round(mem.total / (1024**3), 2)} GB")

# GPU / CUDA
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
else:
    print("GPU: None (CUDA not available)")

# Python / Torch versions
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
