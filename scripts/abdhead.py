import os
import signal
import sys
import torch

# Set environment proxy
os.environ['http_proxy'] = 'http://x98.local:1092'
os.environ['https_proxy'] = 'http://x98.local:1092'
print("🌐 Proxy set to http://x98.local:1092")

# Automatically select GPU and check memory
def select_gpu_and_check_memory():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(0)
        free_mem_gb = free_mem / 1024 ** 3  # Convert bytes to GB

        print(f"🖥️  GPU selected: {device_name} Free Memory: {free_mem_gb:.2f}  / {total_mem / 1024 ** 3:.2f} GB ")

    else:
        print("⚠️  No GPU available.")
        sys.exit(1)  # If no GPU available, exit. Adjust as necessary for your use case.

select_gpu_and_check_memory()

# Signal exit handler
def ExitHandler(signum, frame):
    print(f"🛑 Signal {signum} received, exiting...")
    sys.exit(0)

# Setup to look for key presses
def LookForKeys():
    # Set up a handler for Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP)
    signal.signal(signal.SIGINT, ExitHandler)
    signal.signal(signal.SIGTSTP, ExitHandler)

LookForKeys()

print(f"🆔 Process ID: {os.getpid()}")
