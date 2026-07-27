import subprocess, os, sys

def run(cmd):
    print(f"\n>>> {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", flush=True)
        if "check=True" not in cmd:
            return False
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return True

# Clone repo
if not os.path.exists("cross-view-geolocator"):
    run("git clone https://github.com/farmino1/cross-view-geolocator.git")

# Install core training deps first (Colab already has torch/torchvision)
run("pip install -q Pillow numpy mercantile requests tqdm matplotlib")

# Install huggingface_hub for dataset download
run("pip install -q huggingface_hub")

# GPU check
import torch
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print("Setup complete.")
