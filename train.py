#!/usr/bin/env -S colab run --gpu T4 --keep --timeout 86400
"""
Cross-View Geolocation — Full Training Pipeline
Run with: colab run --gpu T4 --keep train.py
"""
import subprocess
import sys
import os

REPO = "https://github.com/farmino1/cross-view-geolocator.git"
CITIES = ["seattle", "london", "tokyo", "sydney"]
DATA_DIR = "data"
CKPT_DIR = "checkpoints"

def run(cmd):
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# 1. Clone repo
if not os.path.exists("cross-view-geolocator"):
    run(f"git clone {REPO}")
os.chdir("cross-view-geolocator")

# 2. Install dependencies
run("pip install -q -r requirements.txt")
run("pip install -q huggingface_hub onnx onnxruntime onnx-tf")

# 3. Download CV-Cities dataset
from huggingface_hub import hf_hub_download
import zipfile

for city in CITIES:
    print(f"Downloading {city}...")
    zip_path = hf_hub_download(
        repo_id="gaoshuang98/CV-Cities",
        filename=f"{city}.zip",
        repo_type="dataset",
    )
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print(f"  Done.")

print("\nDataset ready.")
for city in CITIES:
    # CV-Cities zip extracts to sat_images/ and pano_images/
    city_dir = f"{DATA_DIR}/{city}"
    for name in ["sat_images", "satellite"]:
        p = f"{city_dir}/{name}"
        if os.path.isdir(p):
            sat = len(os.listdir(p))
            break
    else:
        sat = 0
    for name in ["pano_images", "streetview"]:
        p = f"{city_dir}/{name}"
        if os.path.isdir(p):
            street = len(os.listdir(p))
            break
    else:
        street = 0
    print(f"  {city}: {sat} satellite, {street} streetview")

# 4. Train teacher encoders (ResNet-50)
print("\n=== Training Teacher (ResNet-50) ===")
from src.training.train import train

train(
    data_dir=DATA_DIR,
    cities=CITIES,
    output_dir=f"{CKPT_DIR}/teacher",
    epochs=50,
    batch_size=256,
    lr=3e-4,
    weight_decay=0.2,
    embed_dim=256,
    warmup_epochs=1,
    checkpoint_interval=10,
    device="cuda",
)

# 5. Distill to MobileNetV3 student
print("\n=== Distilling to MobileNetV3 ===")
from src.training.distill import distill

distill(
    teacher_checkpoint=f"{CKPT_DIR}/teacher/best_model.pt",
    data_dir=DATA_DIR,
    cities=CITIES,
    output_dir=f"{CKPT_DIR}/student",
    epochs=30,
    batch_size=256,
    lr=1e-4,
    embed_dim=256,
    device="cuda",
)

# 6. Export to ONNX + TFLite
print("\n=== Exporting Models ===")
from src.training.export import export

export(
    checkpoint_path=f"{CKPT_DIR}/teacher/best_model.pt",
    output_dir="exported",
    embed_dim=256,
    export_satellite=True,
    export_phone=True,
    quantize_tflite=True,
)

# 7. Summary
print("\n=== Training Complete ===")
for f in os.listdir("exported"):
    size = os.path.getsize(f"exported/{f}") / (1024 * 1024)
    print(f"  {f}: {size:.1f} MB")

print("\nDownload with:")
print("  colab download -s geolocator exported/phone_encoder.tflite .")
print("  colab download -s geolocator exported/satellite_encoder.onnx .")
