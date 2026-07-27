import sys, os
sys.path.insert(0, "cross-view-geolocator")

from src.training.export import export

export(
    checkpoint_path="checkpoints/teacher/best_model.pt",
    output_dir="exported",
    embed_dim=256,
    export_satellite=True,
    export_phone=True,
    quantize_tflite=True,
)

print("\n=== Exported Files ===")
for f in os.listdir("exported"):
    size = os.path.getsize(f"exported/{f}") / (1024 * 1024)
    print(f"  {f}: {size:.1f} MB")
