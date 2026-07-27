import sys
sys.path.insert(0, "cross-view-geolocator")

from src.training.train import train

train(
    data_dir="data",
    cities=["seattle", "london", "tokyo", "sydney"],
    output_dir="checkpoints/teacher",
    epochs=50,
    batch_size=256,
    lr=3e-4,
    weight_decay=0.2,
    embed_dim=256,
    warmup_epochs=1,
    checkpoint_interval=10,
    device="cuda",
)
