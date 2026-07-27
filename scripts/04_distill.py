import sys
sys.path.insert(0, "cross-view-geolocator")

from src.training.distill import distill

distill(
    teacher_checkpoint="checkpoints/teacher/best_model.pt",
    data_dir="data",
    cities=["seattle", "london", "tokyo", "sydney"],
    output_dir="checkpoints/student",
    epochs=30,
    batch_size=256,
    lr=1e-4,
    embed_dim=256,
    device="cuda",
)
