import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..training.models import create_resnet50_encoder


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_satellite_encoder(
    checkpoint_path: str, embed_dim: int = 256, device: str = "cpu"
) -> torch.nn.Module:
    """Load trained satellite encoder from checkpoint."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = create_resnet50_encoder(embed_dim=embed_dim, pretrained=False)
    encoder.load_state_dict(checkpoint["sat_encoder"])
    encoder.to(device)
    encoder.eval()
    return encoder


def precompute_embeddings(
    tile_info: list[dict],
    encoder: torch.nn.Module,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run satellite encoder on all tile images to produce embeddings.

    Args:
        tile_info: list of dicts with 'path', 'lat', 'lon' keys
        encoder: trained satellite encoder
        batch_size: inference batch size
        device: device for inference

    Returns:
        embeddings: [N, embed_dim] float32 array
        gps: [N, 2] float64 array (lat, lon)
    """
    device = torch.device(device)
    all_embeddings = []
    all_gps = []

    # Process in batches
    paths = [t["path"] for t in tile_info]
    gps = np.array([[t["lat"], t["lon"]] for t in tile_info])

    start = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(eval_transform(img))

            batch_tensor = torch.stack(images).to(device)
            embeddings = encoder(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    elapsed = time.time() - start
    print(f"Encoded {len(paths)} tiles in {elapsed:.1f}s ({len(paths)/elapsed:.1f} tiles/sec)")

    return embeddings, gps


def precompute_index(
    tiles_dir: str,
    checkpoint_path: str,
    output_path: str,
    embed_dim: int = 256,
    batch_size: int = 64,
    device: str = "cpu",
):
    """
    Full pipeline: load tiles from directory → encode → save index.
    """
    tiles_path = Path(tiles_dir)
    encoder = load_satellite_encoder(checkpoint_path, embed_dim, device)

    # Find all zoom level directories
    zoom_dirs = sorted(tiles_path.glob("zoom_*"))

    all_data = {}
    for zoom_dir in zoom_dirs:
        zoom = int(zoom_dir.name.split("_")[1])
        tile_files = sorted(zoom_dir.glob("*.jpg"))

        if not tile_files:
            continue

        print(f"Processing zoom {zoom}: {len(tile_files)} tiles...")

        tile_info = []
        for tf in tile_files:
            # Parse x_y.jpg filename
            stem = tf.stem
            x, y = map(int, stem.split("_"))

            # Compute GPS from tile coordinates
            import mercantile
            tile = mercantile.Tile(x=x, y=y, z=zoom)
            bounds = mercantile.bounds(tile)
            lat = (bounds.south + bounds.north) / 2
            lon = (bounds.west + bounds.east) / 2

            tile_info.append({
                "path": str(tf),
                "lat": lat,
                "lon": lon,
                "x": x,
                "y": y,
                "zoom": zoom,
            })

        embeddings, gps = precompute_embeddings(
            tile_info, encoder, batch_size, device
        )

        all_data[f"embeddings_z{zoom}"] = embeddings.astype(np.float32)
        all_data[f"gps_z{zoom}"] = gps.astype(np.float64)
        print(f"  Zoom {zoom}: {embeddings.shape[0]} embeddings, shape {embeddings.shape}")

    # Save
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "satellite_index.npz", **all_data)
    print(f"\nIndex saved to {output / 'satellite_index.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute satellite embeddings")
    parser.add_argument("--tiles_dir", type=str, required=True, help="Directory with tile images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="./index", help="Output directory")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    precompute_index(
        tiles_dir=args.tiles_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        device=args.device,
    )
