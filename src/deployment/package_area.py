import argparse
import time
from pathlib import Path

import numpy as np

from ..training.models import create_resnet50_encoder
from .precompute_index import load_satellite_encoder, precompute_embeddings
from .tile_downloader import download_area_tiles


def setup_area(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    checkpoint_path: str,
    output_dir: str,
    zooms: list[int] = [17, 18],
    embed_dim: int = 256,
    batch_size: int = 64,
    device: str = "cpu",
):
    """
    Full deployment pipeline for a single area:

    1. Download satellite tiles from Esri
    2. Run satellite encoder on all tiles
    3. Save embeddings + GPS as .npz file for phone
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Download tiles
    print("=" * 60)
    print("Step 1: Downloading satellite tiles")
    print("=" * 60)
    tiles_dir = str(output_path / "tiles")
    tile_results = download_area_tiles(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        zooms=zooms,
        output_dir=tiles_dir,
    )

    # Step 2: Load encoder
    print("\n" + "=" * 60)
    print("Step 2: Loading satellite encoder")
    print("=" * 60)
    encoder = load_satellite_encoder(checkpoint_path, embed_dim, device)

    # Step 3: Encode tiles
    print("\n" + "=" * 60)
    print("Step 3: Encoding tiles")
    print("=" * 60)
    all_data = {
        "center_lat": np.array(center_lat),
        "center_lon": np.array(center_lon),
        "radius_km": np.array(radius_km),
    }

    for zoom in zooms:
        if zoom not in tile_results:
            continue

        tile_info = tile_results[zoom]
        if not tile_info:
            print(f"  Zoom {zoom}: No tiles, skipping")
            continue

        embeddings, gps = precompute_embeddings(
            tile_info, encoder, batch_size, device
        )

        all_data[f"embeddings_z{zoom}"] = embeddings
        all_data[f"gps_z{zoom}"] = gps

    # Step 4: Save index
    print("\n" + "=" * 60)
    print("Step 4: Saving index")
    print("=" * 60)
    np.savez_compressed(output_path / "area_index.npz", **all_data)

    # Report sizes
    index_path = output_path / "area_index.npz"
    size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"Index saved: {index_path} ({size_mb:.1f} MB)")
    print(f"\nSummary:")
    for zoom in zooms:
        key = f"embeddings_z{zoom}"
        if key in all_data:
            print(f"  Zoom {zoom}: {all_data[key].shape[0]} tiles, "
                  f"embeddings shape {all_data[key].shape}")


def setup_multiple_areas(
    areas: list[dict],
    checkpoint_path: str,
    output_base_dir: str,
    zooms: list[int] = [17, 18],
    embed_dim: int = 256,
    batch_size: int = 64,
    device: str = "cpu",
):
    """
    Setup multiple areas in sequence.

    Args:
        areas: list of dicts with 'name', 'lat', 'lon', 'radius_km'
    """
    # Load encoder once
    encoder = load_satellite_encoder(checkpoint_path, embed_dim, device)

    for area in areas:
        print(f"\n{'=' * 60}")
        print(f"Setting up area: {area['name']}")
        print(f"{'=' * 60}")

        area_dir = Path(output_base_dir) / area["name"]
        setup_area(
            center_lat=area["lat"],
            center_lon=area["lon"],
            radius_km=area["radius_km"],
            checkpoint_path=checkpoint_path,
            output_dir=str(area_dir),
            zooms=zooms,
            embed_dim=embed_dim,
            batch_size=batch_size,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup area for phone deployment")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--name", type=str, default="area")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./deployed_areas")
    parser.add_argument("--zooms", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    setup_area(
        center_lat=args.lat,
        center_lon=args.lon,
        radius_km=args.radius,
        checkpoint_path=args.checkpoint,
        output_dir=f"{args.output}/{args.name}",
        zooms=args.zooms,
        device=args.device,
    )
