from pathlib import Path

import numpy as np


def load_area_index(index_path: str) -> dict:
    """
    Load a pre-computed area index from .npz file.

    Returns dict with keys like:
        'embeddings_z17', 'gps_z17',
        'embeddings_z18', 'gps_z18',
        'center_lat', 'center_lon', 'radius_km'
    """
    data = np.load(index_path)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


def save_area_index(index: dict, output_path: str):
    """Save area index to .npz file."""
    np.savez_compressed(output_path, **index)


def get_tiles_in_region(
    area_index: dict,
    zoom: int,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract tiles within a region from the area index.

    Args:
        area_index: loaded area index dict
        zoom: zoom level to use
        center_lat, center_lon: center of search region
        radius_km: radius in km

    Returns:
        (embeddings, gps) for tiles in the region
    """
    key_emb = f"embeddings_z{zoom}"
    key_gps = f"gps_z{zoom}"

    if key_emb not in area_index or key_gps not in area_index:
        raise KeyError(f"Zoom {zoom} not found in index")

    embeddings = area_index[key_emb]
    gps = area_index[key_gps]

    # Filter by radius
    from .weighted_average import haversine_distance

    distances = haversine_distance(
        gps[:, 0], gps[:, 1],
        center_lat, center_lon,
    )
    mask = distances <= radius_km

    return embeddings[mask], gps[mask]


def list_available_areas(areas_dir: str) -> list[str]:
    """List all available area indices in the areas directory."""
    areas_path = Path(areas_dir)
    return [d.name for d in areas_path.iterdir() if d.is_dir() and (d / "area_index.npz").exists()]
