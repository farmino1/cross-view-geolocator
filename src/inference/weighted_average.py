import math

import numpy as np


def haversine_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute haversine distance in kilometers.

    Args:
        lat1, lon1: first point(s) in degrees
        lat2, lon2: second point(s) in degrees

    Returns:
        Distance in km (scalar or array matching input shape)
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    x_scaled = x / temperature
    x_shifted = x_scaled - np.max(x_scaled)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def weighted_gps_estimate(
    query_embedding: np.ndarray,
    tile_embeddings: np.ndarray,
    tile_gps: np.ndarray,
    temperature: float = 0.05,
    k: int = 10,
    max_distance_km: float | None = None,
) -> tuple[float, float, float]:
    """
    Estimate GPS coordinates via weighted average of top-K matches.

    Args:
        query_embedding: [D] — L2-normalized query embedding
        tile_embeddings: [N, D] — L2-normalized tile embeddings
        tile_gps: [N, 2] — (lat, lon) per tile
        temperature: softmax temperature (lower = sharper)
        k: number of top matches to consider
        max_distance_km: if set, only average tiles within this distance
                        of the top match (spatial constraint)

    Returns:
        (estimated_lat, estimated_lon, confidence)
    """
    # Cosine similarity (dot product on normalized vectors)
    sims = tile_embeddings @ query_embedding

    # Top-K
    k = min(k, len(sims))
    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_sims = sims[top_k_idx]
    top_k_gps = tile_gps[top_k_idx]

    # Sort by similarity (best first)
    sort_order = np.argsort(-top_k_sims)
    top_k_sims = top_k_sims[sort_order]
    top_k_gps = top_k_gps[sort_order]

    # Spatial constraint: only keep tiles near the top match
    if max_distance_km is not None:
        top_gps = top_k_gps[0]
        distances = haversine_distance(
            top_k_gps[:, 0], top_k_gps[:, 1],
            top_gps[0], top_gps[1],
        )
        mask = distances <= max_distance_km
        top_k_sims = top_k_sims[mask]
        top_k_gps = top_k_gps[mask]

    if len(top_k_sims) == 0:
        # Fallback: return top match without averaging
        return float(tile_gps[0, 0]), float(tile_gps[0, 1]), 0.0

    # Softmax weights
    weights = softmax(top_k_sims, temperature)

    # Weighted average of GPS coordinates
    est_lat = float(np.sum(weights * top_k_gps[:, 0]))
    est_lon = float(np.sum(weights * top_k_gps[:, 1]))

    # Confidence from inverse entropy
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    max_entropy = math.log(len(weights))
    confidence = float(1.0 - entropy / max_entropy) if max_entropy > 0 else 1.0

    return est_lat, est_lon, confidence
