import math

import numpy as np
import torch

from src.inference.weighted_average import (
    haversine_distance,
    softmax,
    weighted_gps_estimate,
)


class TestHaversineDistance:
    def test_same_point(self):
        dist = haversine_distance(47.6, -122.3, 47.6, -122.3)
        assert abs(dist) < 0.001

    def test_known_distance(self):
        # London to Paris ≈ 343 km
        dist = haversine_distance(51.5, -0.1, 48.9, 2.3)
        assert 330 < dist < 360

    def test_antipodal(self):
        # ~20,000 km
        dist = haversine_distance(0, 0, 0, 180)
        assert 20000 < dist < 20100

    def test_array_input(self):
        lats = np.array([51.5, 48.9])
        lons = np.array([-0.1, 2.3])
        dist = haversine_distance(lats, lons, 51.5, -0.1)
        assert dist[0] < 0.001
        assert 330 < dist[1] < 360


class TestSoftmax:
    def test_peaked(self):
        x = np.array([10.0, 0.0, 0.0])
        w = softmax(x, temperature=0.1)
        assert w[0] > 0.99

    def test_uniform(self):
        x = np.array([1.0, 1.0, 1.0])
        w = softmax(x, temperature=1.0)
        np.testing.assert_allclose(w, 1.0 / 3, atol=1e-6)

    def test_sums_to_one(self):
        x = np.array([0.5, 1.2, -0.3, 2.1])
        w = softmax(x, temperature=0.5)
        assert abs(np.sum(w) - 1.0) < 1e-6


class TestWeightedGPSEstimate:
    def test_perfect_match(self):
        # Single tile, perfect match
        query = np.random.randn(256).astype(np.float32)
        query = query / np.linalg.norm(query)
        tile_emb = query.reshape(1, -1)
        tile_gps = np.array([[47.6, -122.3]])

        lat, lon, conf = weighted_gps_estimate(
            query, tile_emb, tile_gps, temperature=0.05, k=1
        )
        assert abs(lat - 47.6) < 0.001
        assert abs(lon - (-122.3)) < 0.001
        assert conf > 0.9

    def test_average_of_close_tiles(self):
        # Two very similar tiles, should average
        query = np.random.randn(256).astype(np.float32)
        query = query / np.linalg.norm(query)

        tile_emb = np.stack([query, query + np.random.randn(256).astype(np.float32) * 0.01])
        tile_emb = tile_emb / np.linalg.norm(tile_emb, axis=1, keepdims=True)
        tile_gps = np.array([
            [47.600, -122.300],
            [47.602, -122.302],
        ])

        lat, lon, conf = weighted_gps_estimate(
            query, tile_emb, tile_gps, temperature=0.05, k=2
        )
        # Should be between the two GPS points
        assert 47.600 < lat < 47.602
        assert -122.302 < lon < -122.300

    def test_spatial_constraint_filters_distant(self):
        query = np.random.randn(256).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Two identical tiles, one far away
        tile_emb = np.stack([query, query])
        tile_gps = np.array([
            [47.600, -122.300],
            [50.000, -100.000],  # Very far away
        ])

        lat, lon, conf = weighted_gps_estimate(
            query, tile_emb, tile_gps, temperature=0.05, k=2,
            max_distance_km=1.0,  # Should filter out the distant tile
        )
        assert abs(lat - 47.6) < 0.001
        assert abs(lon - (-122.3)) < 0.001
