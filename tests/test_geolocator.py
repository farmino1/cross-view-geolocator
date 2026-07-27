import numpy as np
import pytest

from src.inference.geolocator import select_search_radius


class TestSelectSearchRadius:
    def test_high_confidence_narrow(self):
        assert select_search_radius(0.9) == 2.5

    def test_medium_confidence_medium(self):
        assert select_search_radius(0.6) == 5.0

    def test_low_confidence_wide(self):
        assert select_search_radius(0.3) == 10.0

    def test_boundary_high_medium(self):
        assert select_search_radius(0.8) == 5.0  # Just at boundary

    def test_boundary_medium_low(self):
        assert select_search_radius(0.5) == 10.0  # 0.5 is not > 0.5, falls to low

    def test_zero_confidence(self):
        assert select_search_radius(0.0) == 10.0

    def test_perfect_confidence(self):
        assert select_search_radius(1.0) == 2.5


class TestIntegration:
    """Integration tests using mock data."""

    def test_geolocator_init(self):
        """Test that Geolocator can be initialized with mock data."""
        import tempfile
        from pathlib import Path

        # Create a temporary area index
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "area_index.npz"
            np.savez_compressed(
                index_path,
                embeddings_z17=np.random.randn(100, 256).astype(np.float32),
                gps_z17=np.random.uniform(47, 48, (100, 2)).astype(np.float64),
                embeddings_z18=np.random.randn(500, 256).astype(np.float32),
                gps_z18=np.random.uniform(47, 48, (500, 2)).astype(np.float64),
                center_lat=47.6,
                center_lon=-122.3,
                radius_km=5.0,
            )

            from src.inference.geolocator import Geolocator

            geo = Geolocator(str(index_path))
            assert geo.area_index is not None
            assert "embeddings_z17" in geo.area_index

    def test_geolocate_mock(self):
        """Test geolocation with mock embeddings."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "area_index.npz"

            # Create a known arrangement: tiles on a grid around Seattle
            n_per_side = 10
            lats = np.linspace(47.55, 47.65, n_per_side)
            lons = np.linspace(-122.35, -122.25, n_per_side)

            gps = []
            for lat in lats:
                for lon in lons:
                    gps.append([lat, lon])
            gps = np.array(gps)

            # Create embeddings where each tile has a unique signature
            np.random.seed(42)
            embeddings = np.random.randn(len(gps), 256).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            np.savez_compressed(
                index_path,
                embeddings_z17=embeddings,
                gps_z17=gps,
                embeddings_z18=embeddings,
                gps_z18=gps,
                center_lat=47.6,
                center_lon=-122.3,
                radius_km=10.0,
            )

            from src.inference.geolocator import Geolocator

            geo = Geolocator(str(index_path))

            # Query with one of the known embeddings
            query = embeddings[50]  # Should match tile at index 50
            result = geo.geolocate(query_embedding=query)

            assert "lat" in result
            assert "lon" in result
            assert "confidence" in result
            # The estimated GPS should be close to the true GPS
            true_lat, true_lon = gps[50]
            assert abs(result["lat"] - true_lat) < 0.05
            assert abs(result["lon"] - true_lon) < 0.05
