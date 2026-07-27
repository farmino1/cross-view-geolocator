from pathlib import Path

import numpy as np
from PIL import Image

from .index_manager import get_tiles_in_region, load_area_index
from .weighted_average import weighted_gps_estimate


def select_search_radius(coarse_confidence: float) -> float:
    """
    Choose zoom 18 search radius based on coarse search confidence.

    Args:
        coarse_confidence: float 0-1 from zoom 17 search

    Returns:
        Radius in km
    """
    if coarse_confidence > 0.8:
        return 2.5
    elif coarse_confidence > 0.5:
        return 5.0
    else:
        return 10.0


class Geolocator:
    """
    Hierarchical geolocator that matches phone photos to satellite tiles.

    Search flow:
        1. Coarse search (zoom 17) → rough GPS + confidence
        2. Adaptive radius selection based on confidence
        3. Fine search (zoom 18) within radius → precise GPS + confidence
    """

    def __init__(
        self,
        area_index_path: str,
        model_path: str | None = None,
        embed_dim: int = 256,
    ):
        """
        Args:
            area_index_path: path to area_index.npz
            model_path: path to TFLite model (if None, assumes embeddings
                       are pre-computed and query embedding is provided directly)
            embed_dim: embedding dimension
        """
        self.area_index = load_area_index(area_index_path)
        self.embed_dim = embed_dim

        # Load TFLite model if provided
        self.interpreter = None
        if model_path:
            self._load_tflite_model(model_path)

    def _load_tflite_model(self, model_path: str):
        """Load TFLite model for on-device inference."""
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except ImportError:
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
            except ImportError:
                print("Warning: TFLite runtime not available. Model won't be loaded.")

    def encode_photo(self, photo: Image.Image | np.ndarray) -> np.ndarray:
        """
        Encode a phone photo into an embedding using the TFLite model.

        Args:
            photo: PIL Image or numpy array (H, W, 3) in RGB

        Returns:
            [embed_dim] L2-normalized embedding
        """
        if self.interpreter is None:
            raise RuntimeError("TFLite model not loaded")

        # Preprocess
        if isinstance(photo, Image.Image):
            photo = photo.convert("RGB")
            photo = photo.resize((224, 224), Image.BILINEAR)
            photo = np.array(photo, dtype=np.float32) / 255.0
        else:
            photo = np.array(photo, dtype=np.float32) / 255.0
            photo = np.array(Image.fromarray(photo.astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            ), dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        photo = (photo - mean) / std

        # Add batch dimension and transpose to NCHW for TFLite int8
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if input_details[0]["dtype"] == np.int8:
            # Quantized model: need to quantize input
            input_scale, input_zero_point = input_details[0]["quantization"]
            photo = photo / input_scale + input_zero_point
            photo = np.clip(photo, -128, 127).astype(np.int8)

        # TFLite expects NHWC format
        input_tensor = np.expand_dims(photo, axis=0)  # [1, 224, 224, 3]
        self.interpreter.set_tensor(input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(output_details[0]["index"])[0]

        # L2 normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def geolocate(
        self,
        query_embedding: np.ndarray | None = None,
        photo: Image.Image | np.ndarray | None = None,
        coarse_temperature: float = 0.05,
        fine_temperature: float = 0.03,
        k: int = 10,
    ) -> dict:
        """
        Find GPS location for a phone photo.

        Args:
            query_embedding: pre-computed [D] embedding (if provided, photo is ignored)
            photo: phone photo (used to compute embedding if query_embedding is None)
            coarse_temperature: temperature for zoom 17 search
            fine_temperature: temperature for zoom 18 search
            k: number of top matches for weighted average

        Returns:
            dict with 'lat', 'lon', 'confidence', 'coarse_lat', 'coarse_lon'
        """
        if query_embedding is None:
            if photo is None:
                raise ValueError("Either query_embedding or photo must be provided")
            query_embedding = self.encode_photo(photo)

        # Ensure query is L2-normalized
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        result = {"lat": 0.0, "lon": 0.0, "confidence": 0.0}

        # Step 1: Coarse search (zoom 17)
        if "embeddings_z17" in self.area_index:
            coarse_emb, coarse_gps = get_tiles_in_region(
                self.area_index, zoom=17,
                center_lat=self.area_index.get("center_lat", 0),
                center_lon=self.area_index.get("center_lon", 0),
                radius_km=float(self.area_index.get("radius_km", 100)),
            )

            if len(coarse_emb) > 0:
                coarse_lat, coarse_lon, coarse_conf = weighted_gps_estimate(
                    query_embedding, coarse_emb, coarse_gps,
                    temperature=coarse_temperature, k=k,
                )
                result["coarse_lat"] = coarse_lat
                result["coarse_lon"] = coarse_lon
                result["coarse_confidence"] = coarse_conf
            else:
                coarse_lat, coarse_lon, coarse_conf = 0.0, 0.0, 0.0
        else:
            coarse_lat, coarse_lon, coarse_conf = 0.0, 0.0, 0.0

        # Step 2: Select adaptive radius
        radius_km = select_search_radius(coarse_conf)
        result["search_radius_km"] = radius_km

        # Step 3: Fine search (zoom 18)
        if "embeddings_z18" in self.area_index:
            # Use coarse location as center for regional search
            if coarse_lat != 0.0 or coarse_lon != 0.0:
                center_lat, center_lon = coarse_lat, coarse_lon
            else:
                center_lat = float(self.area_index.get("center_lat", 0))
                center_lon = float(self.area_index.get("center_lon", 0))

            fine_emb, fine_gps = get_tiles_in_region(
                self.area_index, zoom=18,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
            )

            if len(fine_emb) > 0:
                fine_lat, fine_lon, fine_conf = weighted_gps_estimate(
                    query_embedding, fine_emb, fine_gps,
                    temperature=fine_temperature, k=k,
                    max_distance_km=2.0,  # Spatial constraint
                )
                result["lat"] = fine_lat
                result["lon"] = fine_lon
                result["confidence"] = fine_conf
            else:
                # Fallback to coarse result
                result["lat"] = coarse_lat
                result["lon"] = coarse_lon
                result["confidence"] = coarse_conf
        else:
            # No zoom 18 available, use coarse
            result["lat"] = coarse_lat
            result["lon"] = coarse_lon
            result["confidence"] = coarse_conf

        return result
