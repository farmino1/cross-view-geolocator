import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import get_phone_augmentation, get_satellite_augmentation


def equirectangular_to_perspective(
    panorama: Image.Image,
    heading: float,
    pitch: float = 0.0,
    fov: float = 90.0,
    output_size: int = 224,
) -> Image.Image:
    """
    Extract a rectilinear (perspective) view from an equirectangular panorama.

    Simulates a phone camera looking in a specific direction.

    Args:
        panorama: equirectangular image (W x H, typically 4096x2048)
        heading: horizontal direction in degrees [0, 360)
        pitch: vertical angle in degrees [-90, 90] (0 = horizontal, -90 = down)
        fov: field of view in degrees
        output_size: output image dimensions

    Returns:
        Perspective-cropped view as PIL.Image
    """
    pano_w, pano_h = panorama.size
    half_fov = math.radians(fov) / 2
    cos_h, sin_h = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    cos_p, sin_p = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    tan_hfov = math.tan(half_fov)

    pano_arr = np.array(panorama)

    # Build grid of output pixel coords
    px = np.arange(output_size)
    py = np.arange(output_size)
    gx, gy = np.meshgrid(px, py)

    # Normalized coords [-1, 1]
    nx = (gx + 0.5) / output_size * 2 - 1
    ny = (gy + 0.5) / output_size * 2 - 1

    # Ray directions in camera space
    dx = tan_hfov * nx
    dy = tan_hfov * ny
    dz = np.ones_like(dx)

    # Rotate by pitch (around x-axis)
    dy2 = dy * cos_p - dz * sin_p
    dz2 = dy * sin_p + dz * cos_p

    # Rotate by heading (around y-axis)
    dx3 = dx * cos_h + dz2 * sin_h
    dz3 = -dx * sin_h + dz2 * cos_h

    # Convert to equirectangular coordinates
    lon = np.arctan2(dx3, dz3)
    lat = np.arctan2(-dy2, np.sqrt(dx3**2 + dz3**2 + 1e-10))

    equirect_x = (lon / math.pi + 1) / 2 * pano_w
    equirect_y = (lat / math.pi + 0.5) * pano_h

    # Bilinear sampling
    x0 = np.astype(equirect_x, np.int32) % pano_w
    y0 = np.clip(np.astype(equirect_y, np.int32), 0, pano_h - 1)
    x1 = (x0 + 1) % pano_w
    y1 = np.minimum(y0 + 1, pano_h - 1)

    fx = equirect_x - np.floor(equirect_x)
    fy = equirect_y - np.floor(equirect_y)

    # Gather and interpolate (HWC image)
    fx = fx[:, :, np.newaxis]
    fy = fy[:, :, np.newaxis]

    pixel = (
        pano_arr[y0, x0] * (1 - fx) * (1 - fy)
        + pano_arr[y0, x1] * fx * (1 - fy)
        + pano_arr[y1, x0] * (1 - fx) * fy
        + pano_arr[y1, x1] * fx * fy
    )

    out = Image.fromarray(pixel.astype(np.uint8))
    return out


def extract_perspective_crops(
    panorama: Image.Image,
    num_crops: int = 4,
    fov: float = 90.0,
    output_size: int = 224,
) -> list[Image.Image]:
    """
    Extract multiple perspective crops from a panorama at random headings.

    Each crop simulates a phone camera looking in a different direction,
    increasing dataset diversity from a single panorama.

    Args:
        panorama: equirectangular panorama image
        num_crops: number of random crops to extract
        fov: field of view per crop
        output_size: output image size

    Returns:
        List of perspective-cropped PIL images
    """
    crops = []
    for _ in range(num_crops):
        heading = random.uniform(0, 360)
        pitch = random.uniform(-15, 15)  # Mostly horizontal, slight variation
        crop = equirectangular_to_perspective(
            panorama, heading=heading, pitch=pitch, fov=fov, output_size=output_size
        )
        crops.append(crop)
    return crops


class CVCitiesDataset(Dataset):
    """
    CV-Cities dataset for cross-view geolocalization.

    Loads paired satellite and street-view panoramic images.
    For training, extracts perspective crops from panoramas to
    simulate phone camera views.

    Expected directory structure after extraction:
        data_dir/
            seattle/
                sat_images/  or satellite/   (satellite images)
                pano_images/ or streetview/  (panoramic street-view images)
            london/
                ...
    """

    def __init__(
        self,
        data_dir: str,
        cities: list[str],
        phone_augmentation: bool = True,
        satellite_augmentation: bool = True,
        crops_per_panorama: int = 4,
        val_split: float = 0.2,
        is_val: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.cities = cities
        self.crops_per_panorama = crops_per_panorama
        self.phone_aug = get_phone_augmentation(training=phone_augmentation)
        self.sat_aug = get_satellite_augmentation(training=satellite_augmentation)

        # Collect all paired samples
        self.samples = []
        for city in cities:
            city_dir = self.data_dir / city

            # Support both CV-Cities naming (sat_images/pano_images)
            # and original naming (satellite/streetview)
            sat_dir = None
            for name in ["sat_images", "satellite"]:
                candidate = city_dir / name
                if candidate.exists():
                    sat_dir = candidate
                    break

            street_dir = None
            for name in ["pano_images", "streetview"]:
                candidate = city_dir / name
                if candidate.exists():
                    street_dir = candidate
                    break

            if sat_dir is None or street_dir is None:
                continue

            sat_files = sorted(sat_dir.glob("*.jpg")) + sorted(sat_dir.glob("*.png"))
            street_files = (
                sorted(street_dir.glob("*.jpg")) + sorted(street_dir.glob("*.png"))
            )

            # Pair satellite and street-view by filename
            sat_dict = {f.stem: f for f in sat_files}
            for street_file in street_files:
                stem = street_file.stem
                if stem in sat_dict:
                    self.samples.append(
                        {
                            "satellite": sat_dict[stem],
                            "streetview": street_file,
                            "city": city,
                        }
                    )

        # Deterministic train/val split
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        split = int(len(self.samples) * (1 - val_split))
        if is_val:
            self.samples = [self.samples[i] for i in indices[split:]]
        else:
            self.samples = [self.samples[i] for i in indices[:split]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load satellite image
        sat_img = Image.open(sample["satellite"]).convert("RGB")

        # Load street-view panorama and extract perspective crop
        panorama = Image.open(sample["streetview"]).convert("RGB")
        phone_img = extract_perspective_crops(
            panorama, num_crops=1, output_size=224
        )[0]

        # Apply augmentations
        sat_tensor = self.sat_aug(sat_img)
        phone_tensor = self.phone_aug(phone_img)

        return {
            "satellite": sat_tensor,
            "phone": phone_tensor,
            "city": sample["city"],
        }


class CVCitiesDatasetWithPickle(Dataset):
    """
    Alternative loader that works directly with the CV-Cities pickle format.

    Each city zip contains pickle files with image data and GPS coordinates.
    This loader handles the raw pickle structure without requiring
    pre-extracted image directories.
    """

    def __init__(
        self,
        pickle_path: str,
        phone_augmentation: bool = True,
        satellite_augmentation: bool = True,
        crops_per_panorama: int = 4,
        val_split: float = 0.2,
        is_val: bool = False,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        import pickle

        self.crops_per_panorama = crops_per_panorama
        self.phone_aug = get_phone_augmentation(training=phone_augmentation)
        self.sat_aug = get_satellite_augmentation(training=satellite_augmentation)

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # The pickle structure varies by city but typically contains:
        # - 'satellite': list of satellite images (numpy arrays or file paths)
        # - 'streetview': list of street-view images
        # - 'gps': GPS coordinates
        self.satellite_data = data.get("satellite", data.get("sat_images", []))
        self.streetview_data = data.get(
            "streetview", data.get("street_images", data.get("pano_images", []))
        )
        self.gps_data = data.get("gps", data.get("gps_dict", None))

        if max_samples:
            self.satellite_data = self.satellite_data[:max_samples]
            self.streetview_data = self.streetview_data[:max_samples]

        # Train/val split
        n = len(self.satellite_data)
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        split = int(n * (1 - val_split))
        if is_val:
            self.indices = indices[split:]
        else:
            self.indices = indices[:split]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]

        # Load satellite image
        sat_data = self.satellite_data[real_idx]
        if isinstance(sat_data, str):
            sat_img = Image.open(sat_data).convert("RGB")
        elif isinstance(sat_data, np.ndarray):
            sat_img = Image.fromarray(sat_data)
        else:
            sat_img = sat_data

        # Load street-view panorama
        street_data = self.streetview_data[real_idx]
        if isinstance(street_data, str):
            panorama = Image.open(street_data).convert("RGB")
        elif isinstance(street_data, np.ndarray):
            panorama = Image.fromarray(street_data)
        else:
            panorama = street_data

        # Extract perspective crop from panorama
        phone_img = extract_perspective_crops(
            panorama, num_crops=1, output_size=224
        )[0]

        # Apply augmentations
        sat_tensor = self.sat_aug(sat_img)
        phone_tensor = self.phone_aug(phone_img)

        result = {
            "satellite": sat_tensor,
            "phone": phone_tensor,
        }

        if self.gps_data and real_idx < len(self.gps_data):
            gps = self.gps_data[real_idx]
            if isinstance(gps, dict):
                result["lat"] = gps.get("lat", gps.get("latitude", 0.0))
                result["lon"] = gps.get("lon", gps.get("longitude", 0.0))
            elif isinstance(gps, (list, tuple)) and len(gps) >= 2:
                result["lat"] = float(gps[0])
                result["lon"] = float(gps[1])

        return result
