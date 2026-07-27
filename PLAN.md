# Cross-View Geolocation System — Implementation Plan

## Overview

A system that geolocates a phone camera photo by matching it against pre-computed
satellite tile embeddings. Runs entirely on-device with no internet required.

**Core pipeline:**
```
Phone photo → MobileNetV3 encoder → 256-dim embedding
                                          ↓
                    Hierarchical search against pre-computed satellite embeddings
                                          ↓
                              GPS coordinates + confidence score
```

**Design principles:**
- Two separate encoders (satellite and phone) trained with contrastive learning
- On-device inference via TFLite (no internet needed)
- Hierarchical coarse-to-fine search with adaptive radius
- Weighted average of top-K matches for sub-tile accuracy
- Multiple areas supported, each stored as a small index (~4MB)

---

## 1. Architecture

### 1.1 Encoder Design

Two separate encoders project into a shared 256-dimensional embedding space:

```
Satellite tile (224x224) → ResNet-50 → Linear(2048, 256) → L2 norm → embedding
Phone photo   (224x224) → ResNet-50 → Linear(2048, 256) → L2 norm → embedding
```

- **Separate encoders** because the perspectives are maximally different
  (top-down satellite vs. horizontal phone camera)
- **Linear projection** (not MLP) — CLIP found this sufficient for cross-modal
  contrastive learning with large datasets
- **L2 normalization** so cosine similarity equals dot product

### 1.2 Mobile Deployment

After training, the phone encoder is distilled to MobileNetV3-Large:

```
Training:   ResNet-50 → Linear(2048, 256) → L2 norm  (teacher)
Mobile:     MobileNetV3-Large → Linear(960, 256) → L2 norm  (student)
```

The satellite encoder stays as ResNet-50 — it only runs on desktop to pre-compute
embeddings.

### 1.3 Mobile Model Specs

| Model | Size | Inference (SD8Gen2) | Purpose |
|-------|------|---------------------|---------|
| MobileNetV3-Large | ~5.4MB | ~5ms | Phone encoder (on-device) |
| ResNet-50 | ~97MB | N/A | Satellite encoder (desktop only) |

---

## 2. Loss Function: Symmetric InfoNCE

CLIP-style bidirectional contrastive loss. For a batch of N (satellite, phone) pairs:

```python
def symmetric_infonce(sat_emb, phone_emb, temperature=0.07):
    """
    sat_emb:   [batch_size, 256] — L2-normalized satellite embeddings
    phone_emb: [batch_size, 256] — L2-normalized phone embeddings
    """
    # NxN similarity matrix
    logits = sat_emb @ phone_emb.T / temperature  # [N, N]
    labels = torch.arange(len(logits), device=logits.device)

    # Bidirectional: satellite-to-phone + phone-to-satellite
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    return loss
```

**Why this works:**
- Each satellite image must match its correct phone photo out of N candidates
- With batch size 256, each image gets 255 negatives
- Forces the model to learn geographic features (building layouts, road patterns,
  terrain shapes) rather than superficial pixel statistics
- Temperature τ=0.07 (learnable, clipped to [0.01, 100]) controls sharpness

**Key hyperparameters:**

| Parameter | Value | Source |
|-----------|-------|--------|
| Temperature | 0.07 (learnable) | CLIP standard |
| Batch size | 256 | Limited by Colab T4 VRAM |
| Optimizer | AdamW | lr=3e-4, weight_decay=0.2, betas=(0.9, 0.95) |
| LR schedule | Cosine annealing | 1-epoch linear warmup |
| Epochs | 50-100 | Monitor validation recall@1 |

---

## 3. Dataset: CV-Cities

### 3.1 Dataset Overview

| Property | Value |
|----------|-------|
| Source | HuggingFace: `gaoshuang98/CV-Cities` |
| License | BSD-3-Clause (permissive) |
| Total pairs | ~223,000 across 16 cities |
| Ground images | Google Street View panoramas (4096x2048 equirectangular) |
| Satellite images | Google Maps satellite imagery |
| GPS coordinates | Included in `gps_dict_10_cities.pkl` |

### 3.2 Selected Cities (subset for training)

| City | Zip Size | Pairs (approx) | Why |
|------|----------|-----------------|-----|
| seattle | 3.08 GB | ~15,000 | US city, urban/suburban mix |
| london | 3.33 GB | ~15,000 | European architecture |
| tokyo | 2.86 GB | ~15,000 | Dense Asian city |
| sydney | 953 MB | ~8,000 | Southern hemisphere, small |
| **Total** | **~10.2 GB** | **~53,000** | |

### 3.3 Panorama → Perspective Crop Extraction

CV-Cities uses equirectangular panoramas (4096x2048), not standard phone photos.
We extract perspective crops to simulate phone camera views:

```python
def extract_perspective_crop(panorama, heading, pitch, fov=90, output_size=224):
    """
    Crop a rectilinear view from an equirectangular panorama.
    Simulates a phone camera looking in a specific direction.

    Args:
        panorama: numpy array (H, W, 3) — equirectangular image
        heading: float 0-360 — compass direction (yaw)
        pitch: float -90 to 90 — vertical angle (0=horizontal, -90=down)
        fov: float — field of view in degrees
        output_size: int — output image size

    Returns:
        PIL.Image — perspective-cropped view
    """
    # Convert equirectangular coordinates to pixel coordinates
    # Apply perspective transform (rectilinear projection)
    # Crop at specified heading/pitch/fov
    # Resize to output_size x output_size
    ...
```

Each panorama generates multiple phone-like crops at different headings,
increasing dataset diversity. For training, we randomly sample 1-4 crops
per panorama at random headings.

### 3.4 Augmentation Pipelines

```python
# Phone/horizontal view augmentation
phone_augment = T.Compose([
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Satellite/top-down view augmentation
sat_augment = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

**Why these augmentations:**
- **Random crop**: Forces model to recognize features from partial views
- **Color jitter**: Prevents color histogram shortcuts (critical for contrastive learning)
- **Horizontal flip**: Standard spatial augmentation
- **No grayscale for phone**: Phone photos have color information that's geographically
  meaningful (brick buildings, green trees, blue roofs)

---

## 4. Training Pipeline

### 4.1 Phase 1: Train Teacher Encoders (ResNet-50)

**Environment:** Google Colab free tier (T4 GPU, 16GB VRAM)

| Setting | Value |
|---------|-------|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Embedding dim | 256 |
| Projection | Linear(2048, 256) per encoder |
| Temperature | 0.07 (learnable, log-parameterized, clipped) |
| Batch size | 256 |
| Optimizer | AdamW, lr=3e-4, weight_decay=0.2, betas=(0.9, 0.95) |
| LR schedule | Cosine annealing with 1-epoch warmup |
| Epochs | 50-100 |
| Mixed precision | AMP (fp16) — essential for T4 VRAM |
| Dataset | CV-Cities: seattle + london + tokyo + sydney |

**Validation metrics:**
- Recall@1: percentage where the correct match is the top-1 result
- Recall@5: percentage where the correct match is in the top-5
- Recall@10: percentage where the correct match is in the top-10
- Validation set: 20% of each city held out

**Expected training time on Colab T4:**
- ~53K pairs, batch 256 = ~207 batches/epoch
- ~15-20 seconds per batch with AMP
- ~52-69 minutes per epoch
- 50 epochs = ~44-58 hours total
- Colab free tier has ~12hr sessions, so ~4-5 sessions needed

### 4.2 Phase 2: Knowledge Distillation to MobileNetV3

**Purpose:** Compress ResNet-50 phone encoder to MobileNetV3-Large for mobile.

| Setting | Value |
|---------|-------|
| Teacher | Trained ResNet-50 phone encoder (frozen) |
| Student | MobileNetV3-Large (random init or ImageNet pretrained) |
| Loss | MSE between student and teacher embeddings |
| Epochs | 30-50 |
| Optimizer | AdamW, lr=1e-4 |
| Batch size | 256 |
| Augmentation | Same as Phase 1 |

The student learns to produce the same embeddings as the teacher, inheriting
its cross-view matching ability in a smaller model.

**Alternative (simpler):** Train MobileNetV3 directly on the contrastive task.
Skip distillation. May need more epochs but avoids complexity.

### 4.3 Phase 3: Export to TFLite

```python
import tensorflow as tf

# Convert PyTorch model to ONNX, then to TFLite
# Or use tf2onnx + TFLite converter

# Quantize to int8 for faster inference on mobile
converter = tf.lite.TFLiteConverter.from_saved_model("phone_encoder")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # int8 quantization
tflite_model = converter.convert()

with open("phone_encoder.tflite", "wb") as f:
    f.write(tflite_model)
```

**Export two files:**
1. `phone_encoder.tflite` — quantized MobileNetV3 (~5.5MB)
2. `satellite_encoder.onnx` — full ResNet-50 for desktop deployment

---

## 5. Deployment Pipeline (Desktop)

### 5.1 Area Setup

Given a center point and radius, generate the satellite tile index:

```python
def setup_area(center_lat, center_lon, radius_km, satellite_encoder_path):
    """
    1. Determine tile bounds at zoom 17 and 18
    2. Download satellite tiles from Esri
    3. Run satellite encoder on all tiles → embeddings
    4. Save as .npz file
    """
    # Get tiles covering the area
    tiles_z17 = get_tiles_in_radius(center_lat, center_lon, radius_km, zoom=17)
    tiles_z18 = get_tiles_in_radius(center_lat, center_lon, radius_km, zoom=18)

    # Download from Esri (free, no API key)
    images_z17 = [download_tile(t) for t in tiles_z17]
    images_z18 = [download_tile(t) for t in tiles_z18]

    # Pre-compute embeddings (float16 for storage efficiency)
    embeddings_z17 = satellite_encoder.encode_batch(images_z17).half()
    embeddings_z18 = satellite_encoder.encode_batch(images_z18).half()

    # Extract GPS per tile
    gps_z17 = np.array([tile_to_gps(t) for t in tiles_z17])
    gps_z18 = np.array([tile_to_gps(t) for t in tiles_z18])

    # Save
    np.savez_compressed("area_index.npz",
        embeddings_z17=embeddings_z17, gps_z17=gps_z17,
        embeddings_z18=embeddings_z18, gps_z18=gps_z18,
        center_lat=center_lat, center_lon=center_lon, radius_km=radius_km,
    )
```

### 5.2 Tile Download Source

**Esri World Imagery** — free, no API key, sub-meter resolution:

```python
import mercantile
import requests
from PIL import Image
from io import BytesIO

def download_tile(lat, lon, zoom):
    """Download a single satellite tile from Esri."""
    tile = mercantile.tile(lon, lat, zoom)
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{tile.z}/{tile.y}/{tile.x}"
    )
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))
```

### 5.3 Tile Scale Reference

| Zoom | Tile Size (mid-lat) | Tiles in 10x10km | Purpose |
|------|---------------------|-------------------|---------|
| 17 | ~306m | ~1,100 | Coarse localization |
| 18 | ~120m | ~7,000 | Fine localization |
| 19 | ~60m | ~28,000 | Precision (optional) |

### 5.4 Tile Size Calculation

```python
import math

def tile_size_meters(zoom, latitude):
    """Width/height of a 256px tile in meters at given latitude."""
    equatorial_circumference = 40_075_016.686  # meters
    return equatorial_circumference * math.cos(math.radians(latitude)) / (2 ** zoom)
```

---

## 6. Inference Pipeline (Phone)

### 6.1 Hierarchical Search

```
Step 1 — Coarse search (zoom 17):
  Load zoom 17 index (~1,100 vectors, ~0.56MB)
  Compute cosine similarity against all vectors
  Top-K weighted average → coarse GPS (~300m accuracy)
  Compute confidence from weight distribution entropy

Step 2 — Adaptive radius selection:
  confidence > 0.8 → radius = 2.5km  (high confidence, narrow search)
  confidence > 0.5 → radius = 5.0km  (medium confidence)
  confidence ≤ 0.5 → radius = 10.0km (low confidence, full area fallback)

Step 3 — Fine search (zoom 18):
  Load zoom 18 index for the selected radius
  (441 vectors for 2.5km, up to 7,000 for full area)
  Top-K weighted average → fine GPS (~120m accuracy)
  Compute confidence

Step 4 — Optional refinement:
  If fine confidence > 0.8, can optionally search zoom 19
  within an even tighter radius for ~60m accuracy
```

### 6.2 Weighted Average with Spatial Constraint

```python
def weighted_gps_constrained(
    query_embedding,       # [256] — L2-normalized
    tile_embeddings,       # [N, 256] — L2-normalized, from index
    tile_gps,              # [N, 2] — (lat, lon) per tile
    temperature=0.05,      # Controls sharpness of weight distribution
    k=10,                  # Number of top matches to consider
    max_distance_km=2.0,   # Spatial constraint radius
):
    """
    Find GPS estimate via weighted average of top-K matches,
    constrained to tiles within max_distance_km of the top match.

    Returns:
        (estimated_lat, estimated_lon, confidence)
    """
    # Compute similarities
    sims = tile_embeddings @ query_embedding  # [N]

    # Top-K
    top_k_sims, top_k_idx = torch.topk(sims, k=min(k, len(sims)))
    top_k_gps = tile_gps[top_k_idx]

    # Spatial constraint: only keep tiles near the top match
    top_gps = top_k_gps[0]
    distances = haversine_batch(top_k_gps, top_gps)  # [K]
    mask = distances <= max_distance_km

    sims_filtered = top_k_sims[mask]
    gps_filtered = top_k_gps[mask]

    # Softmax weights (temperature controls sharpness)
    weights = F.softmax(sims_filtered / temperature, dim=0)

    # Weighted average of GPS coordinates
    est_lat = (weights * gps_filtered[:, 0]).sum().item()
    est_lon = (weights * gps_filtered[:, 1]).sum().item()

    # Confidence from inverse entropy
    # Low entropy = peaked distribution = high confidence
    # High entropy = flat distribution = low confidence
    entropy = -(weights * torch.log(weights + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(float(len(weights))))
    confidence = (1.0 - entropy / max_entropy).item()

    return est_lat, est_lon, confidence


def haversine_batch(coords1, coords2):
    """
    Compute haversine distance in km between coords1 and coords2.
    coords: [N, 2] where each row is (lat, lon) in degrees.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1 = torch.radians(coords1[:, 0]), torch.radians(coords1[:, 1])
    lat2, lon2 = torch.radians(coords2[0]), torch.radians(coords2[1])

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return R * c
```

### 6.3 Adaptive Radius Logic

```python
def select_search_radius(coarse_confidence, coarse_lat, coarse_lon):
    """
    Choose zoom 18 search radius based on coarse search confidence.

    Args:
        coarse_confidence: float 0-1 from zoom 17 search
        coarse_lat, coarse_lon: estimated position from zoom 17

    Returns:
        (radius_km, center_lat, center_lon)
    """
    if coarse_confidence > 0.8:
        radius_km = 2.5
    elif coarse_confidence > 0.5:
        radius_km = 5.0
    else:
        radius_km = 10.0  # full area fallback

    return radius_km, coarse_lat, coarse_lon
```

### 6.4 Full Inference Flow

```python
class Geolocator:
    def __init__(self, model_path, area_index_path):
        self.model = load_tflite_model(model_path)  # MobileNetV3
        self.area = load_area_index(area_index_path)

    def geolocate(self, phone_photo):
        """
        Args:
            phone_photo: PIL.Image — photo from phone camera

        Returns:
            dict with 'lat', 'lon', 'confidence'
        """
        # 1. Encode phone photo
        query = self.encode_photo(phone_photo)  # [256], L2-normalized

        # 2. Coarse search (zoom 17)
        coarse_lat, coarse_lon, coarse_conf = weighted_gps_constrained(
            query,
            self.area["embeddings_z17"],
            self.area["gps_z17"],
            temperature=0.05,
            k=10,
            max_distance_km=5.0,
        )

        # 3. Select adaptive radius
        radius_km, center_lat, center_lon = select_search_radius(
            coarse_conf, coarse_lat, coarse_lon
        )

        # 4. Fine search (zoom 18) within radius
        fine_lat, fine_lon, fine_conf = weighted_gps_constrained(
            query,
            self.area["embeddings_z18"],
            self.area["gps_z18"],
            temperature=0.03,
            k=10,
            max_distance_km=2.0,
        )

        return {
            "lat": fine_lat,
            "lon": fine_lon,
            "confidence": fine_conf,
        }
```

---

## 7. Storage Requirements

### 7.1 Per-Area Index

| Component | Tiles | Embeddings (float16) | GPS Metadata | Total |
|-----------|-------|---------------------|--------------|-------|
| Zoom 17 | ~1,100 | 0.56MB | 0.02MB | 0.58MB |
| Zoom 18 | ~7,000 | 3.5MB | 0.1MB | 3.6MB |
| **Total per area** | **~8,100** | **4.06MB** | **0.12MB** | **~4.2MB** |

### 7.2 Model Storage

| Item | Size |
|------|------|
| MobileNetV3-Large (TFLite, int8) | ~5.5MB |
| Per area index | ~4.2MB |
| **Total (5 areas + model)** | **~27MB** |

### 7.3 Training Data (Colab)

| Item | Size |
|------|------|
| CV-Cities (4 cities) | ~10.2GB downloaded |
| Extracted/processed | ~8GB |
| Training checkpoints | ~2GB |
| **Total Colab storage needed** | **~20GB** (fits in Drive free tier) |

---

## 8. File Structure

```
visual_odometry/
├── PLAN.md                          # This file
├── requirements.txt                 # Python dependencies
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── models.py                # ResNet-50 + MobileNetV3 encoder wrappers
│   │   ├── losses.py                # Symmetric InfoNCE loss
│   │   ├── dataset.py               # CV-Cities loader + panorama crop extraction
│   │   ├── augmentation.py          # Phone + satellite augmentation pipelines
│   │   ├── train.py                 # Phase 1: teacher training loop
│   │   ├── distill.py               # Phase 2: ResNet-50 → MobileNetV3 distillation
│   │   └── export.py                # Phase 3: TFLite export with int8 quantization
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── tile_downloader.py       # Download Esri satellite tiles by GPS area
│   │   ├── precompute_index.py      # Run satellite encoder → embeddings
│   │   └── package_area.py          # Bundle embeddings + GPS → .npz for phone
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── geolocator.py            # Hierarchical search + adaptive radius
│   │   ├── weighted_average.py      # Top-K weighted GPS + confidence + spatial constraint
│   │   └── index_manager.py         # Load/save area indices
│   └── mobile/
│       ├── (phone_encoder.tflite)   # Generated by export.py
│       └── areas/                   # Generated by package_area.py
│           ├── seattle.npz
│           ├── london.npz
│           └── ...
├── notebooks/
│   └── train_colab.ipynb            # Colab training notebook
└── tests/
    ├── test_losses.py
    ├── test_weighted_average.py
    └── test_geolocator.py
```

---

## 9. Dependencies

```txt
# Training (Colab)
torch>=2.0
torchvision>=0.15
datasets>=2.14          # HuggingFace datasets for CV-Cities
Pillow>=9.0
numpy>=1.24

# Deployment (desktop)
mercantile>=1.3         # GPS ↔ tile coordinate conversion
requests>=2.28          # HTTP tile downloads
numpy>=1.24
Pillow>=9.0

# Mobile export
tensorflow>=2.12        # TFLite conversion
tflite-runtime>=2.12    # On-device inference
tf2onnx>=1.14           # PyTorch → ONNX → TFLite bridge

# Inference (phone / numpy-only)
numpy>=1.24             # Cosine similarity + softmax

# Development
pytest>=7.0
matplotlib>=3.7         # Visualization for notebooks
tqdm>=4.65              # Progress bars
```

---

## 10. Implementation Order

### Step 1: `src/training/losses.py`
Symmetric InfoNCE loss. Pure PyTorch, no dependencies beyond torch.
Testable standalone with random tensors.

### Step 2: `src/training/models.py`
ResNet-50 and MobileNetV3 encoder wrappers.
Each returns L2-normalized 256-dim embeddings.
Include `encode_batch()` for efficient inference.

### Step 3: `src/training/augmentation.py`
Phone and satellite augmentation pipelines as defined in Section 3.4.

### Step 4: `src/training/dataset.py`
CV-Cities dataset loader.
- Download city zips from HuggingFace
- Extract perspective crops from equirectangular panoramas
- Return (satellite_image, phone_crop, gps_coords) triples
- Train/val split (80/20 per city)

### Step 5: `src/training/train.py`
Full training loop:
- DataLoader with augmentation
- AMP mixed precision
- Cosine LR schedule with warmup
- Validation with recall@1/5/10
- Checkpoint saving (best + periodic)
- Wandb or tensorboard logging (optional)

### Step 6: `src/training/distill.py`
Knowledge distillation:
- Load trained ResNet-50 teacher (frozen)
- Train MobileNetV3 student with MSE loss on embeddings
- Same training loop structure as train.py

### Step 7: `src/training/export.py`
Export pipeline:
- PyTorch → ONNX → TFLite
- Int8 quantization for mobile
- Validate TFLite model produces same embeddings

### Step 8: `notebooks/train_colab.ipynb`
Colab notebook tying steps 1-7 together.
- Mount Google Drive
- Install dependencies
- Download CV-Cities
- Train teacher → distill → export
- Save to Drive

### Step 9: `src/deployment/tile_downloader.py`
Download Esri satellite tiles by GPS area.
- Input: center lat/lon, radius_km, zoom levels
- Output: directory of tile images + metadata

### Step 10: `src/deployment/precompute_index.py`
Run satellite encoder on downloaded tiles.
- Input: directory of tiles + trained satellite encoder
- Output: embeddings + GPS metadata

### Step 11: `src/deployment/package_area.py`
Bundle into phone-ready .npz file.
- Input: embeddings + GPS metadata
- Output: compressed .npz with all data

### Step 12: `src/inference/weighted_average.py`
Top-K weighted GPS estimation + confidence + spatial constraint.
As defined in Section 6.2.

### Step 13: `src/inference/geolocator.py`
Full hierarchical search with adaptive radius.
As defined in Section 6.4.

### Step 14: `src/inference/index_manager.py`
Load/save area indices from phone storage.

### Step 15: `requirements.txt`
Consolidated dependencies.

### Step 16: Tests
- `tests/test_losses.py` — verify loss decreases with correct pairings
- `tests/test_weighted_average.py` — verify GPS estimation accuracy
- `tests/test_geolocator.py` — end-to-end test with synthetic data

---

## 11. Known Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CV-Cities panoramas too different from phone photos | Model doesn't generalize | Extract perspective crops at multiple headings; augment heavily |
| Colab session limits (12hr) interrupt training | Incomplete training | Save checkpoints frequently; resume from checkpoint |
| TFLite int8 quantization degrades accuracy | Lower mobile accuracy | Test fp32 vs int8; consider fp16 if int8 too aggressive |
| Few negatives in batch 256 | Weak contrastive signal | Gradient accumulation to simulate larger batches; or use VIGOR hard negatives |
| Adaptive radius too narrow → miss true location | Wrong GPS estimate | Always include fallback to full-area search when confidence is low |
| Esri tile server rate limits | Deployment slow | Add retry logic + caching; tiles don't change often |

---

## 12. Future Improvements (Post-MVP)

1. **Hard negative mining**: Use nearest-neighbor embeddings from wrong cities as
   hard negatives in the contrastive loss
2. **Multi-scale training**: Train on zoom 17, 18, 19 simultaneously with
   multi-scale contrastive loss
3. **GPS-aware augmentation**: During training, add GPS jitter to simulate
   imperfect tile alignment
4. **Orientation estimation**: Add heading prediction to the model so the user
   knows which direction they're facing
5. **Online learning**: Allow the model to improve from user feedback in the field
6. **Larger areas**: Support hierarchical tiling for areas larger than 10x10km
   (city-scale or country-scale)
7. **iOS support**: Export to Core ML in addition to TFLite
