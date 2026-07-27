import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mercantile
import requests
from PIL import Image
from io import BytesIO


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> mercantile.Tile:
    """Convert lat/lon to tile coordinates."""
    return mercantile.tile(lon, lat, zoom)


def tile_to_lat_lon(tile: mercantile.Tile) -> tuple[float, float]:
    """Convert tile coordinates to center lat/lon."""
    bounds = mercantile.bounds(tile)
    center_lat = (bounds.south + bounds.north) / 2
    center_lon = (bounds.west + bounds.east) / 2
    return center_lat, center_lon


def tile_size_meters(zoom: int, latitude: float) -> float:
    """Width/height of a 256px tile in meters at given latitude."""
    equatorial_circumference = 40_075_016.686
    return equatorial_circumference * math.cos(math.radians(latitude)) / (2**zoom)


def get_tiles_in_radius(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    zoom: int,
) -> list[mercantile.Tile]:
    """
    Get all tile coordinates covering a circular area.

    Args:
        center_lat: center latitude in degrees
        center_lon: center longitude in degrees
        radius_km: radius in kilometers
        zoom: zoom level

    Returns:
        List of mercantile.Tile objects
    """
    # Approximate degrees per km at this latitude
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))

    delta_lat = radius_km / km_per_deg_lat
    delta_lon = radius_km / km_per_deg_lon

    # Get bounding box tiles
    sw_tile = mercantile.tile(center_lon - delta_lon, center_lat - delta_lat, zoom)
    ne_tile = mercantile.tile(center_lon + delta_lon, center_lat + delta_lat, zoom)

    tiles = []
    for x in range(sw_tile.x, ne_tile.x + 1):
        for y in range(ne_tile.y, sw_tile.y + 1):
            tiles.append(mercantile.Tile(x=x, y=y, z=zoom))

    return tiles


def download_tile_image(
    tile: mercantile.Tile,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Image.Image | None:
    """
    Download a single satellite tile from Esri World Imagery.

    Args:
        tile: mercantile.Tile with x, y, z
        max_retries: number of retry attempts
        retry_delay: seconds between retries

    Returns:
        PIL Image or None if download fails
    """
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{tile.z}/{tile.y}/{tile.x}"
    )

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except (requests.RequestException, IOError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"Failed to download tile {tile}: {e}")
                return None


def download_area_tiles(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    zooms: list[int],
    output_dir: str,
    max_workers: int = 8,
) -> dict[int, list[dict]]:
    """
    Download all satellite tiles for an area at multiple zoom levels.

    Args:
        center_lat: center latitude
        center_lon: center longitude
        radius_km: radius in kilometers
        zooms: list of zoom levels (e.g., [17, 18])
        output_dir: directory to save tile images
        max_workers: concurrent download threads

    Returns:
        Dict mapping zoom level to list of tile info dicts:
        {"zoom": int, "tile": Tile, "path": str, "lat": float, "lon": float}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for zoom in zooms:
        zoom_dir = output_path / f"zoom_{zoom}"
        zoom_dir.mkdir(exist_ok=True)

        tiles = get_tiles_in_radius(center_lat, center_lon, radius_km, zoom)
        print(f"Downloading {len(tiles)} tiles at zoom {zoom}...")

        tile_info = []

        def download_one(tile):
            img = download_tile_image(tile)
            if img:
                path = str(zoom_dir / f"{tile.x}_{tile.y}.jpg")
                img.save(path, quality=95)
                lat, lon = tile_to_lat_lon(tile)
                return {
                    "zoom": zoom,
                    "tile": tile,
                    "path": path,
                    "lat": lat,
                    "lon": lon,
                }
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, t): t for t in tiles}
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                if result:
                    tile_info.append(result)
                if done % 100 == 0:
                    print(f"  {done}/{len(tiles)} tiles downloaded")

        results[zoom] = tile_info
        print(f"  Zoom {zoom}: {len(tile_info)}/{len(tiles)} tiles downloaded")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download satellite tiles")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius in km")
    parser.add_argument("--zooms", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--output", type=str, default="./tiles")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    download_area_tiles(
        center_lat=args.lat,
        center_lon=args.lon,
        radius_km=args.radius,
        zooms=args.zooms,
        output_dir=args.output,
        max_workers=args.workers,
    )
