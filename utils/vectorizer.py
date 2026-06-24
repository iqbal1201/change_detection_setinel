"""
Part 3 (step 1) — Convert binary raster change map to vector polygons.
Uses rasterio.features.shapes for contiguous region extraction, then
filters by minimum area and attaches confidence from the continuous map.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_polygons(
    change_binary: np.ndarray,
    change_prob: np.ndarray,
    profile: Dict,
    method_name: str,
    date_before: str = "2023-08-12",
    date_after: str = "2023-09-02",
    min_area_m2: float = 900.0,
) -> gpd.GeoDataFrame:
    """
    Vectorise a binary change raster into a GeoDataFrame of polygons.

    Parameters
    ----------
    change_binary : np.ndarray (H, W), uint8
        Binary change map: 1 = change, 0 = no-change.
    change_prob : np.ndarray (H, W), float32
        Continuous change probability [0, 1].
    profile : dict
        Rasterio profile carrying CRS and transform.
    method_name : str
        Label stored in the 'method' column.
    min_area_m2 : float
        Polygons smaller than this (m²) are discarded (default 1 pixel at 30m = 900m²).

    Returns
    -------
    GeoDataFrame with columns:
        id, method, date_before, date_after, area_m2, confidence, geometry
    """
    transform = profile["transform"]
    crs = profile["crs"]

    mask = (change_binary == 1).astype(np.uint8)
    records: List[Dict[str, Any]] = []

    for geom_dict, value in shapes(mask, mask=mask, transform=transform):
        if value != 1:
            continue
        geom = shape(geom_dict)
        # area in CRS units (metres if projected)
        area_m2 = geom.area

        if area_m2 < min_area_m2:
            continue

        # mean confidence of all pixels inside the polygon
        from rasterio.features import geometry_mask
        poly_mask = geometry_mask(
            [geom_dict],
            out_shape=change_binary.shape,
            transform=transform,
            invert=True,
        )
        confidence = float(change_prob[poly_mask].mean()) if poly_mask.any() else 0.0

        records.append({
            "method": method_name,
            "date_before": date_before,
            "date_after": date_after,
            "area_m2": round(area_m2, 2),
            "confidence": round(confidence, 4),
            "geometry": geom,
        })

    if not records:
        gdf = gpd.GeoDataFrame(
            columns=["method", "date_before", "date_after",
                     "area_m2", "confidence", "geometry"],
            geometry="geometry",
            crs=crs,
        )
        return gdf

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    gdf.insert(0, "id", range(1, len(gdf) + 1))
    return gdf


def save_raster(
    arr: np.ndarray,
    profile: Dict,
    out_path: Path,
    dtype: str = "float32",
    nodata: float = -9999.0,
) -> None:
    """Write a (H, W) array to a single-band GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, compress="lzw", nodata=nodata)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr.astype(dtype), 1)
