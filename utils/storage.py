"""
Part 3 (step 2) — Persist change-feature GeoDataFrames to a GeoPackage.
All four methods write to the same .gpkg file in separate layers so the
results are comparable in QGIS or any GIS tool.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

GPKG_PATH = Path(__file__).parents[1] / "outputs" / "change_features.gpkg"


def save_to_geopackage(gdf: gpd.GeoDataFrame, layer_name: str,
                       gpkg_path: Path = GPKG_PATH) -> None:
    """
    Append (or replace) a layer inside the shared GeoPackage.

    Parameters
    ----------
    gdf : GeoDataFrame
        Features to store.
    layer_name : str
        Layer name inside the .gpkg (e.g. 'method_01_index').
    gpkg_path : Path
        Destination GeoPackage file.
    """
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        print(f"  [storage] {layer_name}: no features to save (empty result).")
        return

    # Re-project to WGS84 for maximum compatibility
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
    print(f"  [storage] Saved {len(gdf)} features → {gpkg_path.name} [{layer_name}]")


def load_from_geopackage(layer_name: str,
                         gpkg_path: Path = GPKG_PATH) -> gpd.GeoDataFrame:
    """Read a layer back from the GeoPackage."""
    return gpd.read_file(gpkg_path, layer=layer_name)


def list_layers(gpkg_path: Path = GPKG_PATH):
    """Print all layers stored in the GeoPackage."""
    import fiona
    if not gpkg_path.exists():
        print("GeoPackage not found.")
        return
    layers = fiona.listlayers(str(gpkg_path))
    print(f"Layers in {gpkg_path.name}:")
    for lyr in layers:
        gdf = gpd.read_file(gpkg_path, layer=lyr)
        print(f"  {lyr}: {len(gdf)} features")
