"""
Part 4 — Visualization.
Produces:
  1. matplotlib multi-panel figure (static PNG) per method.
  2. A single combined folium interactive HTML map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import folium
from folium.plugins import MeasureControl, MiniMap

VIZ_DIR = Path(__file__).parents[1] / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "method_01_index":      "#e74c3c",
    "method_02_irmad":      "#2ecc71",
    "method_03_ml":         "#3498db",
    "method_04a_dinov2":    "#9b59b6",
    "method_04b_sam2":      "#f39c12",
    "method_05_dcva":       "#1abc9c",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_rgb(img: np.ndarray, percentile: int = 2) -> np.ndarray:
    """Stretch (H,W,3) float image to uint8 for display (band order B,G,R → R,G,B)."""
    rgb = img[:, :, [2, 1, 0]].copy()          # B02,B03,B04 → R,G,B display
    lo = np.percentile(rgb, percentile)
    hi = np.percentile(rgb, 100 - percentile)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


# ── static matplotlib figure ─────────────────────────────────────────────────

def plot_method_result(
    img1: np.ndarray,
    img2: np.ndarray,
    change_prob: np.ndarray,
    change_binary: np.ndarray,
    method_name: str,
    gdf: Optional[gpd.GeoDataFrame] = None,
    profile: Optional[Dict] = None,
) -> Path:
    """
    Save a 4-panel figure: [Before | After | Change Intensity | Binary + Polygons].
    Returns the saved figure path.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(f"Change Detection — {method_name}", fontsize=14, fontweight="bold")

    rgb1 = _to_rgb(img1)
    rgb2 = _to_rgb(img2)

    axes[0].imshow(rgb1)
    axes[0].set_title("Before (2023-08-12)")
    axes[0].axis("off")

    axes[1].imshow(rgb2)
    axes[1].set_title("After (2023-09-02)")
    axes[1].axis("off")

    im = axes[2].imshow(change_prob, cmap="hot_r", vmin=0, vmax=1)
    axes[2].set_title("Change Intensity")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(rgb2, alpha=0.6)
    axes[3].imshow(
        np.ma.masked_where(change_binary == 0, change_binary),
        cmap="Reds", vmin=0, vmax=1, alpha=0.7,
    )
    axes[3].set_title("Binary Change")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = VIZ_DIR / f"{method_name}_result.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved {out_path.name}")
    return out_path


# ── folium interactive map ────────────────────────────────────────────────────

def build_folium_map(
    all_gdfs: Dict[str, gpd.GeoDataFrame],
    aoi_path: Optional[Path] = None,
) -> Path:
    """
    Build a folium map with one layer group per method.

    Parameters
    ----------
    all_gdfs : dict  {method_name: GeoDataFrame (WGS84)}
    aoi_path : Path  optional path to aoi.geojson

    Returns
    -------
    Path to the saved HTML file.
    """
    # Determine map centre from first non-empty GeoDataFrame
    centre = [-12.245, 25.865]  # Zambia AOI fallback
    for gdf in all_gdfs.values():
        if not gdf.empty:
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
            break

    m = folium.Map(location=centre, zoom_start=12,
                   tiles="CartoDB positron", control_scale=True)

    # Satellite basemap option
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
    ).add_to(m)

    # AOI boundary
    if aoi_path and aoi_path.exists():
        aoi_gdf = gpd.read_file(aoi_path)
        folium.GeoJson(
            aoi_gdf.__geo_interface__,
            name="AOI",
            style_function=lambda _: {
                "color": "#ffffff", "weight": 2,
                "fillOpacity": 0, "dashArray": "6 4",
            },
        ).add_to(m)

    # One layer group per method
    for method_name, gdf in all_gdfs.items():
        if gdf.empty:
            continue
        gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs and not gdf.crs.is_geographic else gdf
        color = METHOD_COLORS.get(method_name, "#e74c3c")

        fg = folium.FeatureGroup(name=method_name, show=True)

        for _, row in gdf_wgs.iterrows():
            popup_html = (
                f"<b>{method_name}</b><br>"
                f"Area: {row.get('area_m2', 'N/A'):,.0f} m²<br>"
                f"Confidence: {row.get('confidence', 0):.3f}<br>"
                f"Before: {row.get('date_before', '')}<br>"
                f"After: {row.get('date_after', '')}"
            )
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=color: {
                    "color": c, "weight": 1.5,
                    "fillColor": c, "fillOpacity": 0.35,
                },
                tooltip=folium.Tooltip(f"{method_name} — conf: {row.get('confidence', 0):.3f}"),
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg)

        fg.add_to(m)

    MeasureControl(position="topleft").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = VIZ_DIR / "change_map.html"
    m.save(str(out_path))
    print(f"  [viz] Interactive map saved → {out_path.name}")
    return out_path


def plot_comparison_summary(
    results: Dict[str, Dict],
    out_path: Optional[Path] = None,
) -> Path:
    """
    Bar chart comparing total changed area (m²) and mean confidence across methods.
    `results` = {method_name: {'area_total': float, 'mean_confidence': float}}
    """
    methods = list(results.keys())
    areas = [results[m].get("area_total", 0) / 1e6 for m in methods]  # km²
    confs = [results[m].get("mean_confidence", 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    ax1.bar(methods, areas, color=colors, edgecolor="white")
    ax1.set_title("Total Changed Area (km²)")
    ax1.set_ylabel("km²")
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(methods, confs, color=colors, edgecolor="white")
    ax2.set_title("Mean Confidence Score")
    ax2.set_ylabel("0 – 1")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out_path = out_path or (VIZ_DIR / "comparison_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Summary chart saved → {out_path.name}")
    return out_path
