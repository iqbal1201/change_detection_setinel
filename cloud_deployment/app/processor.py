"""
Background job processor.
Runs change-detection methods and stores base64 figures + folium HTML
into the shared `jobs` dict so the FastAPI routes can serve results.
"""

from __future__ import annotations

import base64
import io
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root so utils/ and methods/ are importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Method metadata ────────────────────────────────────────────────────────────

METHOD_META: dict[str, dict] = {
    "1":  {
        "name":  "Method 1 — Multi-Index PCA Fusion",
        "layer": "method_01_index",
        "gpu":   False,
        "desc":  "Spectral indices (NDVI, NDWI, NDBI) fused with PCA for robust change detection.",
    },
    "2":  {
        "name":  "Method 2 — IR-MAD (Statistical)",
        "layer": "method_02_irmad",
        "gpu":   False,
        "desc":  "Iteratively Reweighted Multivariate Alteration Detection — a classical statistical approach.",
    },
    "3":  {
        "name":  "Method 3 — GMM-EM + Isolation Forest",
        "layer": "method_03_ml",
        "gpu":   False,
        "desc":  "Gaussian Mixture Models with EM learning, refined by Isolation Forest anomaly scoring.",
    },
    "4a": {
        "name":  "Method 4a — DINOv2 Zero-Shot",
        "layer": "method_04a_dinov2",
        "gpu":   True,
        "desc":  "Foundation model feature distance using DINOv2 (requires PyTorch + GPU).",
    },
    "4b": {
        "name":  "Method 4b — SAM2 / AnyChange",
        "layer": "method_04b_sam2",
        "gpu":   True,
        "desc":  "Segment Anything Model v2 structural comparison (requires PyTorch + GPU).",
    },
    "5":  {
        "name":  "Method 5 — Deep CVA (DCVA)",
        "layer": "method_05_dcva",
        "gpu":   True,
        "desc":  "Deep Change Vector Analysis using learned CNN features (requires PyTorch + GPU).",
    },
    "6":  {
        "name":  "Method 6 — VLM Semantic Description",
        "layer": "method_06_vlm",
        "gpu":   False,
        "desc":  "Vision-Language Model interprets the change mask and generates a structured text description.",
    },
}

DEFAULT_VLM_PROMPT = (
    "You are analyzing a composite satellite image showing a land-surface change "
    "detection result. The image has three panels side by side:\n"
    "  LEFT   — the scene BEFORE change (T1, multispectral visible bands)\n"
    "  MIDDLE — the scene AFTER change  (T2, same sensor)\n"
    "  RIGHT  — the CHANGE MASK overlaid in red on T2 (red = detected change)\n\n"
    "Please provide a structured analysis with these five sections:\n"
    "1. CHANGE TYPE: What kind of surface change is visible?\n"
    "2. SPATIAL EXTENT: How much area is affected? Concentrated or dispersed?\n"
    "3. CHANGE PATTERN: Shape and arrangement of changed areas.\n"
    "4. LIKELY CAUSE: Most plausible land-use activity or event.\n"
    "5. CONFIDENCE: Your confidence level and supporting visual evidence.\n\n"
    "State uncertainty explicitly — do not hallucinate NIR or SWIR information."
)


# ── Image loading ──────────────────────────────────────────────────────────────

def _load_bands_from_dir(band_dir: Path):
    """Load B02/B03/B04 TIFFs → (H, W, 3) float32 + rasterio profile."""
    import rasterio
    bands, profile = [], None
    for bname in ["B02.tif", "B03.tif", "B04.tif"]:
        candidates = list(band_dir.glob("*.tif")) + list(band_dir.glob("*.TIF"))
        match = next(
            (f for f in candidates if f.stem.upper() == bname.replace(".tif", "").upper()),
            None,
        )
        if match is None:
            raise FileNotFoundError(
                f"Band file '{bname}' not found in {band_dir}. "
                f"Available: {[f.name for f in candidates]}"
            )
        with rasterio.open(match) as src:
            arr = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile.copy()
        bands.append(arr)
    return np.stack(bands, axis=-1), profile


def _extract_and_load(zip_path: Path, out_dir: Path):
    """Extract ZIP and locate / load the band TIFFs."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    search_dirs = [out_dir] + [p for p in out_dir.rglob("*") if p.is_dir()]
    for d in search_dirs:
        tifs = list(d.glob("*.tif")) + list(d.glob("*.TIF"))
        if any("B02" in f.name.upper() for f in tifs):
            return _load_bands_from_dir(d)

    raise FileNotFoundError(
        f"Could not find B02.tif inside the uploaded ZIP. "
        f"Contents: {[str(p.relative_to(out_dir)) for p in out_dir.rglob('*')][:20]}"
    )


# ── Visualization helpers ──────────────────────────────────────────────────────

def _to_rgb(img: np.ndarray) -> np.ndarray:
    rgb = img[:, :, [2, 1, 0]].copy().astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_change_figure(img1, img2, prob, binary, title: str) -> str:
    """Return a 4-panel change-detection figure as a base64 PNG string."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#e6edf3")

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].imshow(_to_rgb(img1))
    axes[0].set_title("Before (T1)", color="#8b949e", fontsize=10)

    axes[1].imshow(_to_rgb(img2))
    axes[1].set_title("After (T2)", color="#8b949e", fontsize=10)

    im = axes[2].imshow(prob, cmap="hot_r", vmin=0, vmax=1)
    axes[2].set_title("Change Intensity", color="#8b949e", fontsize=10)
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

    axes[3].imshow(_to_rgb(img2), alpha=0.6)
    axes[3].imshow(
        np.ma.masked_where(binary == 0, binary),
        cmap="Reds", vmin=0, vmax=1, alpha=0.75,
    )
    axes[3].set_title("Binary Change", color="#8b949e", fontsize=10)

    plt.tight_layout()
    return _fig_to_b64(fig)


def _make_folium_html(gdf, method_layer: str) -> Optional[str]:
    """Build a folium map for detected polygons and return as HTML string."""
    import folium

    if gdf is None or gdf.empty:
        return None

    gdf_wgs = (
        gdf.to_crs("EPSG:4326")
        if gdf.crs and not gdf.crs.is_geographic
        else gdf
    )
    bounds = gdf_wgs.total_bounds
    centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")
    folium.TileLayer(
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
    ).add_to(m)

    color_map = {
        "method_01_index":   "#e74c3c",
        "method_02_irmad":   "#2ecc71",
        "method_03_ml":      "#3498db",
        "method_04a_dinov2": "#9b59b6",
        "method_04b_sam2":   "#f39c12",
        "method_05_dcva":    "#1abc9c",
    }
    color = color_map.get(method_layer, "#e74c3c")

    for _, row in gdf_wgs.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "color": c, "weight": 1.5,
                "fillColor": c, "fillOpacity": 0.4,
            },
            tooltip=folium.Tooltip(
                f"Conf: {row.get('confidence', 0):.3f} | "
                f"Area: {row.get('area_m2', 0):,.0f} m²"
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


# ── Main job runner ────────────────────────────────────────────────────────────

def run_job(
    job_id: str,
    d1_zip: Path,
    d2_zip: Path,
    method: str,
    prompt: Optional[str],
    vlm_mode: str,
    normalize: bool,
    tmp_dir: Path,
    jobs: dict,
) -> None:
    """Execute the selected change-detection method as a background task."""

    def update(msg: str):
        jobs[job_id]["progress"] = msg

    try:
        # ── Load ──────────────────────────────────────────────────────────────
        update("Loading and extracting images…")
        d1_dir, d2_dir = tmp_dir / "date1", tmp_dir / "date2"
        d1_dir.mkdir(); d2_dir.mkdir()

        img1, profile = _extract_and_load(d1_zip, d1_dir)
        img2, _       = _extract_and_load(d2_zip, d2_dir)

        # ── Normalize ─────────────────────────────────────────────────────────
        if normalize:
            update("Applying radiometric normalisation…")
            from utils.normalizer import local_normalize
            img1, img2 = local_normalize(img1, img2)

        meta = METHOD_META[method]
        name = meta["name"]

        # ── Method 6 — VLM ───────────────────────────────────────────────────
        if method == "6":
            update("Running Method 1 to generate base change mask…")
            from methods import method_01_index
            prob, binary = method_01_index.run(img1, img2)

            update("Calling VLM for semantic description…")
            import methods.method_06_vlm as vlm_mod
            vlm_mod._PROMPT = (prompt.strip() if prompt and prompt.strip()
                               else DEFAULT_VLM_PROMPT)
            vlm_result = vlm_mod.describe(
                img1, img2, binary,
                source_method="method_01_index",
                mode=vlm_mode,
            )

            update("Generating visualization…")
            figure_b64 = _make_change_figure(
                img1, img2, prob, binary,
                "VLM — Base Change Mask (Method 1 as input)",
            )

            jobs[job_id].update({
                "status":      "done",
                "progress":    "Complete",
                "method_name": name,
                "figure_b64":  figure_b64,
                "folium_html": None,
                "stats":       {},
                "description": vlm_result.get("description", ""),
                "backend":     vlm_result.get("backend", "unknown"),
            })
            return

        # ── Methods 1–5 ───────────────────────────────────────────────────────
        update(f"Running {name}…")

        if method == "1":
            from methods import method_01_index
            prob, binary = method_01_index.run(img1, img2)
        elif method == "2":
            from methods import method_02_irmad
            prob, binary = method_02_irmad.run(img1, img2)
        elif method == "3":
            from methods import method_03_ml
            prob, binary = method_03_ml.run(img1, img2)
        elif method == "4a":
            from methods.method_04_gfm import run_dinov2
            prob, binary = run_dinov2(img1, img2)
        elif method == "4b":
            from methods.method_04_gfm import run_sam2
            prob, binary = run_sam2(img1, img2)
        elif method == "5":
            from methods.method_05_dcva import run
            prob, binary = run(img1, img2)

        # ── Vectorize ─────────────────────────────────────────────────────────
        update("Vectorizing detected changes…")
        from utils.vectorizer import raster_to_polygons
        gdf = raster_to_polygons(binary, prob, profile, method_name=meta["layer"])

        total_area = float(gdf["area_m2"].sum())    if not gdf.empty else 0.0
        mean_conf  = float(gdf["confidence"].mean()) if not gdf.empty else 0.0
        n_poly     = len(gdf)

        # ── Visualize ─────────────────────────────────────────────────────────
        update("Generating visualizations…")
        figure_b64  = _make_change_figure(img1, img2, prob, binary, name)
        folium_html = _make_folium_html(gdf, meta["layer"])

        jobs[job_id].update({
            "status":      "done",
            "progress":    "Complete",
            "method_name": name,
            "figure_b64":  figure_b64,
            "folium_html": folium_html,
            "stats": {
                "n_polygons":      n_poly,
                "area_km2":        round(total_area / 1e6, 4),
                "mean_confidence": round(mean_conf, 3),
            },
            "description": None,
            "backend":     None,
        })

    except Exception as exc:
        jobs[job_id].update({
            "status":    "error",
            "progress":  "Failed",
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
