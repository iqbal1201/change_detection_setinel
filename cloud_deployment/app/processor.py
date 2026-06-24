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

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Method metadata ────────────────────────────────────────────────────────────

METHOD_META: dict[str, dict] = {
    "1":  {
        "name":  "Multi-Index PCA Fusion",
        "layer": "method_01_index",
        "gpu":   False,
        "desc":  "Computes VARI, NGRDI, BI, ExR, ExG indices for both dates, "
                 "fuses absolute differences via PCA, and thresholds with Otsu.",
    },
    "2":  {
        "name":  "IR-MAD",
        "layer": "method_02_irmad",
        "gpu":   False,
        "desc":  "Iteratively Reweighted Multivariate Alteration Detection. "
                 "Radiometrically invariant — robust to atmospheric and sensor gain differences.",
    },
    "3":  {
        "name":  "GMM-EM + Isolation Forest",
        "layer": "method_03_ml",
        "gpu":   False,
        "desc":  "Fits a 3-component Gaussian Mixture on PCA-reduced difference features, "
                 "then refines using Isolation Forest anomaly scores.",
    },
    "4a": {
        "name":  "DINOv2 Zero-Shot",
        "layer": "method_04a_dinov2",
        "gpu":   True,
        "desc":  "Extracts patch tokens from a frozen DINOv2 ViT-B/14 backbone and "
                 "measures cosine distance between dates. Requires PyTorch.",
    },
    "4b": {
        "name":  "SAM2 / AnyChange",
        "layer": "method_04b_sam2",
        "gpu":   True,
        "desc":  "Compares SAM2 automatic mask density maps between dates. "
                 "Structural segment shifts indicate real surface change. Requires PyTorch.",
    },
    "5":  {
        "name":  "Deep CVA (DCVA)",
        "layer": "method_05_dcva",
        "gpu":   True,
        "desc":  "Change Vector Analysis applied to frozen VGG16 conv3_2 feature maps. "
                 "More robust than raw-band CVA. Requires PyTorch.",
    },
    "6":  {
        "name":  "VLM Semantic Description",
        "layer": "method_06_vlm",
        "gpu":   False,
        "desc":  "Uses a Vision-Language Model to produce a structured natural-language "
                 "description of what changed and why. Runs on top of Method 1's binary mask.",
    },
}

DEFAULT_VLM_PROMPT = (
    "You are analyzing a composite satellite image showing a land-surface change "
    "detection result. The image has three panels side by side:\n"
    "  LEFT   — the scene BEFORE change (T1, multispectral visible bands)\n"
    "  MIDDLE — the scene AFTER change  (T2, same sensor)\n"
    "  RIGHT  — the CHANGE MASK overlaid in red on T2 (red = detected change)\n\n"
    "Provide a structured analysis:\n"
    "1. CHANGE TYPE: What kind of surface change is visible?\n"
    "2. SPATIAL EXTENT: How much area is affected? Concentrated or dispersed?\n"
    "3. CHANGE PATTERN: Shape and arrangement of changed areas.\n"
    "4. LIKELY CAUSE: Most plausible land-use activity or event.\n"
    "5. CONFIDENCE: Your confidence level and supporting visual evidence.\n\n"
    "State uncertainty explicitly — do not hallucinate NIR or SWIR information."
)


# ── Image loading ──────────────────────────────────────────────────────────────

def _load_bands_from_dir(band_dir: Path):
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
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor("#0e1524")
    fig.suptitle(title, fontsize=12, fontweight="600", color="#dce8f8",
                 y=1.01, ha="left", x=0.01)

    panel_bg = "#0e1524"
    label_color = "#7b90b2"

    for ax in axes:
        ax.set_facecolor(panel_bg)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].imshow(_to_rgb(img1))
    axes[0].set_title("Before (T1)", color=label_color, fontsize=9, pad=6)

    axes[1].imshow(_to_rgb(img2))
    axes[1].set_title("After (T2)", color=label_color, fontsize=9, pad=6)

    im = axes[2].imshow(prob, cmap="YlOrRd", vmin=0, vmax=1)
    axes[2].set_title("Change Intensity", color=label_color, fontsize=9, pad=6)
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=label_color, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=label_color)
    cb.outline.set_edgecolor("#283a5c")

    axes[3].imshow(_to_rgb(img2), alpha=0.55)
    axes[3].imshow(
        np.ma.masked_where(binary == 0, binary),
        cmap="Reds", vmin=0, vmax=1, alpha=0.8,
    )
    axes[3].set_title("Detected Change", color=label_color, fontsize=9, pad=6)

    plt.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def _make_folium_html(gdf, method_layer: str) -> Optional[str]:
    import folium
    from folium.plugins import MeasureControl, MiniMap
    from branca.element import Element

    if gdf is None or gdf.empty:
        return None

    gdf_wgs = (
        gdf.to_crs("EPSG:4326")
        if gdf.crs and not gdf.crs.is_geographic
        else gdf
    )
    bounds = gdf_wgs.total_bounds
    centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Stats for legend
    n_poly     = len(gdf_wgs)
    total_area = float(gdf_wgs["area_m2"].sum()) if "area_m2" in gdf_wgs.columns else 0.0
    area_ha    = total_area / 1e4
    conf_col   = gdf_wgs["confidence"].dropna() if "confidence" in gdf_wgs.columns else []
    conf_min   = float(conf_col.min()) if len(conf_col) > 0 else 0.0
    conf_max   = float(conf_col.max()) if len(conf_col) > 0 else 1.0

    method_display = {
        "method_01_index":   "Multi-Index PCA Fusion",
        "method_02_irmad":   "IR-MAD",
        "method_03_ml":      "GMM-EM + Isolation Forest",
        "method_04a_dinov2": "DINOv2 Zero-Shot",
        "method_04b_sam2":   "SAM2 / AnyChange",
        "method_05_dcva":    "Deep CVA",
    }.get(method_layer, method_layer)

    color_map = {
        "method_01_index":   "#e74c3c",
        "method_02_irmad":   "#2ecc71",
        "method_03_ml":      "#3498db",
        "method_04a_dinov2": "#9b59b6",
        "method_04b_sam2":   "#f39c12",
        "method_05_dcva":    "#1abc9c",
    }
    color = color_map.get(method_layer, "#e74c3c")

    m = folium.Map(
        location=centre,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer(
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
    ).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    fg = folium.FeatureGroup(name=f"Changed areas ({n_poly} polygons)")

    for _, row in gdf_wgs.iterrows():
        conf    = float(row.get("confidence", 0.5))
        area_m2 = float(row.get("area_m2", 0))

        popup_html = (
            f'<div style="font-family:-apple-system,\'Segoe UI\',sans-serif;'
            f'min-width:190px;font-size:13px;">'
            f'<div style="font-weight:700;margin-bottom:8px;padding-bottom:6px;'
            f'border-bottom:1px solid #e2e8f0;color:#1a202c;">Change Region</div>'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<tr><td style="color:#718096;padding:3px 0;font-size:12px;">Area</td>'
            f'<td style="text-align:right;font-weight:600;color:#2d3748;">'
            f'{area_m2:,.0f}&nbsp;m²<br>'
            f'<span style="font-weight:400;color:#718096;font-size:11px;">'
            f'{area_m2/1e4:.3f}&nbsp;ha</span></td></tr>'
            f'<tr><td style="color:#718096;padding:3px 0;font-size:12px;">Confidence</td>'
            f'<td style="text-align:right;font-weight:600;color:#2d3748;">{conf:.3f}</td>'
            f'</tr></table>'
            f'<div style="margin-top:8px;height:3px;background:#e2e8f0;'
            f'border-radius:2px;overflow:hidden;">'
            f'<div style="height:100%;width:{conf*100:.0f}%;background:{color};'
            f'border-radius:2px;"></div></div></div>'
        )

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color, cf=conf: {
                "color":       c,
                "weight":      1.5,
                "fillColor":   c,
                "fillOpacity": max(0.15, cf * 0.65),
            },
            tooltip=folium.Tooltip(
                f"{area_m2:,.0f} m² &nbsp;|&nbsp; conf: {conf:.3f}",
                sticky=False,
            ),
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(fg)

    fg.add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft",
            width=140, height=140, zoom_level_offset=-6).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Cartographic legend
    legend_html = (
        f'<div id="cd-legend" style="'
        f'position:fixed;bottom:30px;right:10px;z-index:1000;'
        f'background:rgba(10,16,30,0.93);backdrop-filter:blur(8px);'
        f'border:1px solid rgba(255,255,255,0.1);border-radius:8px;'
        f'padding:16px 18px;font-family:-apple-system,\'Segoe UI\',sans-serif;'
        f'color:#dce8f8;font-size:12px;min-width:205px;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.5);pointer-events:none;">'
        f'<div style="font-weight:700;font-size:13px;margin-bottom:12px;'
        f'color:#f0f6ff;letter-spacing:0.02em;">Change Detection</div>'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'<div style="width:14px;height:14px;background:{color};'
        f'border-radius:2px;flex-shrink:0;opacity:0.85;"></div>'
        f'<span>{method_display}</span></div>'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
        f'<div style="width:48px;height:4px;'
        f'background:linear-gradient(to right,{color}28,{color});'
        f'border-radius:2px;flex-shrink:0;"></div>'
        f'<span style="color:#7b90b2;font-size:11px;">opacity = confidence</span></div>'
        f'<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:10px 0;">'
        f'<table style="width:100%;border-collapse:collapse;font-size:11px;">'
        f'<tr><td style="color:#7b90b2;padding:2px 0;">Polygons</td>'
        f'<td style="text-align:right;color:#dce8f8;font-weight:600;">{n_poly}</td></tr>'
        f'<tr><td style="color:#7b90b2;padding:2px 0;">Total area</td>'
        f'<td style="text-align:right;color:#dce8f8;font-weight:600;">{area_ha:.2f} ha</td></tr>'
        f'<tr><td style="color:#7b90b2;padding:2px 0;">Confidence</td>'
        f'<td style="text-align:right;color:#dce8f8;font-weight:600;">'
        f'{conf_min:.2f} – {conf_max:.2f}</td></tr>'
        f'</table>'
        f'<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:10px 0;">'
        f'<div style="font-size:10px;color:#4a6080;">Click polygon for details</div>'
        f'</div>'
    )
    m.get_root().html.add_child(Element(legend_html))

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

    def update(msg: str):
        jobs[job_id]["progress"] = msg

    try:
        update("Loading and extracting images…")
        d1_dir, d2_dir = tmp_dir / "date1", tmp_dir / "date2"
        d1_dir.mkdir(); d2_dir.mkdir()

        img1, profile = _extract_and_load(d1_zip, d1_dir)
        img2, _       = _extract_and_load(d2_zip, d2_dir)

        if normalize:
            update("Applying radiometric normalisation…")
            from utils.normalizer import local_normalize
            img1, img2 = local_normalize(img1, img2)

        meta = METHOD_META[method]
        name = meta["name"]

        # ── GPU availability check ────────────────────────────────────────────
        if meta.get("gpu", False):
            try:
                import torch  # noqa: F401
            except ModuleNotFoundError:
                raise RuntimeError(
                    f"'{name}' requires PyTorch, which is not installed in this "
                    "environment. Please select a CPU method: "
                    "Multi-Index PCA, IR-MAD, or GMM-EM."
                )

        # ── Method 6 — VLM ───────────────────────────────────────────────────
        if method == "6":
            update("Generating base change mask (Method 1)…")
            from methods import method_01_index
            prob, binary = method_01_index.run(img1, img2)

            update("Running VLM semantic description…")
            import methods.method_06_vlm as vlm_mod
            vlm_mod._PROMPT = (
                prompt.strip() if prompt and prompt.strip()
                else DEFAULT_VLM_PROMPT
            )
            vlm_result = vlm_mod.describe(
                img1, img2, binary,
                source_method="method_01_index",
                mode=vlm_mode,
            )

            update("Generating figure…")
            figure_b64 = _make_change_figure(
                img1, img2, prob, binary,
                f"VLM Semantic Description  ·  base mask: Multi-Index PCA",
            )
            jobs[job_id].update({
                "status":      "done",
                "progress":    "Complete",
                "method_name": name,
                "figure_b64":  figure_b64,
                "folium_html": None,
                "stats":       None,
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
            from methods.method_05_dcva import run as run_dcva
            prob, binary = run_dcva(img1, img2)

        update("Vectorizing changes…")
        from utils.vectorizer import raster_to_polygons
        gdf = raster_to_polygons(binary, prob, profile, method_name=meta["layer"])

        total_area = float(gdf["area_m2"].sum())     if not gdf.empty else 0.0
        mean_conf  = float(gdf["confidence"].mean()) if not gdf.empty else 0.0
        n_poly     = len(gdf)

        update("Building visualizations…")
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
