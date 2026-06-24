"""
Solafune Change Detection Pipeline
===================================
Runs all parts of the assignment end-to-end:

  Part 1 — Data loading & stacking          (utils/data_loader.py)
  Part 2 — Change detection (6 methods):
            Method 1:  Multi-Index PCA Fusion (methods/method_01_index.py)
            Method 2:  IR-MAD statistical     (methods/method_02_irmad.py)
            Method 3:  GMM-EM + Iso. Forest   (methods/method_03_ml.py)
            Method 4a: DINOv2 zero-shot       (methods/method_04_gfm.py)
            Method 4b: SAM2 / AnyChange       (methods/method_04_gfm.py)
            Method 5:  Deep CVA (DCVA)        (methods/method_05_dcva.py)
  Part 3 — Vectorisation & GeoPackage storage
  Part 4 — Visualisation (matplotlib + folium)
  Part 5 — VLM semantic description         (methods/method_06_vlm.py)

Usage
-----
  python pipeline.py                  # run all methods
  python pipeline.py --skip-gfm       # skip DINOv2 / SAM2 / DCVA (no PyTorch)
  python pipeline.py --skip-vlm       # skip VLM description step
  python pipeline.py --method 1       # run only method 1
  python pipeline.py --vlm-mode rule_based   # force rule-based VLM fallback
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ── project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── utils ─────────────────────────────────────────────────────────────────────
from utils.data_loader import run_part1
from utils.normalizer import local_normalize
from utils.vectorizer import raster_to_polygons, save_raster
from utils.storage import save_to_geopackage, list_layers
from utils.visualizer import (
    plot_method_result,
    build_folium_map,
    plot_comparison_summary,
)

# ── methods ───────────────────────────────────────────────────────────────────
from methods import method_01_index, method_02_irmad, method_03_ml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_method(
    name: str,
    fn,
    img1: np.ndarray,
    img2: np.ndarray,
    profile: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single change-detection method with error isolation."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        prob, binary = fn(img1, img2)
        return prob, binary
    except Exception:
        print(f"  ERROR in {name}:")
        traceback.print_exc()
        h, w = img1.shape[:2]
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.uint8)


def _save_outputs(
    name: str,
    layer_name: str,
    prob: np.ndarray,
    binary: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    profile: Dict,
) -> Dict:
    """Save rasters, vectorise, store in GeoPackage, plot."""
    # Rasters
    save_raster(prob,   profile, PROCESSED_DIR / f"{layer_name}_prob.tif")
    save_raster(binary, profile, PROCESSED_DIR / f"{layer_name}_binary.tif", dtype="uint8", nodata=255)

    # Vectorise
    gdf = raster_to_polygons(binary, prob, profile, method_name=layer_name)

    # GeoPackage
    save_to_geopackage(gdf, layer_name)

    # Matplotlib figure
    plot_method_result(img1, img2, prob, binary, name, gdf=gdf, profile=profile)

    # Summary stats
    total_area = float(gdf["area_m2"].sum()) if not gdf.empty else 0.0
    mean_conf  = float(gdf["confidence"].mean()) if not gdf.empty else 0.0
    n_poly     = len(gdf)
    print(f"  Summary: {n_poly} polygons | {total_area/1e6:.4f} km² | "
          f"mean confidence {mean_conf:.3f}")

    return {"area_total": total_area, "mean_confidence": mean_conf,
            "n_polygons": n_poly, "gdf": gdf}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(
    run_methods=None,
    skip_gfm=False,
    skip_vlm=False,
    vlm_mode="auto",
    normalize=True,
    norm_window=64,
):
    if run_methods is None:
        run_methods = {1, 2, 3, 4, 5}

    # ── Part 1 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PART 1 — Data Loading & Stacking")
    print("="*60)
    img1, img2, profile = run_part1()

    # ── Radiometric normalisation ─────────────────────────────────────────────
    if normalize:
        print("\n" + "="*60)
        print("  PRE-PROCESSING — Local Radiometric Normalisation")
        print("="*60)
        print(f"  Applying local z-score normalisation (window={norm_window} px)...")
        img1, img2 = local_normalize(img1, img2, window_size=norm_window)
        print("  Done — mosaic seams and brightness offsets suppressed.")
    else:
        print("\n  [Pre-processing] Radiometric normalisation skipped (--no-normalize).")

    results: Dict[str, Dict] = {}
    all_gdfs: Dict[str, object] = {}

    # ── Part 2 + 3 (Methods 1–3) ─────────────────────────────────────────────
    method_map = {
        1: ("Method 1 — Multi-Index PCA Fusion", "method_01_index", method_01_index.run),
        2: ("Method 2 — IR-MAD (Statistical)",   "method_02_irmad",  method_02_irmad.run),
        3: ("Method 3 — GMM-EM + Iso. Forest",   "method_03_ml",     method_03_ml.run),
    }

    for method_id, (name, layer, fn) in method_map.items():
        if method_id not in run_methods:
            continue
        prob, binary = _run_method(name, fn, img1, img2, profile)
        stats = _save_outputs(name, layer, prob, binary, img1, img2, profile)
        results[layer] = stats
        all_gdfs[layer] = stats["gdf"]

    # ── Part 2 + 3 (Methods 4 & 5 — GFM / Deep) ─────────────────────────────
    if not skip_gfm and (run_methods & {4, 5}):
        try:
            import torch  # noqa: F401 — probe only

            # 4a: DINOv2
            if 4 in run_methods:
                from methods.method_04_gfm import run_dinov2, run_sam2

                print(f"\n{'='*60}")
                print("  Method 4a — DINOv2 Zero-Shot Feature Distance")
                print(f"{'='*60}")
                try:
                    prob4a, bin4a = run_dinov2(img1, img2)
                    stats4a = _save_outputs(
                        "Method 4a — DINOv2",
                        "method_04a_dinov2",
                        prob4a, bin4a, img1, img2, profile,
                    )
                    results["method_04a_dinov2"] = stats4a
                    all_gdfs["method_04a_dinov2"] = stats4a["gdf"]
                except Exception:
                    print("  DINOv2 failed:")
                    traceback.print_exc()

                # 4b: SAM2 / AnyChange
                print(f"\n{'='*60}")
                print("  Method 4b — SAM2 / AnyChange Structural Comparison")
                print(f"{'='*60}")
                try:
                    prob4b, bin4b = run_sam2(img1, img2)
                    stats4b = _save_outputs(
                        "Method 4b — SAM2/AnyChange",
                        "method_04b_sam2",
                        prob4b, bin4b, img1, img2, profile,
                    )
                    results["method_04b_sam2"] = stats4b
                    all_gdfs["method_04b_sam2"] = stats4b["gdf"]
                except Exception:
                    print("  SAM2 failed:")
                    traceback.print_exc()

            # 5: Deep CVA (DCVA)
            if 5 in run_methods:
                print(f"\n{'='*60}")
                print("  Method 5 — Deep Change Vector Analysis (DCVA)")
                print(f"{'='*60}")
                try:
                    from methods.method_05_dcva import run as run_dcva
                    prob5, bin5 = run_dcva(img1, img2)
                    stats5 = _save_outputs(
                        "Method 5 — Deep CVA",
                        "method_05_dcva",
                        prob5, bin5, img1, img2, profile,
                    )
                    results["method_05_dcva"] = stats5
                    all_gdfs["method_05_dcva"] = stats5["gdf"]
                except Exception:
                    print("  DCVA failed:")
                    traceback.print_exc()

        except ImportError:
            print("\n  [Methods 4 & 5] Skipped — PyTorch not installed.")
            print("  Install with:")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            print("    pip install transformers")
            print("    pip install git+https://github.com/facebookresearch/sam2.git")

    # ── Part 4 — Folium map ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PART 4 — Building Interactive Folium Map")
    print("="*60)
    aoi_path = ROOT / "aoi.geojson"
    if all_gdfs:
        build_folium_map(all_gdfs, aoi_path=aoi_path)
    else:
        print("  No results to visualise.")

    # Summary comparison chart
    if results:
        summary_data = {
            k: {"area_total": v["area_total"], "mean_confidence": v["mean_confidence"]}
            for k, v in results.items()
        }
        plot_comparison_summary(summary_data)

    # ── Part 5 — VLM Semantic Description ────────────────────────────────────
    if not skip_vlm and results:
        print("\n" + "="*60)
        print("  PART 5 — VLM Semantic Description")
        print("="*60)
        try:
            from methods.method_06_vlm import describe as vlm_describe

            # Pick the binary mask from the method with the most polygons
            best_layer = max(results, key=lambda k: results[k]["n_polygons"])
            best_binary_path = PROCESSED_DIR / f"{best_layer}_binary.tif"

            try:
                import rasterio
                with rasterio.open(best_binary_path) as src:
                    best_binary = src.read(1)
            except Exception:
                best_binary = np.zeros(img1.shape[:2], dtype=np.uint8)

            print(f"  [VLM] Using change mask from: {best_layer}")
            vlm_result = vlm_describe(
                img1, img2, best_binary,
                source_method=best_layer,
                mode=vlm_mode,
            )
            print("\n  [VLM] Description preview:")
            for line in vlm_result["description"][:600].splitlines():
                print(f"    {line}")
            if len(vlm_result["description"]) > 600:
                print("    ... (see visualizations/vlm_description.txt for full text)")
        except Exception:
            print("  VLM description failed:")
            traceback.print_exc()

    # ── GeoPackage layer listing (current run only) ───────────────────────────
    print("\n" + "="*60)
    print("  PART 3 — GeoPackage Contents (this run)")
    print("="*60)
    if results:
        for layer_name, stats in results.items():
            print(f"  {layer_name}: {stats['n_polygons']} features, "
                  f"{stats['area_total']/1e6:.4f} km², "
                  f"conf={stats['mean_confidence']:.3f}")
    else:
        print("  No layers written in this run.")
    print("  (To see all historical layers: open outputs/change_features.gpkg in QGIS)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Processed rasters : {PROCESSED_DIR}")
    print(f"  GeoPackage        : outputs/change_features.gpkg")
    print(f"  Interactive map   : visualizations/change_map.html")
    print(f"  Static figures    : visualizations/method_*_result.png")
    print(f"  VLM description   : visualizations/vlm_description.txt")
    print(f"  VLM panel image   : visualizations/vlm_panel.jpg")
    print(f"  Report template   : report.md")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solafune change-detection pipeline."
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip local radiometric normalisation (use raw band values).",
    )
    parser.add_argument(
        "--norm-window", type=int, default=64, metavar="PX",
        help="Sliding-window size for local normalisation in pixels (default: 64).",
    )
    parser.add_argument(
        "--skip-gfm", action="store_true",
        help="Skip DINOv2, SAM2, and DCVA (no PyTorch required).",
    )
    parser.add_argument(
        "--skip-vlm", action="store_true",
        help="Skip the VLM semantic description step.",
    )
    parser.add_argument(
        "--vlm-mode",
        choices=["auto", "anthropic", "openai", "llava", "rule_based"],
        default="auto",
        help="VLM backend to use (default: auto — tries API keys then local model).",
    )
    parser.add_argument(
        "--method", type=int, choices=[1, 2, 3, 4, 5], default=None,
        help="Run only one specific method (1-5).",
    )
    args = parser.parse_args()

    methods = {args.method} if args.method else {1, 2, 3, 4, 5}
    main(
        run_methods=methods,
        skip_gfm=args.skip_gfm,
        skip_vlm=args.skip_vlm,
        vlm_mode=args.vlm_mode,
        normalize=not args.no_normalize,
        norm_window=args.norm_window,
    )
