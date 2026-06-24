"""
test_fixes.py — Verify that all required outputs from the fixed pipeline exist
and are structurally correct.

Usage
-----
  # First generate outputs (run Method 1 only — fastest):
  python pipeline.py --method 1 --skip-vlm

  # Then validate:
  python test_fixes.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global passed, failed
    if condition:
        passed += 1
        print(f"  {PASS} {label}")
        if detail:
            print(f"        {detail}")
    else:
        failed += 1
        print(f"  {FAIL} {label}")
        if detail:
            print(f"        {detail}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Stacked band TIFs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 1 — Stacked band GeoTIFFs")
print("=" * 60)

for fname in ["sentinel2_20230812_stack.tif", "sentinel2_20230902_stack.tif"]:
    p = ROOT / "data" / "processed" / fname
    check(f"Exists: data/processed/{fname}", p.exists())
    if p.exists():
        try:
            import rasterio
            with rasterio.open(p) as src:
                check(
                    f"  {fname}: 3 bands, float32",
                    src.count == 3 and src.dtypes[0] == "float32",
                    f"bands={src.count}, dtype={src.dtypes[0]}, size={src.width}×{src.height}",
                )
                check(
                    f"  {fname}: has valid CRS",
                    src.crs is not None,
                    str(src.crs),
                )
        except ImportError:
            print(f"  {INFO} rasterio not available — skipping band validation")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Required change raster outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2 — Required change rasters")
print("=" * 60)

required_rasters = {
    "data/processed/change_map.tif":    ("float32", 1, (0.0, 1.0)),
    "data/processed/change_binary.tif": ("uint8",   1, (0,   1)),
}

try:
    import rasterio
    import numpy as np

    for rel_path, (expected_dtype, expected_bands, val_range) in required_rasters.items():
        p = ROOT / rel_path
        exists = check(f"Exists: {rel_path}", p.exists())
        if exists:
            with rasterio.open(p) as src:
                arr = src.read(1)
                dtype_ok  = src.dtypes[0] == expected_dtype
                bands_ok  = src.count == expected_bands
                crs_ok    = src.crs is not None
                vmin, vmax = float(arr.min()), float(arr.max())
                range_ok  = vmin >= val_range[0] and vmax <= val_range[1]
                check(
                    f"  {rel_path}: dtype={expected_dtype}, bands={expected_bands}",
                    dtype_ok and bands_ok,
                    f"actual dtype={src.dtypes[0]}, bands={src.count}",
                )
                check(
                    f"  {rel_path}: values in [{val_range[0]}, {val_range[1]}]",
                    range_ok,
                    f"actual min={vmin:.4f}, max={vmax:.4f}",
                )
                check(f"  {rel_path}: has valid CRS", crs_ok, str(src.crs))

except ImportError:
    print(f"  {INFO} rasterio not available — skipping raster validation")
    for rel_path in required_rasters:
        p = ROOT / rel_path
        check(f"Exists: {rel_path}", p.exists())


# ─────────────────────────────────────────────────────────────────────────────
# PART 3a — SQLite change_features.db
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3a — SQLite database (change_features.db)")
print("=" * 60)

db_path = ROOT / "outputs" / "change_features.db"
db_exists = check("Exists: outputs/change_features.db", db_path.exists())

if db_exists:
    conn = sqlite3.connect(db_path)
    try:
        # Table exists
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        check("Table 'change_features' exists", "change_features" in tables, str(tables))

        # Schema columns
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(change_features)"
        ).fetchall()}
        required_cols = {"id", "date_before", "date_after", "area_m2", "confidence", "geometry"}
        check(
            "Schema has all required columns",
            required_cols.issubset(cols),
            f"found: {sorted(cols)}",
        )

        # Row count
        n_rows = conn.execute("SELECT COUNT(*) FROM change_features").fetchone()[0]
        check("Table has at least 1 row", n_rows > 0, f"rows={n_rows}")

        # Geometry is WKT string (not empty)
        sample_geom = conn.execute(
            "SELECT geometry FROM change_features LIMIT 1"
        ).fetchone()
        if sample_geom:
            geom_str = sample_geom[0]
            check(
                "Geometry stored as WKT text",
                isinstance(geom_str, str) and geom_str.startswith(("POLYGON", "MULTIPOLYGON")),
                geom_str[:60] + "...",
            )

        # Query 1: top by area
        print(f"\n  {INFO} Query — top 5 by area_m2:")
        rows = conn.execute(
            "SELECT id, date_before, date_after, "
            "ROUND(area_m2,1) AS area_m2, ROUND(confidence,4) AS confidence "
            "FROM change_features ORDER BY area_m2 DESC LIMIT 5"
        ).fetchall()
        header = f"  {'id':>4}  {'date_before':12}  {'date_after':12}  {'area_m2':>12}  {'confidence':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            print(f"  {r[0]:>4}  {r[1]:12}  {r[2]:12}  {r[3]:>12.1f}  {r[4]:>10.4f}")

        # Query 2: summary
        print(f"\n  {INFO} Query — summary statistics:")
        row = conn.execute(
            "SELECT COUNT(*) AS n, "
            "ROUND(SUM(area_m2)/1e6,4) AS total_km2, "
            "ROUND(AVG(confidence),4) AS avg_conf "
            "FROM change_features"
        ).fetchone()
        print(f"  n_polygons={row[0]}  total_area={row[1]} km²  avg_confidence={row[2]}")

        # Query 3: high-confidence filter
        n_hc = conn.execute(
            "SELECT COUNT(*) FROM change_features WHERE confidence > 0.6"
        ).fetchone()[0]
        print(f"\n  {INFO} High-confidence polygons (confidence > 0.6): {n_hc}")

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# PART 3b — GeoPackage
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3b — GeoPackage (change_features.gpkg)")
print("=" * 60)

gpkg_path = ROOT / "outputs" / "change_features.gpkg"
gpkg_exists = check("Exists: outputs/change_features.gpkg", gpkg_path.exists())

if gpkg_exists:
    try:
        import fiona
        layers = fiona.listlayers(str(gpkg_path))
        check("GeoPackage has at least 1 layer", len(layers) > 0, str(layers))
        print(f"  {INFO} Layers: {layers}")

        import geopandas as gpd
        for lyr in layers:
            gdf = gpd.read_file(gpkg_path, layer=lyr)
            print(f"  {INFO}  {lyr}: {len(gdf)} features, CRS={gdf.crs}")
    except ImportError:
        print(f"  {INFO} fiona/geopandas not available — skipping GeoPackage validation")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3c — utils/storage.py API
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3c — utils.storage query API")
print("=" * 60)

if db_exists:
    try:
        from utils.storage import query_change_features, load_geometry_from_sqlite

        df = query_change_features(
            "SELECT id, area_m2, confidence FROM change_features "
            "ORDER BY area_m2 DESC LIMIT 3"
        )
        check("query_change_features() returns DataFrame", len(df) > 0,
              f"shape={df.shape}")

        gdf = load_geometry_from_sqlite(where="confidence > 0")
        check("load_geometry_from_sqlite() returns GeoDataFrame",
              hasattr(gdf, "geometry") and len(gdf) > 0,
              f"features={len(gdf)}, CRS={gdf.crs}")
    except Exception as exc:
        check("utils.storage API works", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 4 — Visualizations")
print("=" * 60)

viz_files = [
    "visualizations/method_01_index_result.png",
    "visualizations/change_map.html",
    "visualizations/comparison_summary.png",
]
for rel in viz_files:
    p = ROOT / rel
    exists = p.exists()
    size   = p.stat().st_size if exists else 0
    check(f"Exists: {rel}", exists, f"size={size:,} bytes" if exists else "")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = passed + failed
print(f"Result: {passed}/{total} checks passed"
      + (" — ALL GOOD" if failed == 0 else f" — {failed} FAILED"))
print("=" * 60)

if failed > 0:
    print("\nTo generate missing outputs, run:")
    print("  python pipeline.py --method 1 --skip-vlm")
    sys.exit(1)
