"""
Part 3 (step 2) — Persist change-feature GeoDataFrames to:

  (a) SQLite database  — change_features table, geometry stored as WKT text.
                         No SpatiaLite extension required; query with plain SQL.
  (b) GeoPackage       — one layer per method, spatially indexed.
                         Compatible with QGIS and any OGC-compliant GIS tool.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd

OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"
GPKG_PATH   = OUTPUTS_DIR / "change_features.gpkg"
SQLITE_PATH = OUTPUTS_DIR / "change_features.db"


# ── SQLite ─────────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    """Create the change_features table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS change_features (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date_before TEXT    NOT NULL,
            date_after  TEXT    NOT NULL,
            area_m2     REAL    NOT NULL,
            confidence  REAL    NOT NULL,
            geometry    TEXT    NOT NULL
        )
    """)
    conn.commit()


def save_to_sqlite(
    gdf: gpd.GeoDataFrame,
    db_path: Path = SQLITE_PATH,
) -> None:
    """
    Insert change features into the SQLite change_features table.

    Geometry is stored as WKT text (no SpatiaLite extension needed).
    Re-projects to WGS84 (EPSG:4326) before inserting.

    Parameters
    ----------
    gdf     : GeoDataFrame with columns date_before, date_after,
              area_m2, confidence, geometry.
    db_path : Path to the SQLite database file (created if absent).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        print("  [sqlite] No features to save.")
        return

    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    rows = [
        (
            str(row.get("date_before", "2023-08-12")),
            str(row.get("date_after",  "2023-09-02")),
            float(row["area_m2"]),
            float(row["confidence"]),
            row["geometry"].wkt,
        )
        for _, row in gdf.iterrows()
    ]

    conn = sqlite3.connect(db_path)
    try:
        _init_db(conn)
        conn.executemany(
            "INSERT INTO change_features "
            "(date_before, date_after, area_m2, confidence, geometry) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        print(f"  [sqlite] {len(rows)} rows → {db_path.name} [change_features]")
    finally:
        conn.close()


def query_change_features(
    sql: str = (
        "SELECT id, date_before, date_after, area_m2, confidence "
        "FROM change_features ORDER BY area_m2 DESC LIMIT 10"
    ),
    db_path: Path = SQLITE_PATH,
) -> pd.DataFrame:
    """
    Run any SELECT query against the change_features table.

    Parameters
    ----------
    sql     : SQL query string.
    db_path : Path to the SQLite database file.

    Returns
    -------
    pd.DataFrame with query results.

    Examples
    --------
    # Top 5 largest change polygons
    query_change_features(
        "SELECT id, area_m2, confidence FROM change_features "
        "ORDER BY area_m2 DESC LIMIT 5"
    )

    # Summary by date pair
    query_change_features(
        "SELECT date_before, date_after, COUNT(*) as n, "
        "SUM(area_m2) as total_m2, AVG(confidence) as avg_conf "
        "FROM change_features GROUP BY date_before, date_after"
    )

    # High-confidence polygons only
    query_change_features(
        "SELECT * FROM change_features WHERE confidence > 0.7 "
        "ORDER BY area_m2 DESC"
    )
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def load_geometry_from_sqlite(
    where: str = "",
    db_path: Path = SQLITE_PATH,
) -> gpd.GeoDataFrame:
    """
    Load change_features rows as a GeoDataFrame (geometry parsed from WKT).

    Parameters
    ----------
    where   : Optional SQL WHERE clause, e.g. "confidence > 0.6".
    db_path : Path to the SQLite database file.
    """
    sql = "SELECT * FROM change_features"
    if where:
        sql += f" WHERE {where}"
    df = query_change_features(sql, db_path)
    from shapely import wkt as shp_wkt
    df["geometry"] = df["geometry"].apply(shp_wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


# ── GeoPackage ────────────────────────────────────────────────────────────────

def save_to_geopackage(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    gpkg_path: Path = GPKG_PATH,
) -> None:
    """
    Append (or replace) a layer inside the shared GeoPackage.

    Parameters
    ----------
    gdf        : GeoDataFrame to store.
    layer_name : Layer name inside the .gpkg, e.g. 'method_01_index'.
    gpkg_path  : Destination GeoPackage file.
    """
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        print(f"  [gpkg] {layer_name}: no features to save.")
        return

    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
    print(f"  [gpkg] {len(gdf)} features → {gpkg_path.name} [{layer_name}]")


def load_from_geopackage(
    layer_name: str,
    gpkg_path: Path = GPKG_PATH,
) -> gpd.GeoDataFrame:
    """Read a layer back from the GeoPackage."""
    return gpd.read_file(gpkg_path, layer=layer_name)


def list_layers(gpkg_path: Path = GPKG_PATH) -> None:
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
