# Solafune Technical Assessment — Sentinel-2 Change Detection

**AOI**: Open-pit mining site, Zambia (25.79°–25.94°E, 12.17°–12.32°S)  
**Sensor**: Sentinel-2 Level-2A, Bands B02/B03/B04 (10 m resolution)  
**Dates**: 2023-08-12 (T1) → 2023-09-02 (T2)

---

## Quick Start (local pipeline)

```bash
# 1. Create and activate a virtual environment
python -m venv solafune-env
source solafune-env/bin/activate          # Linux / macOS
solafune-env\Scripts\activate             # Windows

# 2. Install dependencies
pip install rasterio numpy scipy scikit-image scikit-learn \
            geopandas shapely folium matplotlib pandas fiona

# 3. Run the full pipeline (Methods 1–3; GPU methods skipped)
python pipeline.py --skip-gfm --skip-vlm

# 4. Run only Method 1 (fastest, produces required outputs immediately)
python pipeline.py --method 1 --skip-vlm
```

### CLI options

| Flag | Default | Effect |
|---|---|---|
| `--skip-gfm` | off | Skip DINOv2, SAM2, DCVA (no PyTorch needed) |
| `--skip-vlm` | off | Skip VLM semantic description |
| `--method N` | all | Run only method N (1–5) |
| `--no-normalize` | off | Use raw band values (no local radiometric normalization) |
| `--norm-window PX` | 64 | Window size for local z-score normalization |

---

## Output Files

### Part 1 — Stacked bands
```
data/processed/sentinel2_20230812_stack.tif
data/processed/sentinel2_20230902_stack.tif
```

### Part 2 — Change rasters
```
data/processed/change_map.tif             # change intensity [0–1]  (Method 1, primary)
data/processed/change_binary.tif          # binary change {0,1}      (Method 1, primary)
data/processed/method_0N_index_prob.tif   # per-method intensity
data/processed/method_0N_index_binary.tif # per-method binary
```

### Part 3 — Geospatial database
```
outputs/change_features.db     # SQLite — change_features table
outputs/change_features.gpkg   # GeoPackage — one layer per method (QGIS-compatible)
```

**SQLite schema**:
```sql
CREATE TABLE change_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date_before TEXT,
    date_after  TEXT,
    area_m2     REAL,
    confidence  REAL,
    geometry    TEXT   -- WKT, EPSG:4326
);
```

**Example queries** (run automatically at end of pipeline):
```python
from utils.storage import query_change_features

# Largest change polygons
query_change_features(
    "SELECT id, area_m2, confidence FROM change_features "
    "ORDER BY area_m2 DESC LIMIT 10"
)

# Summary by date pair
query_change_features(
    "SELECT date_before, date_after, COUNT(*) AS n, "
    "SUM(area_m2)/1e6 AS km2, AVG(confidence) AS avg_conf "
    "FROM change_features GROUP BY date_before, date_after"
)

# Load as GeoDataFrame
from utils.storage import load_geometry_from_sqlite
gdf = load_geometry_from_sqlite(where="confidence > 0.6")
```

### Part 4 — Visualizations
```
visualizations/method01_result.png      # 4-panel figure  (Method 1)
visualizations/method02_result.png      # 4-panel figure  (Method 2)
visualizations/method03_ml_result.png   # 4-panel figure  (Method 3)
visualizations/change_map.html          # Interactive folium map (AOI + all polygons)
visualizations/comparison_summary.png   # Area & confidence comparison chart
```

---

## Cloud Web App

A FastAPI web application for on-demand change detection is in `cloud_deployment/`.  
See [cloud_deployment/README.md](cloud_deployment/README.md) for deployment instructions.

---

## Methods

| ID | Name | Library | GPU |
|---|---|---|---|
| 1 | Multi-Index PCA Fusion | scikit-learn, scikit-image | No |
| 2 | IR-MAD (Statistical) | scipy | No |
| 3 | GMM-EM + Isolation Forest | scikit-learn | No |
| 4a | DINOv2 Zero-Shot | transformers, PyTorch | Yes |
| 4b | SAM2 / AnyChange | sam2, PyTorch | Yes |
| 5 | Deep CVA (DCVA) | PyTorch | Yes |

---

## Approach

**Part 1** — `utils/data_loader.py` reads individual B02/B03/B04 GeoTIFFs, verifies CRS/transform/shape consistency, and writes multi-band stacked GeoTIFFs to `data/processed/`.

**Part 2** — Each method in `methods/` returns `(change_prob, change_binary)`. Method 1 output is written to the canonical `change_map.tif` / `change_binary.tif` paths; all methods also write their own named copies.

**Part 3** — `utils/vectorizer.py` converts binary rasters to Shapely polygons via `rasterio.features.shapes`, filters small areas (< 900 m²), and attaches mean confidence from the probability map. `utils/storage.py` persists results to both SQLite (plain WKT geometry) and GeoPackage.

**Part 4** — `utils/visualizer.py` produces a 4-panel matplotlib figure per method, a combined folium interactive HTML map with AOI boundary and Google Satellite basemap, and a comparison bar chart.

**Part 5** — `report.md` documents method rationale, observed change patterns, and interpretation in the context of open-pit mining dynamics.

---

## Assumptions

- Bands are already co-registered and clipped to the same extent (provided data).
- Only B02, B03, B04 (visible bands) are available — no NIR or SWIR.
- Methods 4a/4b/5 require PyTorch ≥ 2.0 and a CUDA-capable GPU (or run slowly on CPU).
- Minimum detectable change area: 900 m² (one 30 m pixel; actual pixel is 10 m → 100 m², but morphological filtering effectively raises this threshold).
