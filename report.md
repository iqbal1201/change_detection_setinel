# Change Detection Report
## Solafune Technical Assessment

**AOI**: Open-pit mining site, Zambia (25.79°–25.94°E, 12.17°–12.32°S)
**Date Before**: 2023-08-12 (Sentinel-2 Level-2A)
**Date After**: 2023-09-02 (Sentinel-2 Level-2A)
**Bands used**: B02 (Blue), B03 (Green), B04 (Red) — 10 m resolution

---

## Part 1 — Data Preparation

Both dates share the same CRS (UTM Zone 35S / EPSG:32735), spatial
resolution (10 m), and extent.  Bands were loaded from individual GeoTIFFs,
verified for dimensional and CRS consistency, then stacked into two
3-band GeoTIFF outputs:

- `data/processed/sentinel2_20230812_stack.tif`
- `data/processed/sentinel2_20230902_stack.tif`

---

## Part 2 — Change Detection Methods

### Method 1 — Multi-Index PCA Fusion

**Algorithm**: Five visible-band spectral indices are computed for both dates
(VARI, NGRDI, Brightness Index, Excess Red, Excess Green).  The absolute
per-index differences are stacked and fused via Principal Component Analysis,
using explained-variance weights so high-signal indices contribute more to
the composite change score.  Binary thresholding uses Otsu's method.

**Why**: Index-based methods are fully interpretable.  PCA fusion avoids
arbitrary per-index weighting.  VARI and ExG are sensitive to vegetation loss
(mining clearance), while BI and ExR capture bare-soil expansion.

**Rationale for index selection**: Mining activity on this AOI should manifest
as (a) vegetation removal → VARI/NGRDI decrease, (b) bare-earth increase →
BI/ExR increase.  All five indices are computable from visible bands only.

---

### Method 2 — IR-MAD (Iteratively Reweighted Multivariate Alteration Detection)

**Algorithm**: Canonical Correlation Analysis (CCA) is applied between the two
images with pixel weights that are iteratively updated to suppress changed
pixels in the background estimation.  This produces band-decorrelated MAD
variates whose sum of squares follows a chi-squared distribution under the
no-change hypothesis.  Change probability = chi-squared CDF evaluated at each
pixel's test statistic.

**Reference**: Nielsen, A.A. (2007). The regularized iteratively reweighted MAD
method for change detection in multi- and hyperspectral data. *IEEE
Transactions on Image Processing*, 16(2), 463–478.

**Why**: IR-MAD is the gold standard for bi-temporal optical change detection.
Unlike simple band differencing, it accounts for inter-band correlation and
sensor variability, and provides a statistically principled probability output.
No threshold selection is required — the 95th percentile of the chi-squared
null distribution is used.

---

### Method 3 — GMM-EM + Isolation Forest

**Algorithm**: A 10-feature vector is built per pixel (raw band differences,
squared differences, log-ratio per band, Euclidean magnitude).  Two
complementary unsupervised algorithms are applied:

1. **Gaussian Mixture Model (k=3, EM)**: fits change / no-change / ambiguous
   components.  The component with highest mean magnitude is labelled "change".
2. **Isolation Forest**: detects changed pixels as anomalies in the
   multi-dimensional difference space.  Changed pixels are structurally
   different from the majority stable population.

Both scores are averaged (equal weight) into a fused change probability.

**References**:
- Bruzzone, L. & Prieto, D.F. (2000). *IEEE TGRS*, 38(3), 1171–1182.
- Liu, F.T. et al. (2008). Isolation Forest. *ICDM 2008*.

**Why**: GMM-EM brings probabilistic multi-class semantics; Isolation Forest
captures non-Gaussian change signatures (outlier patches of mine expansion).
Together they outperform either algorithm alone on heterogeneous change types.

---

### Method 4a — DINOv2 Zero-Shot Feature Distance

**Algorithm**: Both images are treated as pseudo-RGB (B4→R, B3→G, B2→B) and
passed through a frozen DINOv2-Base (ViT-B/14) backbone.  Dense patch tokens
(14×14 pixel patches) are extracted for both dates.  Cosine distance between
corresponding patch-feature vectors produces a coarse change map, which is
bilinearly upsampled to full resolution and Gaussian-smoothed.

**Reference**: Oquab, M. et al. (2024). DINOv2: Learning Robust Visual
Features without Supervision. *Transactions on Machine Learning Research*.

**Why**: DINOv2 learned rich semantic representations from 142 M images.
Without any task-specific fine-tuning it detects structural and semantic
changes that spectral indices miss.  Patch-level cosine distance is sensitive
to appearance change regardless of illumination direction — relevant for
a 21-day interval with potential sun-angle differences.

---

### Method 4b — SAM2 / AnyChange Structural Mask Comparison

**Algorithm**: SAM2's automatic mask generator is run independently on both
dates.  A smooth "mask density map" — weighted by inverse segment size — is
built for each date.  The absolute difference in density maps forms the
primary change signal.  A binary segment-coverage difference adds spatial
precision.  Both signals are fused (0.6 / 0.4 weights) and thresholded.

**References**:
- Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos.
  *arXiv:2408.00714*. FAIR / Meta AI.
- Zheng, C. et al. (2024). AnyChange: Adaptable Zero-Shot Change Detection
  via Structured Semantic Correspondence. *NeurIPS 2024*.

**Why**: SAM2 segments semantically coherent regions rather than individual
pixels.  Structural changes in a mining context (new pits, tailings ponds,
haul roads) manifest as fundamentally different segmentation patterns between
dates — something SAM2 captures that band-level methods miss.

---

## Part 3 — Feature Extraction & Storage

Changed pixels are vectorised using `rasterio.features.shapes`, filtered by
minimum area (900 m² = 1 × 10 m pixel), and stored as polygon features in a
shared GeoPackage (`outputs/change_features.gpkg`) with one layer per method:

| Layer | Schema |
|---|---|
| `method_01_index` | id, method, date_before, date_after, area_m2, confidence, geometry |
| `method_02_irmad` | id, method, date_before, date_after, area_m2, confidence, geometry |
| `method_03_ml` | id, method, date_before, date_after, area_m2, confidence, geometry |
| `method_04a_dinov2` | id, method, date_before, date_after, area_m2, confidence, geometry |
| `method_04b_sam2` | id, method, date_before, date_after, area_m2, confidence, geometry |

All layers stored in WGS84 (EPSG:4326) for GIS compatibility.

---

## Part 4 — Visualization

- **Static figures**: `visualizations/<method>_result.png` — 4-panel layout
  (before / after / change intensity / binary + polygons).
- **Interactive map**: `visualizations/change_map.html` — folium map with
  satellite basemap, per-method polygon layers, AOI boundary, and measure tool.
- **Summary chart**: `visualizations/comparison_summary.png` — total changed
  area and mean confidence across all methods.

---

## Part 5 — Results & Interpretation

*(Fill in after running the pipeline — replace the placeholders below.)*

### Where changes occur

> *[Describe spatial distribution: NW corner of AOI, along haul road, etc.]*

### Patterns observed

> *[Number of polygons, size distribution, clustering vs. scattered pattern]*

### Method agreement

> *[Which methods agree? Where do they diverge? What does divergence imply?]*

### Interpretation

The AOI is an active open-pit copper mine.  In a 21-day window (Aug 12 →
Sep 2, 2023), expected change drivers are:

| Change type | Expected signal |
|---|---|
| Pit expansion / overburden removal | BI ↑, ExR ↑, VARI ↓ |
| New haul roads | Linear bright features |
| Tailings pond change | Water index shift, SAM2 mask topology change |
| Vegetation clearing | NGRDI ↓, ExG ↓ |
| Atmospheric / illumination artefact | Spatially diffuse, all methods agree weakly |

> *[State which of the above best matches the observed change polygons and why.]*

---

## Limitations

- Only three visible bands available (no NIR, SWIR) — limits vegetation and
  water indices that would improve discrimination.
- Single 21-day bi-temporal pair — cannot distinguish seasonal phenology from
  structural change.
- DINOv2 and SAM2 were pre-trained on natural (non-satellite) imagery —
  domain gap may reduce sensitivity to subtle reflectance changes.
- No ground truth available for quantitative accuracy assessment.
