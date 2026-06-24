"""
Method 3 — Unsupervised Machine Learning: GMM-EM + Isolation Forest.

Two complementary unsupervised ML approaches are applied to the difference
image and their outputs are fused:

(A) Gaussian Mixture Model with Expectation-Maximisation (GMM-EM)
    Reference:
      Bruzzone, L. & Prieto, D.F. (2000). Automatic analysis of the
      difference image for unsupervised change detection.
      IEEE Transactions on Geoscience and Remote Sensing, 38(3), 1171–1182.

    Fits k=3 Gaussian components (no-change / ambiguous / change) to a
    PCA-reduced difference image.  The component with the highest mean
    magnitude is labelled "change".

(B) Isolation Forest
    Reference:
      Liu, F.T., Ting, K.M. & Zhou, Z-H. (2008). Isolation Forest.
      ICDM 2008.  Applied here as anomaly detection on multi-feature
      difference vectors — changed pixels are anomalous.

Both scores are averaged into a single fused change probability map.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Build a rich feature matrix for each pixel.

    Features (per pixel):
      - Raw band differences  ΔB, ΔG, ΔR  (3)
      - Squared differences               (3)
      - Ratio log(T2/T1) per band         (3)
      - Magnitude (Euclidean distance)    (1)
    Total: 10 features.
    """
    d = (img2 - img1).astype(np.float64)      # (H, W, 3)
    h, w, b = d.shape

    # Avoid log(0): shift to strictly positive
    img1_pos = img1.astype(np.float64) + 1.0
    img2_pos = img2.astype(np.float64) + 1.0

    log_ratio = np.log(img2_pos / img1_pos)   # (H, W, 3)
    magnitude = np.linalg.norm(d, axis=2, keepdims=True)  # (H, W, 1)

    feature_stack = np.concatenate(
        [d, d ** 2, log_ratio, magnitude], axis=-1
    )  # (H, W, 10)

    return feature_stack.reshape(h * w, -1).astype(np.float32)


# ── GMM-EM component ─────────────────────────────────────────────────────────

def _gmm_change_score(X: np.ndarray, n_components: int = 3,
                      pca_components: int = 5) -> np.ndarray:
    """
    Fit a GMM on PCA-reduced features. Return per-pixel change probability.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA reduces dimensionality and decorrelates features
    n_pca = min(pca_components, X.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=300,
        n_init=5,
        random_state=42,
    )
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)           # (N,)
    probs  = gmm.predict_proba(X_pca)     # (N, k)

    # Identify the "change" component: highest mean Euclidean distance in raw space
    means_raw = np.array([
        X[labels == k].mean(axis=0) if (labels == k).any() else np.zeros(X.shape[1])
        for k in range(n_components)
    ])
    # Use last feature (magnitude) to rank components
    change_component = int(np.argmax(np.abs(means_raw[:, -1])))

    change_prob = probs[:, change_component]
    return change_prob.astype(np.float32)


# ── Isolation Forest component ────────────────────────────────────────────────

def _isolation_forest_score(X: np.ndarray,
                             contamination: float = 0.1) -> np.ndarray:
    """
    Return per-pixel anomaly score normalised to [0, 1].
    Higher = more anomalous = more likely changed.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # decision_function returns negative anomaly score (lower = more anomalous)
    raw_scores = iso.decision_function(X_scaled)  # (N,), lower = more anomalous

    # Invert and normalise to [0, 1]
    scores = -raw_scores
    lo, hi = scores.min(), scores.max()
    norm = (scores - lo) / (hi - lo + 1e-8)
    return norm.astype(np.float32)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    img1: np.ndarray,
    img2: np.ndarray,
    gmm_components: int = 3,
    iso_contamination: float = 0.10,
    fusion_weights: Tuple[float, float] = (0.5, 0.5),
    sigma_smooth: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run GMM-EM + Isolation Forest change detection.

    Parameters
    ----------
    img1, img2          : (H, W, 3) float32
    gmm_components      : number of GMM components (default 3)
    iso_contamination   : expected fraction of changed pixels for IF
    fusion_weights      : (w_gmm, w_iso) for score averaging
    sigma_smooth        : Gaussian smoothing sigma

    Returns
    -------
    change_prob   : (H, W) float32, [0, 1]
    change_binary : (H, W) uint8, {0, 1}
    """
    print("  [Method 3] Building feature matrix...")
    X = _build_features(img1, img2)   # (N, 10)
    h, w = img1.shape[:2]

    print(f"  [Method 3] Fitting GMM (k={gmm_components})...")
    gmm_score = _gmm_change_score(X, n_components=gmm_components)

    print("  [Method 3] Fitting Isolation Forest...")
    iso_score = _isolation_forest_score(X, contamination=iso_contamination)

    # Fuse
    w_gmm, w_iso = fusion_weights
    change_prob = (w_gmm * gmm_score + w_iso * iso_score).reshape(h, w)

    # Normalise fused score to [0, 1]
    lo, hi = change_prob.min(), change_prob.max()
    change_prob = ((change_prob - lo) / (hi - lo + 1e-8)).astype(np.float32)

    # Smooth
    change_prob = gaussian_filter(change_prob, sigma=sigma_smooth)

    # Binary threshold at mean + 1.5 * std (robust to skewed distributions)
    thresh = change_prob.mean() + 1.5 * change_prob.std()
    thresh = min(thresh, 0.85)  # cap so we don't get an empty binary map
    binary = (change_prob > thresh).astype(np.uint8)

    struct = np.ones((3, 3), dtype=bool)
    binary = binary_opening(binary, structure=struct).astype(np.uint8)
    binary = binary_closing(binary, structure=struct).astype(np.uint8)

    n_changed = int(binary.sum())
    pct = 100.0 * n_changed / binary.size
    print(f"  [Method 3] Done — {n_changed:,} changed pixels ({pct:.2f}%)")

    return change_prob, binary
