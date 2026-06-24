"""
Method 2 — Iteratively Reweighted Multivariate Alteration Detection (IR-MAD).

Reference:
  Nielsen, A.A. (2007). The regularized iteratively reweighted MAD method
  for change detection in multi- and hyperspectral data.
  IEEE Transactions on Image Processing, 16(2), 463–478.

Algorithm summary
-----------------
1. Initialise pixel weights w_i = 1/N.
2. Solve a *weighted* Canonical Correlation Analysis (CCA) between X1 and X2.
3. Compute MAD variates: Z = a'X1 - b'X2 (one per band).
4. Form chi-squared statistic from the MAD variates.
5. Update weights: w_i ∝ P(no-change | χ²_i) — changed pixels get lower weight.
6. Repeat until convergence (typically 5-20 iterations).
Output: change_prob = 1 - P(no-change), i.e. chi-squared CDF evaluated
        at each pixel's test statistic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening


# ── Weighted CCA ─────────────────────────────────────────────────────────────

def _weighted_cov(A: np.ndarray, B: np.ndarray,
                  w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute weighted cross-covariance matrices.

    Returns (Sigma_AA, Sigma_BB, Sigma_AB) where w is a 1-D weight vector
    summing to 1.
    """
    w = w / w.sum()

    mu_A = (w[:, None] * A).sum(axis=0)
    mu_B = (w[:, None] * B).sum(axis=0)
    Ac = A - mu_A
    Bc = B - mu_B

    w_sqrt = np.sqrt(w)[:, None]
    Aw = Ac * w_sqrt
    Bw = Bc * w_sqrt

    S_AA = Aw.T @ Aw
    S_BB = Bw.T @ Bw
    S_AB = Aw.T @ Bw

    return S_AA, S_BB, S_AB


def _weighted_cca(X1: np.ndarray, X2: np.ndarray,
                  w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted CCA: find canonical vectors a, b for X1 and X2.
    Returns (A, B) as (n_bands × n_bands) matrices (columns = variates).
    """
    S11, S22, S12 = _weighted_cov(X1, X2, w)

    n = X1.shape[1]
    # Regularise to avoid singular matrices
    eps_reg = 1e-6 * np.trace(S11) / n
    S11 += eps_reg * np.eye(n)
    S22 += eps_reg * np.eye(n)

    # Cholesky decompositions for numerical stability
    try:
        L11 = np.linalg.cholesky(S11)
        L22 = np.linalg.cholesky(S22)
    except np.linalg.LinAlgError:
        L11 = np.linalg.cholesky(S11 + 1e-5 * np.eye(n))
        L22 = np.linalg.cholesky(S22 + 1e-5 * np.eye(n))

    L11_inv = np.linalg.inv(L11)
    L22_inv = np.linalg.inv(L22)

    # Solve symmetric eigenvalue problem
    M = L11_inv @ S12 @ L22_inv.T
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    A = L11_inv.T @ U     # canonical directions for X1
    B = L22_inv.T @ Vt.T  # canonical directions for X2

    return A, B


# ── IR-MAD iteration ─────────────────────────────────────────────────────────

def run(
    img1: np.ndarray,
    img2: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-4,
    sigma_smooth: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run IR-MAD change detection.

    Parameters
    ----------
    img1, img2 : (H, W, 3) float32
    max_iter   : maximum CCA iterations
    tol        : convergence threshold on weight change (L-inf norm)
    sigma_smooth : Gaussian smoothing applied to the change probability map

    Returns
    -------
    change_prob   : (H, W) float32, chi-squared CDF → [0, 1]
    change_binary : (H, W) uint8, {0, 1}
    """
    print("  [Method 2] Initialising IR-MAD...")
    h, w, n_bands = img1.shape
    N = h * w

    X1 = img1.reshape(N, n_bands).astype(np.float64)
    X2 = img2.reshape(N, n_bands).astype(np.float64)

    # Normalise each band to [0,1] to make bands comparable
    for j in range(n_bands):
        lo, hi = X1[:, j].min(), X1[:, j].max()
        X1[:, j] = (X1[:, j] - lo) / (hi - lo + 1e-8)
        lo, hi = X2[:, j].min(), X2[:, j].max()
        X2[:, j] = (X2[:, j] - lo) / (hi - lo + 1e-8)

    # Initial uniform weights
    w_vec = np.ones(N, dtype=np.float64) / N
    chi2_vals = np.zeros(N, dtype=np.float64)

    for iteration in range(max_iter):
        A, B = _weighted_cca(X1, X2, w_vec)

        # MAD variates: Z_i = a_i' x1 - b_i' x2
        Z = X1 @ A - X2 @ B               # (N, n_bands)

        # Variance of each MAD variate (weighted)
        w_norm = w_vec / w_vec.sum()
        mu_Z = (w_norm[:, None] * Z).sum(axis=0)
        Zc = Z - mu_Z
        var_Z = (w_norm[:, None] * Zc ** 2).sum(axis=0)
        var_Z = np.maximum(var_Z, 1e-10)

        # Chi-squared statistic (sum of squared standardised MAD variates)
        chi2_vals = np.sum(Zc ** 2 / var_Z, axis=1)

        # Update weights: P(no change | chi2)
        w_new = 1.0 - chi2.cdf(chi2_vals, df=n_bands)
        w_new = np.maximum(w_new, 1e-10)
        w_new /= w_new.sum()

        delta = np.max(np.abs(w_new - w_vec))
        w_vec = w_new

        print(f"    Iteration {iteration + 1:2d}/{max_iter} — "
              f"weight Δ = {delta:.2e}")
        if delta < tol:
            print(f"    Converged after {iteration + 1} iterations.")
            break

    # Change probability = chi-squared CDF (higher = more likely changed)
    change_prob = chi2.cdf(chi2_vals, df=n_bands).reshape(h, w).astype(np.float32)

    # Smooth
    change_prob = gaussian_filter(change_prob, sigma=sigma_smooth)

    # Threshold: 95th percentile of chi-squared under H0
    thresh = chi2.ppf(0.95, df=n_bands)
    binary = (chi2_vals.reshape(h, w) > thresh).astype(np.uint8)

    struct = np.ones((3, 3), dtype=bool)
    binary = binary_opening(binary, structure=struct).astype(np.uint8)
    binary = binary_closing(binary, structure=struct).astype(np.uint8)

    n_changed = int(binary.sum())
    pct = 100.0 * n_changed / binary.size
    print(f"  [Method 2] Done — {n_changed:,} changed pixels ({pct:.2f}%)")

    return change_prob, binary
