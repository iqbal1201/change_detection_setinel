import numpy as np
from typing import Tuple


def compute_change_distance(
    image_before: np.ndarray,
    image_after: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute change detection between two co-registered images using
    Euclidean distance in feature space.

    This method assumes both images are aligned and have identical shape.

    Parameters
    ----------
    image_before : np.ndarray
        Image at time T1.
        Shape: (height, width, bands)

    image_after : np.ndarray
        Image at time T2.
        Shape: (height, width, bands)

    Returns
    -------
    change_uint8 : np.ndarray
        Change map normalized to 0–255 (uint8).
        Useful for visualization or saving as raster.

    change_probability : np.ndarray
        Change probability map normalized to 0–1 (float32).
        Higher values indicate stronger change.

    Notes
    -----
    This is a simple baseline method.

    More advanced approaches may include:

    - Log-ratio (for SAR data)
    - Statistical tests
    - Machine learning
    - Multi-temporal analysis
    """

    # Convert to float to avoid overflow / underflow
    image_before = image_before.astype(np.float32)
    image_after = image_after.astype(np.float32)

    # Compute pixel-wise difference
    difference = image_after - image_before

    # Compute Euclidean distance across bands
    # Result shape: (height, width)
    distance = np.linalg.norm(difference, axis=2)

    # Normalize distance to 0–1 range
    min_val = distance.min()
    max_val = distance.max()

    # Avoid division by zero
    if max_val - min_val == 0:
        change_probability = np.zeros_like(distance, dtype=np.float32)
    else:
        change_probability = (distance - min_val) / (max_val - min_val)

    # Convert to 0–255 for visualization
    change_uint8 = (change_probability * 255).astype(np.uint8)

    return change_uint8, change_probability
