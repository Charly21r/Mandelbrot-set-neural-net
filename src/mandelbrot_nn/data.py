from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .mandelbrot import smooth_escape, compute_ylim_from_x


class IndexedTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


def sample_uniform(n: int, xlim, ylim, seed: int = 0) -> np.ndarray:
    """Sample 2D points uniformly from a rectangular region.

    Args:
        n: Number of samples.
        xlim: Tuple (xmin, xmax).
        ylim: Tuple (ymin, ymax).
        seed: RNG seed for reproducibility.

    Returns:
        Array of shape (n, 2) with float32 points (x, y).
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(xlim[0], xlim[1], n)
    ys = rng.uniform(ylim[0], ylim[1], n)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def build_boundary_biased_dataset(
    n_total: int = 800_000,
    frac_boundary: float = 0.7,
    xlim=(-2.4, 1.0),
    res_for_ylim=(3840, 2160),
    ycenter: float = 0.0,
    max_iter: int = 1000,
    band=(0.05, 0.98),
    seed: int = 0,
):
    """Build a dataset that over-samples the Mandelbrot boundary.

    This constructs a mixture dataset:
      1) Uniform samples over the region.
      2) Boundary-biased samples obtained by oversampling and filtering points
         whose target value lies within a band (low, high). This band tends
         to capture points near the boundary where learning is hardest.
    
    Notes:
        - If the band is too strict and produces fewer than the desired number
          of boundary samples, the function will fall back to using fewer
          boundary points (and compensates with more uniform points).

    Args:
        n_total: Total number of samples returned.
        frac_boundary: Fraction of samples allocated to boundary-biased points.
        xlim: Tuple (xmin, xmax) for sampling.
        res_for_ylim: Resolution (W, H) used to compute ylim with square pixels.
        ycenter: Vertical center in the complex plane.
        max_iter: Iterations used for label generation.
        band: Tuple (low, high) selecting boundary-like targets.
        seed: RNG seed.

    Returns:
        X: Array (n_total, 2) of sample coordinates.
        y: Array (n_total,) of targets in [0, 1].
        ylim: Tuple (ymin, ymax) used for sampling.
    """
    rng = np.random.default_rng(seed)
    ylim = compute_ylim_from_x(xlim, res_for_ylim, ycenter=ycenter)

    n_boundary = int(n_total * frac_boundary)
    n_uniform = n_total - n_boundary

    Xu = sample_uniform(n_uniform, xlim, ylim, seed=seed)

    pool_factor = 20
    pool = sample_uniform(n_boundary * pool_factor, xlim, ylim, seed=seed + 1)

    yp = np.empty((pool.shape[0],), dtype=np.float32)
    for i, (x, y) in enumerate(pool):
        yp[i] = smooth_escape(float(x), float(y), max_iter=max_iter)

    mask = (yp > band[0]) & (yp < band[1])
    Xb = pool[mask]
    yb = yp[mask]

    if len(Xb) < n_boundary:
        keep = min(len(Xb), n_boundary)
        print(f"[warn] Boundary band too strict; got {len(Xb)} boundary points, using {keep}.")
        Xb = Xb[:keep]
        yb = yb[:keep]
        n_boundary = keep
        n_uniform = n_total - n_boundary
        Xu = sample_uniform(n_uniform, xlim, ylim, seed=seed)
    else:
        Xb = Xb[:n_boundary]
        yb = yb[:n_boundary]

    yu = np.empty((Xu.shape[0],), dtype=np.float32)
    for i, (x, y) in enumerate(Xu):
        yu[i] = smooth_escape(float(x), float(y), max_iter=max_iter)

    X = np.concatenate([Xu, Xb], axis=0).astype(np.float32)
    y = np.concatenate([yu, yb], axis=0).astype(np.float32)

    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm], ylim
