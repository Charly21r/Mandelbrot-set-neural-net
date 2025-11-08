from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from .mandelbrot import compute_ylim_from_x
from .palettes import fractal_palette, glow


@torch.no_grad()
def model_grid_tiled(model, device, xlim, ylim, res, tile=(512, 512), amp=True):
    """Evaluate a model on a dense grid using tiled inference to avoid OOM.

    The grid covers [xlim] x [ylim] at resolution `res` and is evaluated in
    tiles to keep GPU/CPU memory bounded.

    Args:
        model: PyTorch model mapping (x, y) -> scalar.
        device: Torch device.
        xlim: Tuple (xmin, xmax).
        ylim: Tuple (ymin, ymax).
        res: Tuple (W, H) output resolution.
        tile: Tuple (tile_w, tile_h) controlling memory usage.
        amp: If True and device is CUDA, uses float16 autocast.

    Returns:
        Float32 array of shape (H, W) containing raw model outputs (logits).
    """
    model.eval()
    W, H = res
    tw, th = tile

    xs = np.linspace(xlim[0], xlim[1], W, endpoint=False, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], H, endpoint=False, dtype=np.float32)

    out = np.empty((H, W), dtype=np.float32)

    for y0 in range(0, H, th):
        y1 = min(y0 + th, H)
        Y = ys[y0:y1]

        for x0 in range(0, W, tw):
            x1 = min(x0 + tw, W)
            X = xs[x0:x1]

            XX, YY = np.meshgrid(X, Y)
            grid = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)

            g = torch.from_numpy(grid).to(
                device,
                dtype=torch.float16 if (amp and device.type == "cuda") else torch.float32,
                non_blocking=True,
            )

            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    v = model(g).squeeze(1)
            else:
                v = model(g).squeeze(1)

            out[y0:y1, x0:x1] = v.float().cpu().numpy().reshape((y1 - y0, x1 - x0))
            del XX, YY, grid, g, v

    return out


def plot_model_heatmap_tiled(
    model,
    device,
    *,
    xlim=(-2.4, 1.0),
    ycenter=0.0,
    res=(3840, 2160),
    tile=(512, 512),
    fname="render.png",
    amp=False,
    gamma=0.85,
    qlo=0.01,
    qhi=0.99,
    cmap_custom="synthwave",
    glow_strength=0.30,
    glow_radius=4,
    glow_threshold=0.5,
):
    """Render a high-resolution visualization of the learned Mandelbrot set.

    Pipeline:
      1) Evaluate model logits on a dense grid (tiled).
      2) Apply sigmoid to map logits -> [0, 1].
      3) Contrast normalization via quantiles (robust to outliers).
      4) Gamma correction to shape perceived contrast.
      5) Optional glow/bloom effect.
      6) Save as a borderless image.

    Args:
        model: PyTorch model mapping (x, y) -> scalar logit.
        device: Torch device.
        xlim: Horizontal bounds in the complex plane.
        ycenter: Center of vertical bounds (ylim computed to preserve aspect).
        res: Output resolution (W, H).
        tile: Inference tile size (tile_w, tile_h).
        fname: Output filename for the saved PNG.
        amp: Use float16 autocast on CUDA.
        gamma: Gamma correction exponent.
        qlo: Lower quantile for contrast normalization.
        qhi: Upper quantile for contrast normalization.
        cmap_custom: Name of the custom palette.
        glow_strength: Strength of glow effect.
        glow_radius: Glow diffusion iterations.
        glow_threshold: Threshold above which pixels emit glow.

    Returns:
        None. Writes an image to `fname`.
    """
    ylim = compute_ylim_from_x(xlim, res, ycenter=ycenter)

    pred = model_grid_tiled(model, device, xlim, ylim, res, tile=tile, amp=amp).astype(np.float32)
    pred = 1.0 / (1.0 + np.exp(-pred))  # sigmoid -> [0,1]

    lo, hi = np.quantile(pred, [qlo, qhi])
    pred = (pred - lo) / (hi - lo + 1e-8)
    pred = np.clip(pred, 0.0, 1.0)

    pred = pred ** gamma
    pred = glow(pred, strength=glow_strength, radius=glow_radius, threshold=glow_threshold)

    dpi = 300
    figsize = (res[0] / dpi, res[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = fractal_palette(cmap_custom)
    ax.imshow(
        pred,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        interpolation="none",
        aspect="equal",
        cmap=cmap,
    )
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(fname, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print("Saved:", fname)
