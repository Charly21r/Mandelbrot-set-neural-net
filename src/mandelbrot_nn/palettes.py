import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def fractal_palette(name: str):
    """Create a custom colormap for fractal visualization.

    Args:
        name: Palette name. Supported values include:
            "fire", "cosmic", "ocean", "synthwave", "ink".

    Returns:
        A matplotlib LinearSegmentedColormap.
    """
    palettes = {
        "fire": [
            (0.02, 0.02, 0.05),
            (0.10, 0.02, 0.20),
            (0.40, 0.05, 0.30),
            (0.80, 0.20, 0.10),
            (0.98, 0.80, 0.30),
        ],
        "cosmic": [
            (0.01, 0.01, 0.04),
            (0.05, 0.02, 0.20),
            (0.20, 0.10, 0.60),
            (0.60, 0.40, 0.90),
            (0.95, 0.85, 0.98),
        ],
        "ocean": [
            (0.01, 0.02, 0.05),
            (0.02, 0.10, 0.20),
            (0.05, 0.40, 0.50),
            (0.30, 0.80, 0.70),
            (0.90, 0.95, 0.85),
        ],
        "synthwave": [
            (0.02, 0.00, 0.08),
            (0.20, 0.00, 0.40),
            (0.60, 0.10, 0.80),
            (0.90, 0.30, 0.60),
            (1.00, 0.90, 0.30),
        ],
        "ink": [
            (0.00, 0.00, 0.00),
            (0.10, 0.10, 0.10),
            (0.40, 0.40, 0.40),
            (0.85, 0.85, 0.85),
        ],
    }
    return LinearSegmentedColormap.from_list(f"fract_{name}", palettes[name], N=2048)


def glow(img, strength=0.25, radius=3, threshold=0.6):
    """Apply a simple glow post-process to a grayscale image.

    This boosts bright regions by diffusing them a few iterations and adding
    the result back to the original image.

    Args:
        img: Float32 array in [0, 1], shape (H, W).
        strength: Amount of glow added back to the base image.
        radius: Number of diffusion iterations (higher = softer glow).
        threshold: Only pixels above this value contribute to glow.

    Returns:
        A float32 array in [0, 1] with glow applied.
    """
    src = img.copy()
    mask = np.clip((src - threshold) / (1.0 - threshold), 0.0, 1.0)
    glow_src = src * mask

    out = glow_src
    for _ in range(radius):
        out = (
            np.roll(out, 1, 0) + np.roll(out, -1, 0) +
            np.roll(out, 1, 1) + np.roll(out, -1, 1) + out
        ) / 5.0

    return np.clip(img + strength * out, 0.0, 1.0)
