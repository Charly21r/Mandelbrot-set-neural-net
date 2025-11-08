import math
import numpy as np


def smooth_escape(x: float, y: float, max_iter: int = 1000) -> float:
    """Compute a smooth, normalized escape-time value for the Mandelbrot set.

    This is a continuous variant of the classic integer escape-time algorithm.
    Points that do not escape within `max_iter` are treated as inside the set
    and return 1.0.

    The returned value is:
      - in [0, 1]
      - higher for points that remain bounded longer
      - log-scaled for better dynamic range near small escape counts

    Args:
        x: Real part of the complex coordinate c = x + i y.
        y: Imaginary part of the complex coordinate c = x + i y.
        max_iter: Maximum number of iterations for the escape-time loop.

    Returns:
        A float in [0, 1], where 1.0 indicates "did not escape within max_iter".
    """
    c = complex(x, y)
    z = 0j
    for n in range(max_iter):
        z = z * z + c
        r2 = z.real * z.real + z.imag * z.imag
        if r2 > 4.0:
            r = math.sqrt(r2)
            mu = n + 1 - math.log(math.log(r)) / math.log(2.0)
            v = math.log1p(mu) / math.log1p(max_iter)
            return float(np.clip(v, 0.0, 1.0))
    return 1.0


def compute_ylim_from_x(xlim, res, ycenter: float = 0.0):
    """Compute y-limits that preserve square pixels in the complex plane.

    When rendering an image of shape (W, H), this ensures that the step size
    in the x direction matches the step size in the y direction, avoiding
    distortion of the fractal.

    Args:
        xlim: Tuple (xmin, xmax) describing the horizontal bounds.
        res: Tuple (W, H) target image resolution in pixels.
        ycenter: Center of the vertical range in the complex plane.

    Returns:
        Tuple (ymin, ymax) such that pixel aspect ratio in the complex plane is 1:1.
    """
    W, H = res
    step = (xlim[1] - xlim[0]) / W
    y_half = step * H / 2
    return (ycenter - y_half, ycenter + y_half)
