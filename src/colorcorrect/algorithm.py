# -*- coding: utf-8 -*-
"""
Colour correction / automatic white balance algorithms.

Every function takes an ``(height, width, 3)`` RGB ``numpy.uint8`` array (see
``colorcorrect.util.from_pil``) and returns a new array of the same shape and
dtype.  The input array is never modified.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from ctypes import POINTER
from ctypes import pointer
from ctypes import Structure
from ctypes import c_ubyte
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_double
from ctypes import c_void_p

import numpy as np

cutilfolder = os.path.abspath(__file__).rsplit(os.path.sep, 1)[0]
cutilname = "_cutil"
libcutil = np.ctypeslib.load_library(cutilname, cutilfolder)


class RGBImage(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
        ("r", POINTER(c_ubyte)),
        ("g", POINTER(c_ubyte)),
        ("b", POINTER(c_ubyte)),
    ]


libcutil.calc_sdwgw.argtypes = [POINTER(RGBImage), c_int, c_int]
libcutil.calc_sdwgw.restype = c_void_p
libcutil.calc_sdlwgw.argtypes = [POINTER(RGBImage), c_int, c_int]
libcutil.calc_sdlwgw.restype = c_void_p
libcutil.calc_lwgw.argtypes = [POINTER(RGBImage), c_int, c_int]
libcutil.calc_lwgw.restype = c_void_p
libcutil.delete_doubleptr.argtypes = [c_void_p]
libcutil.calc_ace.argtypes = [POINTER(RGBImage), c_int, c_double, c_double, c_uint]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _planes(nimg):
    """(H, W, 3) uint8 -> contiguous (3, H, W) uint8 copy."""
    nimg = np.asarray(nimg)
    if nimg.ndim != 3 or nimg.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array, got shape %r" % (nimg.shape,))
    return np.ascontiguousarray(nimg.transpose(2, 0, 1), dtype=np.uint8)


def _to_image(planes):
    return np.ascontiguousarray(planes.transpose(1, 2, 0))


def _apply_gains(nimg, gains, maxval=255):
    """Multiply every channel by its gain and clip to [0, maxval]."""
    out = nimg.astype(np.float64) * np.asarray(gains, dtype=np.float64)
    return np.clip(np.rint(out), 0, maxval).astype(nimg.dtype)


def _safe_ratio(numerator, denominator):
    """numerator / denominator, or 1.0 when the denominator is zero."""
    return numerator / denominator if denominator > 0 else 1.0


def _rgbimage(planes):
    return RGBImage(planes.shape[2],
                    planes.shape[1],
                    planes[0].ctypes.data_as(POINTER(c_ubyte)),
                    planes[1].ctypes.data_as(POINTER(c_ubyte)),
                    planes[2].ctypes.data_as(POINTER(c_ubyte)))


def _c_gains(func, nimg, subwidth, subheight):
    if subwidth <= 0 or subheight <= 0:
        raise ValueError("sub-block size must be positive")
    planes = _planes(nimg)
    ret = func(pointer(_rgbimage(planes)), int(subwidth), int(subheight))
    gains = np.ctypeslib.as_array((c_double * 3).from_address(ret)).copy()
    libcutil.delete_doubleptr(ret)
    return gains


# ---------------------------------------------------------------------------
# global algorithms
# ---------------------------------------------------------------------------
def stretch_pre(nimg):
    """
    Subtract the per channel minimum (first half of "stretch").

    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour
    Slides: An Empirical Evaluation'
    """
    nimg = np.asarray(nimg)
    return (nimg - nimg.min(axis=(0, 1), keepdims=True)).astype(nimg.dtype)


def grey_world(nimg):
    """Grey world: scale R and B so that their means equal the mean of G."""
    nimg = np.asarray(nimg)
    mu = nimg.reshape(-1, 3).mean(axis=0)
    gains = [_safe_ratio(mu[1], mu[0]), 1.0, _safe_ratio(mu[1], mu[2])]
    return _apply_gains(nimg, gains)


def max_white(nimg):
    """Max white: scale every channel so that its maximum becomes the brightest value."""
    nimg = np.asarray(nimg)
    if np.issubdtype(nimg.dtype, np.integer):
        brightest = np.iinfo(nimg.dtype).max
    else:
        brightest = 1.0
    mx = nimg.reshape(-1, 3).max(axis=0)
    gains = [_safe_ratio(brightest, float(m)) for m in mx]
    return _apply_gains(nimg, gains, maxval=brightest)


def stretch(nimg):
    return max_white(stretch_pre(nimg))


def retinex(nimg):
    """Retinex / white patch: scale R and B so that their maxima equal the maximum of G."""
    nimg = np.asarray(nimg)
    mx = nimg.reshape(-1, 3).max(axis=0).astype(np.float64)
    gains = [_safe_ratio(mx[1], mx[0]), 1.0, _safe_ratio(mx[1], mx[2])]
    return _apply_gains(nimg, gains)


def _quadratic_fit(channel, ref_sum, ref_max):
    """
    Solve for (mu, nu) so that mu*x^2 + nu*x satisfies both the grey world and
    the retinex constraint against the reference channel:

        [sum x^2, sum x] [mu]   [sum ref]
        [max x^2, max x] [nu] = [max ref]
    """
    x = channel.astype(np.float64)
    s1, s2 = x.sum(), np.square(x).sum()
    m1 = x.max()
    a = np.array([[s2, s1], [m1 * m1, m1]])
    if m1 <= 0 or abs(np.linalg.det(a)) < 1e-12:
        # flat (or black) channel: the system is singular, fall back to a gain
        return 0.0, _safe_ratio(ref_max, m1)
    mu, nu = np.linalg.solve(a, np.array([ref_sum, ref_max]))
    return mu, nu


def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance
    in Digital Photography' (E. Lam, 2005)

    R and B are mapped through a quadratic x -> mu*x^2 + nu*x whose
    coefficients make them satisfy both the grey world (mean) and the retinex
    (max) assumption with respect to G.
    """
    nimg = np.asarray(nimg)
    out = nimg.astype(np.float64)
    g = out[..., 1]
    ref_sum, ref_max = g.sum(), g.max()
    for c in (0, 2):
        mu, nu = _quadratic_fit(out[..., c], ref_sum, ref_max)
        x = out[..., c]
        out[..., c] = mu * x * x + nu * x
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)



# ---------------------------------------------------------------------------
# block based grey world variants (Lam, Au, Wong, ICASSP 2004)
# ---------------------------------------------------------------------------
LUMINANCE_PEAK = 160.0


def _luminance(rgb):
    return 0.298912 * rgb[..., 0] + 0.586611 * rgb[..., 1] + 0.114478 * rgb[..., 2]


def _triangular_weight(y):
    """Triangular luminance weight: 0 at Y=0, 1 at Y=160, 0 at Y=255."""
    y = np.asarray(y, dtype=np.float64)
    rising = y / LUMINANCE_PEAK
    falling = (255.0 - y) / (255.0 - LUMINANCE_PEAK)
    return np.clip(np.where(y < LUMINANCE_PEAK, rising, falling), 0.0, None)


def _block_reduce(values, subwidth, subheight):
    """Sum ``values`` (H, W[, C]) over sub-blocks; border blocks may be smaller."""
    h, w = values.shape[:2]
    rows = np.add.reduceat(values, np.arange(0, h, subheight), axis=0)
    return np.add.reduceat(rows, np.arange(0, w, subwidth), axis=1)


def _block_gains(nimg, subwidth, subheight, luminance_weighted, std_weighted):
    """
    Numpy implementation of the SDWGW / SDLWGW / LWGW gains:

        SDWA_c = sum_i sigma_c(i) / sum_j sigma_c(j) * mu_c(i)
        gain_c = mean(SDWA) / SDWA_c
    """
    if subwidth <= 0 or subheight <= 0:
        raise ValueError("sub-block size must be positive")
    f = np.asarray(nimg).astype(np.float64)
    h, w, _ = f.shape
    counts = _block_reduce(np.ones((h, w)), subwidth, subheight)[..., None]
    sums = _block_reduce(f, subwidth, subheight)
    means = sums / counts
    if luminance_weighted:
        wgt = _triangular_weight(_luminance(f))
        wsum = _block_reduce(wgt, subwidth, subheight)[..., None]
        wmeans = _block_reduce(f * wgt[..., None], subwidth, subheight)
        with np.errstate(invalid="ignore", divide="ignore"):
            wmeans = wmeans / wsum
        means = np.where(wsum > 0, wmeans, means)
    if std_weighted:
        sq = _block_reduce(np.square(f), subwidth, subheight)
        var = (sq - sums * sums / counts) / np.maximum(counts - 1, 1)
        std = np.sqrt(np.maximum(var, 0.0)).reshape(-1, 3)
    else:
        std = np.ones((means.shape[0] * means.shape[1], 3))
    means = means.reshape(-1, 3)
    std_sum = std.sum(axis=0)
    weights = np.where(std_sum > 0, std / np.where(std_sum > 0, std_sum, 1.0),
                       1.0 / means.shape[0])
    sdwa = (weights * means).sum(axis=0)
    return np.array([_safe_ratio(sdwa.mean(), v) for v in sdwa])


def standard_deviation_weighted_grey_world_python(nimg, subwidth=20, subheight=20):
    """Pure numpy version of :func:`standard_deviation_weighted_grey_world`."""
    gains = _block_gains(nimg, subwidth, subheight, luminance_weighted=False, std_weighted=True)
    return _apply_gains(nimg, gains)


def standard_deviation_and_luminance_weighted_gray_world_python(nimg, subwidth=20, subheight=20):
    """Pure numpy version of :func:`standard_deviation_and_luminance_weighted_gray_world`."""
    gains = _block_gains(nimg, subwidth, subheight, luminance_weighted=True, std_weighted=True)
    return _apply_gains(nimg, gains)


def luminance_weighted_gray_world_python(nimg, subwidth=20, subheight=20):
    """Pure numpy version of :func:`luminance_weighted_gray_world`."""
    gains = _block_gains(nimg, subwidth, subheight, luminance_weighted=True, std_weighted=False)
    return _apply_gains(nimg, gains)


def standard_deviation_weighted_grey_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    gains = _c_gains(libcutil.calc_sdwgw, nimg, subwidth, subheight)
    return _apply_gains(nimg, gains)


def standard_deviation_and_luminance_weighted_gray_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    gains = _c_gains(libcutil.calc_sdlwgw, nimg, subwidth, subheight)
    return _apply_gains(nimg, gains)


def luminance_weighted_gray_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    gains = _c_gains(libcutil.calc_lwgw, nimg, subwidth, subheight)
    return _apply_gains(nimg, gains)


# ---------------------------------------------------------------------------
# automatic color equalization (Rizzi, Gatta, Marini, 2003)
# ---------------------------------------------------------------------------
def _fast_len(n):
    """Smallest 2/3/5-smooth integer >= n (fast FFT size)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _ace_fft(planes, slope, limit, levels, workers=None):
    """
    Exact (whole image) ACE chromatic/spatial adjustment via FFT convolution.

    R_c(p) = sum_{j != p} r(I_c(p) - I_c(j)) / d(p, j)  /  sum_{j != p} 1 / d(p, j)

    For a fixed grey level v the map p -> sum_j r(v - I_c(j)) / d(p, j) is a
    convolution of r(v - I_c) with the 1/d kernel.  It is evaluated for
    ``levels`` equally spaced values of v and linearly interpolated at the
    pixel's own value, in the spirit of Getreuer's fast ACE (IPOL 2012).
    numpy's FFT releases the GIL, so the convolutions run in a thread pool.
    """
    _, h, w = planes.shape
    ph, pw = _fast_len(2 * h - 1), _fast_len(2 * w - 1)
    # circular 1/d kernel; the padding guarantees a linear convolution
    dy = np.arange(ph)
    dy = np.where(dy > ph // 2, dy - ph, dy).astype(np.float64)
    dx = np.arange(pw)
    dx = np.where(dx > pw // 2, dx - pw, dx).astype(np.float64)
    dist = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    with np.errstate(divide="ignore"):
        kernel = np.where(dist > 0, 1.0 / dist, 0.0)
    # single precision halves the memory of the padded transforms and is
    # accurate to ~1e-6 relative, far below one grey level
    kernel_f = np.fft.rfft2(kernel.astype(np.float32))

    def convolve(f):
        spectrum = np.fft.rfft2(f.astype(np.float32), s=(ph, pw)) * kernel_f
        return np.fft.irfft2(spectrum, s=(ph, pw))[:h, :w].astype(np.float64)

    denom = convolve(np.ones((h, w)))
    denom = np.where(denom > 0, denom, 1.0)

    # saturation function of the (integer) pixel difference, in units of `limit`
    sat_lut = np.clip(np.arange(-255, 256, dtype=np.float64) * slope / limit, -1.0, 1.0)
    level_values = np.linspace(0.0, 255.0, levels)
    scale = (levels - 1) / 255.0

    def level_term(args):
        img, k, _ = args
        # weight of level k in the linear interpolation at every pixel
        wk = np.clip(1.0 - np.abs(img * scale - k), 0.0, None)
        if not wk.any():
            return None
        v = level_values[k]
        if float(v).is_integer():
            sat = sat_lut[int(v) - img + 255]
        else:
            sat = np.clip((v - img) * slope / limit, -1.0, 1.0)
        return wk * convolve(sat)

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    channels = [planes[c].astype(np.int64) for c in range(3)]
    jobs = [(channels[c], k, c) for c in range(3) for k in range(levels)]
    scores = np.zeros((3, h, w))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            terms = pool.map(level_term, jobs)
            for (_, _, c), term in zip(jobs, terms):
                if term is not None:
                    scores[c] += term
    else:
        for job in jobs:
            c = job[2]
            term = level_term(job)
            if term is not None:
                scores[c] += term
    scores /= denom
    return scores


def _ace_scale(scores):
    """Dynamic tone reproduction scaling: [min, max] -> [0, 255] per channel."""
    out = np.empty(scores.shape, dtype=np.uint8)
    for c in range(scores.shape[0]):
        lo, hi = scores[c].min(), scores[c].max()
        if hi - lo <= 0:
            out[c] = 128
        else:
            out[c] = np.clip(np.rint((scores[c] - lo) * (255.0 / (hi - lo))), 0, 255)
    return out


def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500,
                                 method="fft", levels=16, seed=None, workers=None):
    """
    A. Rizzi, C. Gatta and D. Marini, "A new algorithm for unsupervised global
    and local color correction.", Pattern Recognition Letters, vol. 24, no. 11, 2003.

    slope, limit: saturation function r(x) = clip(slope * x, -limit, limit)
        applied to pixel differences in [-255, 255]; the default saturates
        for differences larger than 100 levels.
    method: "fft" (default) evaluates the full image sum exactly (up to the
        interpolation over ``levels`` grey levels) with FFT convolutions and
        is deterministic.  "sampling" is the Monte-Carlo C implementation that
        compares every pixel with ``samples`` random pixels; ``seed`` picks
        the random subset (None: a fresh random subset every call).
    levels: number of grey levels at which the "fft" method evaluates the
        convolution (16 keeps the result within one grey level of the exact
        sum on test images; 256 is exact).
    workers: threads used by the "fft" method (None: up to 8 CPU cores).
    """
    if slope <= 0 or limit <= 0:
        raise ValueError("slope and limit must be positive")
    planes = _planes(nimg)
    if method == "fft":
        if levels < 2:
            raise ValueError("levels must be at least 2")
        scores = _ace_fft(planes, float(slope), float(limit), int(levels), workers)
        return _to_image(_ace_scale(scores))
    elif method == "sampling":
        if samples <= 0:
            raise ValueError("samples must be positive")
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 32, dtype=np.uint64))
        libcutil.calc_ace(pointer(_rgbimage(planes)), int(samples), float(slope),
                          float(limit), c_uint(int(seed) & 0xFFFFFFFF))
        return _to_image(planes)
    raise ValueError("unknown method %r (expected 'fft' or 'sampling')" % (method,))
