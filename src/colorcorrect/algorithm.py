# -*- coding: utf-8 -*-
"""
Colour correction / automatic white balance algorithms.

Every function takes an ``(height, width, 3)`` RGB ``numpy.uint8`` array (see
``colorcorrect.util.from_pil``) and returns a new array of the same shape and
dtype.  The input array is never modified.
"""
import os
import sys
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


class _MissingExtension(object):
    """
    Stand-in for the C++ helper library when it could not be loaded (for
    example in a pure Python checkout or a platform without the compiled
    extension).  Any call explains what is missing instead of failing with an
    AttributeError.
    """

    def __init__(self, error):
        self.error = error

    def __getattr__(self, name):
        raise RuntimeError("the colorcorrect C++ extension (_cutil) is not available: %s. "
                           "The block based grey world functions fall back to numpy; "
                           "method='sampling' ACE, random_spray_retinex and race need "
                           "the extension." % (self.error,))

    def __bool__(self):
        return False


try:
    libcutil = np.ctypeslib.load_library(cutilname, cutilfolder)
except OSError as _error:  # pragma: no cover - depends on the installation
    libcutil = _MissingExtension(_error)


def has_extension():
    """True when the compiled C++ helper library is available."""
    return bool(libcutil)


def _spray_workers(workers):
    """Thread count handed to the C++ spray code: 0 = all cores, 1 where threads do not exist."""
    if sys.platform in ("emscripten", "wasi"):
        return 1
    return int(workers) if workers else 0


def _default_workers():
    """Threads for the numpy FFT code paths (1 where threads are unavailable, e.g. Pyodide)."""
    if sys.platform == "emscripten" or sys.platform == "wasi":
        return 1
    return min(8, os.cpu_count() or 1)


class RGBImage(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
        ("r", POINTER(c_ubyte)),
        ("g", POINTER(c_ubyte)),
        ("b", POINTER(c_ubyte)),
    ]


if libcutil:
    libcutil.calc_sdwgw.argtypes = [POINTER(RGBImage), c_int, c_int]
    libcutil.calc_sdwgw.restype = c_void_p
    libcutil.calc_sdlwgw.argtypes = [POINTER(RGBImage), c_int, c_int]
    libcutil.calc_sdlwgw.restype = c_void_p
    libcutil.calc_lwgw.argtypes = [POINTER(RGBImage), c_int, c_int]
    libcutil.calc_lwgw.restype = c_void_p
    libcutil.delete_doubleptr.argtypes = [c_void_p]
    libcutil.calc_ace.argtypes = [POINTER(RGBImage), c_int, c_double, c_double, c_uint, c_double]
    libcutil.calc_ace_scores.argtypes = [POINTER(RGBImage), c_int, c_double, c_double, c_uint, c_double,
                                         np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")]
    libcutil.calc_rsr.argtypes = [POINTER(RGBImage), POINTER(RGBImage), c_int, c_int, c_double, c_uint, c_int]
    libcutil.calc_race.argtypes = [POINTER(RGBImage), POINTER(RGBImage), c_int, c_int, c_double, c_double,
                                   c_int, c_uint, c_int]


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
    # a flat (or black) channel makes the rows collinear; test the determinant
    # relative to the row norms so the check does not depend on image size
    scale = np.linalg.norm(a[0]) * np.linalg.norm(a[1])
    if m1 <= 0 or scale <= 0 or abs(np.linalg.det(a)) <= 1e-9 * scale:
        return 0.0, _safe_ratio(ref_max, m1)
    try:
        mu, nu = np.linalg.solve(a, np.array([ref_sum, ref_max]))
    except np.linalg.LinAlgError:
        return 0.0, _safe_ratio(ref_max, m1)
    return mu, nu


def retinex_with_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance
    in Digital Photography' (E. Lam, 2005)

    R and B are mapped through a quadratic x -> mu*x^2 + nu*x whose
    coefficients make them satisfy both the grey world (mean) and the retinex
    (max) assumption with respect to G.  The mapping already enforces the
    retinex constraint, so no separate retinex pass is run first.
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
    if not libcutil:
        gains = _block_gains(nimg, subwidth, subheight, luminance_weighted=False, std_weighted=True)
    else:
        gains = _c_gains(libcutil.calc_sdwgw, nimg, subwidth, subheight)
    return _apply_gains(nimg, gains)


def standard_deviation_and_luminance_weighted_gray_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    if not libcutil:
        gains = _block_gains(nimg, subwidth, subheight, luminance_weighted=True, std_weighted=True)
    else:
        gains = _c_gains(libcutil.calc_sdlwgw, nimg, subwidth, subheight)
    return _apply_gains(nimg, gains)


def luminance_weighted_gray_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    if not libcutil:
        gains = _block_gains(nimg, subwidth, subheight, luminance_weighted=True, std_weighted=False)
    else:
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
    h, w = planes.shape[1:]
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
        workers = _default_workers()
    channels = [plane.astype(np.int64) for plane in planes]
    jobs = [(channels[c], k, c) for c in range(len(channels)) for k in range(levels)]
    scores = np.zeros((len(channels), h, w))
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


ACE_SCALINGS = ("minmax", "wp_gw", "wp_gw_black", "keep_gray")


def _ace_scale(scores, scaling="minmax", center=None):
    """
    Dynamic tone reproduction scaling of the ACE scores R_c (in [-1, 1]).

    minmax:      O = s_c (R_c - m_c), [m_c, M_c] -> [0, 255]      (Rizzi 2003)
    wp_gw:       O = 127.5 + 127.5 R_c / M_c, negatives clipped   (Rizzi 2003,
                 white patch + grey world: R = 0 is middle grey)
    wp_gw_black: as wp_gw above R = 0, but the negative side gets its own
                 slope 127.5 / |m_c| so that m_c reaches 0 and the whole
                 dynamic is used (Chambah et al. 2003, "WP+GW (black)")
    keep_gray:   wp_gw_black with the original mean grey level of the image
                 (``center``, the same for every channel) instead of 127.5 as
                 the value of R = 0 (Chambah et al. 2003, "keep original
                 gray"): the key of the picture is preserved while the
                 common centre still neutralises the cast
    The piecewise forms are our reading of the textual description in
    Chambah et al.; the paper gives no formula for them.
    """
    if scaling not in ACE_SCALINGS:
        raise ValueError("scaling must be one of %r" % (ACE_SCALINGS,))
    out = np.empty(scores.shape, dtype=np.uint8)
    for c in range(scores.shape[0]):
        r = scores[c]
        lo, hi = r.min(), r.max()
        if scaling == "minmax":
            if hi - lo <= 0:
                out[c] = 128
            else:
                out[c] = np.clip(np.rint((r - lo) * (255.0 / (hi - lo))), 0, 255)
            continue
        mid = 127.5 if scaling != "keep_gray" else float(center)
        up = (255.0 - mid) / hi if hi > 0 else 0.0
        if scaling == "wp_gw":
            down = up
        else:
            down = mid / (-lo) if lo < 0 else 0.0
        mapped = mid + np.where(r >= 0, r * up, r * down)
        out[c] = np.clip(np.rint(mapped), 0, 255)
    return out


def _luminance_plane(planes):
    """(3, H, W) uint8 -> (H, W) float64 BT.601 luma."""
    f = planes.astype(np.float64)
    return 0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]


def _ace_scores(planes, slope, limit, samples, method, levels, seed, workers, min_distance):
    """ACE chromatic/spatial adjustment of a (n, H, W) uint8 stack -> (n, H, W) float scores."""
    if method == "fft":
        if levels < 2:
            raise ValueError("levels must be at least 2")
        return _ace_fft(planes, float(slope), float(limit), int(levels), workers)
    elif method == "sampling":
        if samples <= 0:
            raise ValueError("samples must be positive")
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 32, dtype=np.uint64))
        if min_distance is None:
            # half the mean spacing between random samples
            min_distance = 0.5 * np.sqrt(planes.shape[1] * planes.shape[2] / float(samples))
        work = np.ascontiguousarray(planes)
        n, h, w = work.shape
        if n == 3:
            image = _rgbimage(work)
        else:
            # luminance-only call: hand the same plane to r, g and b; the C
            # code detects the aliasing and processes it once
            plane = work[0].ctypes.data_as(POINTER(c_ubyte))
            image = RGBImage(w, h, plane, plane, plane)
        scores = np.zeros((3, h, w), dtype=np.float32)
        libcutil.calc_ace_scores(pointer(image), int(samples), float(slope), float(limit),
                                 c_uint(int(seed) & 0xFFFFFFFF), float(min_distance), scores)
        return scores[:n].astype(np.float64)
    raise ValueError("unknown method %r (expected 'fft' or 'sampling')" % (method,))


def _ace_planes(planes, slope, limit, samples, method, levels, seed, workers, min_distance,
                scaling="minmax"):
    """Run ACE on a (n, H, W) uint8 stack of planes, return (n, H, W) uint8."""
    scores = _ace_scores(planes, slope, limit, samples, method, levels, seed, workers, min_distance)
    return _ace_scale(scores, scaling, float(planes.mean()))


def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500,
                                 method="fft", levels=16, seed=None, workers=None,
                                 min_distance=None, preserve_chroma=False,
                                 alpha=None, scaling="minmax"):
    """
    A. Rizzi, C. Gatta and D. Marini, "A new algorithm for unsupervised global
    and local color correction.", Pattern Recognition Letters, vol. 24, no. 11, 2003.

    slope, limit: saturation function r(x) = clip(slope * x, -limit, limit)
        applied to pixel differences in [-255, 255]; the default saturates
        for differences larger than 100 levels.  Only the ratio slope/limit
        matters.
    method: "fft" (default) evaluates the full image sum exactly (up to the
        interpolation over ``levels`` grey levels) with FFT convolutions and
        is deterministic.  "sampling" is the Monte-Carlo C implementation that
        compares every pixel with ``samples`` random pixels; ``seed`` picks
        the random subset (None: a fresh random subset every call).
    levels: number of grey levels at which the "fft" method evaluates the
        convolution (16 keeps the result within one grey level of the exact
        sum on test images; 256 is exact).
    workers: threads used by the "fft" method (None: up to 8 CPU cores).
    min_distance: "sampling" only.  Distances shorter than this are clamped
        so that a sample right next to a pixel cannot dominate it, which
        otherwise leaves a blotch around every sample point.  None: half the
        mean spacing between samples, sqrt(H*W/samples)/2.  0 disables it.
    preserve_chroma: run ACE on the luminance only and rescale the RGB
        triplet by the luminance change, keeping the colour of every pixel.
        Useful after a global white balance (e.g. retinex_with_adjust) to add
        ACE's local contrast without the desaturation of the per channel
        stretch.  This is an extension, not part of the paper.
    alpha: slope of the saturation function on intensities normalised to
        [0, 1] (the papers' notation; saturation at a difference of 1/alpha).
        Overrides slope/limit when given: alpha = 255 * slope / limit.
    scaling: dynamic tone reproduction scaling, one of "minmax" (default,
        linear stretch), "wp_gw" (R=0 maps to middle grey, white patch +
        grey world), "wp_gw_black" and "keep_gray" (Chambah et al. 2003, see
        _ace_scale).
    """
    if alpha is not None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        slope, limit = float(alpha), 255.0
    if slope <= 0 or limit <= 0:
        raise ValueError("slope and limit must be positive")
    if scaling not in ACE_SCALINGS:
        raise ValueError("scaling must be one of %r" % (ACE_SCALINGS,))
    planes = _planes(nimg)
    if not preserve_chroma:
        out = _ace_planes(planes, slope, limit, samples, method, levels, seed, workers, min_distance, scaling)
        return _to_image(out)
    luma = _luminance_plane(planes)
    luma8 = np.clip(np.rint(luma), 0, 255).astype(np.uint8)[None]
    luma_ace = _ace_planes(np.ascontiguousarray(luma8), slope, limit, samples, method,
                           levels, seed, workers, min_distance, scaling)[0].astype(np.float64)
    ratio = luma_ace / np.maximum(luma, 1.0)
    out = np.clip(np.rint(planes.astype(np.float64) * ratio[None]), 0, 255).astype(np.uint8)
    return _to_image(out)


# ---------------------------------------------------------------------------
# illuminant estimation: Minkowski norm family (Shades of Gray / Grey Edge)
# ---------------------------------------------------------------------------
def _gaussian_kernels(sigma):
    """1-D Gaussian and its first two derivatives, truncated at 3 sigma."""
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    g /= g.sum()
    g1 = -x / (sigma * sigma) * g
    g2 = (x * x / sigma ** 4 - 1.0 / (sigma * sigma)) * g
    return g, g1, g2


def _convolve_separable(f, ky, kx):
    """2-D convolution with separable kernels and reflected borders."""
    ry, rx = len(ky) // 2, len(kx) // 2
    padded = np.pad(f, ((ry, ry), (rx, rx)), mode="reflect")
    tmp = np.apply_along_axis(np.convolve, 0, padded, ky[::-1], mode="valid")
    return np.apply_along_axis(np.convolve, 1, tmp, kx[::-1], mode="valid")


def _dilate(mask, radius):
    """Binary dilation with a (2r+1)x(2r+1) square."""
    if radius <= 0:
        return mask
    out = mask.copy()
    for axis in (0, 1):
        acc = out.copy()
        for k in range(1, radius + 1):
            acc |= np.roll(out, k, axis=axis)
            acc |= np.roll(out, -k, axis=axis)
        out = acc
    return out


def _minkowski_norm(values, p):
    """(mean |v|^p)^(1/p); p=None or inf gives the maximum."""
    values = np.abs(values)
    if values.size == 0:
        return 1.0
    if p is None or np.isinf(p):
        return float(values.max())
    if p == 1:
        return float(values.mean())
    scale = values.max()
    if scale <= 0:
        return 0.0
    return float(scale * np.power(np.mean(np.power(values / scale, p)), 1.0 / p))


def _gains_from_illuminant(e):
    """Gains that map the estimated illuminant onto its G component."""
    e = np.asarray(e, dtype=np.float64)
    if e[1] <= 0:
        return np.ones(3)
    return np.array([_safe_ratio(e[1], e[0]), 1.0, _safe_ratio(e[1], e[2])])


def _valid_mask(nimg, saturation_threshold, dilate=0):
    """Pixels that are not clipped in any channel (optionally dilated)."""
    if saturation_threshold is None:
        return np.ones(nimg.shape[:2], dtype=bool)
    saturated = nimg.max(axis=2) >= saturation_threshold
    return ~_dilate(saturated, dilate)


def estimate_illuminant_grey_edge(nimg, order=1, p=6, sigma=2.0, saturation_threshold=255):
    """
    Illuminant colour estimate of the general grey world framework

        e_c  proportional to  ( mean | d^n/dx^n  (G_sigma * I_c) |^p )^(1/p)

    order=0 is Shades of Gray (and grey world for p=1, max-RGB for p=inf),
    order=1 / 2 are first / second order Grey Edge.  Returns an array (3,).
    """
    f = np.asarray(nimg).astype(np.float64)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array")
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    if order > 0 and sigma <= 0:
        raise ValueError("derivatives need sigma > 0")
    radius = int(np.ceil(3.0 * sigma)) if sigma > 0 else 0
    mask = _valid_mask(np.asarray(nimg), saturation_threshold, dilate=radius if order > 0 else 0)
    if order > 0 and radius > 0:
        # derivatives next to the border are dominated by the padding
        border = np.zeros_like(mask)
        border[radius:-radius or None, radius:-radius or None] = True
        mask &= border
    if not mask.any():
        mask = np.ones_like(mask)
    e = np.empty(3)
    if sigma > 0:
        g, g1, g2 = _gaussian_kernels(sigma)
    for c in range(3):
        ch = f[..., c]
        if order == 0:
            resp = _convolve_separable(ch, g, g) if sigma > 0 else ch
        elif order == 1:
            gx = _convolve_separable(ch, g, g1)
            gy = _convolve_separable(ch, g1, g)
            resp = np.hypot(gx, gy)
        else:
            gxx = _convolve_separable(ch, g, g2)
            gyy = _convolve_separable(ch, g2, g)
            gxy = _convolve_separable(ch, g1, g1)
            resp = np.sqrt(gxx * gxx + 4.0 * gxy * gxy + gyy * gyy)
        e[c] = _minkowski_norm(resp[mask], p)
    return e


def shades_of_gray(nimg, p=6, saturation_threshold=255):
    """
    G. Finlayson and E. Trezzi, "Shades of Gray and Colour Constancy",
    Color Imaging Conference, 2004.

    Grey world with the mean replaced by a Minkowski p-norm: p=1 is grey
    world, p=None (infinity) is max-RGB, p around 6 is the recommended value.
    Pixels clipped at ``saturation_threshold`` are ignored (None: keep all).
    """
    nimg = np.asarray(nimg)
    e = estimate_illuminant_grey_edge(nimg, order=0, p=p, sigma=0.0,
                                      saturation_threshold=saturation_threshold)
    return _apply_gains(nimg, _gains_from_illuminant(e))


def grey_edge(nimg, order=1, p=6, sigma=2.0, saturation_threshold=255):
    """
    J. van de Weijer, T. Gevers and A. Gijsenij, "Edge-Based Color Constancy",
    IEEE Trans. Image Processing 16(9), 2007.

    The illuminant is estimated from the Minkowski p-norm of the Gaussian
    derivatives (order 1 or 2, scale sigma) of each channel instead of the
    pixel values.  order=0 gives the general grey world (Shades of Gray with
    Gaussian smoothing).
    """
    nimg = np.asarray(nimg)
    e = estimate_illuminant_grey_edge(nimg, order=order, p=p, sigma=sigma,
                                      saturation_threshold=saturation_threshold)
    return _apply_gains(nimg, _gains_from_illuminant(e))


# ---------------------------------------------------------------------------
# illuminant estimation: PCA of bright and dark pixels (Cheng et al. 2014)
# ---------------------------------------------------------------------------
def estimate_illuminant_pca(nimg, fraction=0.035, saturation_threshold=255):
    """
    D. Cheng, D. K. Prasad and M. S. Brown, "Illuminant estimation for color
    constancy: why spatial-domain methods work and the role of the color
    distribution", JOSA A 31(5), 2014.

    Pixels are projected on the direction of the mean colour; the ``fraction``
    most distant pixels on either side (the brightest and darkest along that
    axis) are kept and the illuminant is the first principal component (no
    centering) of their colours.  Returns an array (3,).
    """
    nimg = np.asarray(nimg)
    f = nimg.reshape(-1, 3).astype(np.float64)
    mask = _valid_mask(nimg, saturation_threshold).ravel()
    if mask.any():
        f = f[mask]
    mean = f.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 0:
        return np.ones(3)
    proj = (f - mean) @ (mean / norm)
    k = max(int(round(len(f) * fraction)), 1)
    order = np.argsort(proj)
    sel = f[np.concatenate([order[:k], order[-k:]])]
    _, _, vt = np.linalg.svd(sel, full_matrices=False)
    e = np.abs(vt[0])
    return e if e.sum() > 0 else np.ones(3)


def pca_color_constancy(nimg, fraction=0.035, saturation_threshold=255):
    """White balance with the PCA based illuminant estimate of Cheng et al. 2014."""
    nimg = np.asarray(nimg)
    e = estimate_illuminant_pca(nimg, fraction=fraction, saturation_threshold=saturation_threshold)
    return _apply_gains(nimg, _gains_from_illuminant(e))


# ---------------------------------------------------------------------------
# multi scale retinex (with colour restoration)
# ---------------------------------------------------------------------------
def _gaussian_blur_fft(f, sigma):
    """Gaussian blur of a 2-D array with symmetric (mirrored) borders via FFT."""
    h, w = f.shape
    ph, pw = _fast_len(2 * h), _fast_len(2 * w)
    padded = np.pad(f, ((0, ph - h), (0, pw - w)), mode="symmetric")
    yy = np.minimum(np.arange(ph), ph - np.arange(ph)).astype(np.float64)
    xx = np.minimum(np.arange(pw), pw - np.arange(pw)).astype(np.float64)
    kernel = np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return np.fft.irfft2(np.fft.rfft2(padded) * np.fft.rfft2(kernel), s=(ph, pw))[:h, :w]


def _simplest_color_balance(f, clip):
    """Clip ``clip`` of the pixels at each end of every channel and stretch to [0, 255]."""
    out = np.empty(f.shape, dtype=np.uint8)
    for c in range(f.shape[2]):
        ch = f[..., c]
        lo, hi = np.percentile(ch, [100.0 * clip, 100.0 * (1.0 - clip)])
        if hi - lo <= 0:
            out[..., c] = np.clip(np.rint(ch), 0, 255)
        else:
            out[..., c] = np.clip(np.rint((ch - lo) * (255.0 / (hi - lo))), 0, 255)
    return out


def multi_scale_retinex(nimg, sigmas=(15, 80, 250), color_restoration=True,
                        alpha=125.0, beta=46.0, clip=0.01):
    """
    D. Jobson, Z. Rahman and G. Woodell, "A multiscale retinex for bridging the
    gap between color images and the human observation of scenes", IEEE Trans.
    Image Processing 6(7), 1997 (MSR / MSRCR), with the output scaling of
    A. Petro, C. Sbert and J.-M. Morel, "Multiscale Retinex", IPOL 2014.

    R_c = mean_k [ log I_c - log(G_sigma_k * I_c) ]
    MSRCR_c = beta * [ log(alpha I_c) - log(I_r + I_g + I_b) ] * R_c   (color_restoration)

    followed by the "simplest color balance": ``clip`` of the pixels at both
    ends of each channel are saturated and the rest stretched to [0, 255].
    """
    if not sigmas:
        raise ValueError("at least one scale is needed")
    if not 0 <= clip < 0.5:
        raise ValueError("clip must be in [0, 0.5)")
    f = np.asarray(nimg).astype(np.float64) + 1.0  # avoid log(0)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array")
    log_f = np.log(f)
    jobs = [(c, float(sigma)) for sigma in sigmas for c in range(3)]

    def blurred_log(job):
        c, sigma = job
        return np.log(_gaussian_blur_fft(f[..., c], sigma) + 1.0)

    # numpy's FFT releases the GIL, so the blurs run in parallel
    workers = _default_workers()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(blurred_log, jobs))
    else:
        results = [blurred_log(job) for job in jobs]
    msr = np.zeros_like(f)
    for (c, _), blurred in zip(jobs, results):
        msr[..., c] += log_f[..., c] - blurred
    msr /= len(sigmas)
    if color_restoration:
        restoration = beta * (np.log(alpha * f) - np.log(f.sum(axis=2, keepdims=True)))
        msr *= restoration
    return _simplest_color_balance(msr, clip)


# ---------------------------------------------------------------------------
# random spray retinex (Provenzi et al. 2007)
# ---------------------------------------------------------------------------
def random_spray_retinex(nimg, sprays=10, points=250, radius=None, seed=None, workers=None):
    """
    E. Provenzi, M. Fierro, A. Rizzi, L. De Carli, D. Gadia and D. Marini,
    "Random Spray Retinex", IEEE Trans. Image Processing 16(1), 2007.

    For every pixel ``sprays`` random sprays of ``points`` pixels are drawn
    around it with radial density 1/r up to ``radius`` (None: the image
    diagonal); the lightness is the average over the sprays of the pixel value
    divided by the maximum of the spray (per channel).  Cost grows with
    sprays * points * pixels; ``seed`` makes the result reproducible and
    ``workers`` sets the number of threads (None: all cores).
    """
    if sprays <= 0 or points <= 0:
        raise ValueError("sprays and points must be positive")
    planes = _planes(nimg)
    out = np.empty_like(planes)
    if seed is None:
        seed = int(np.random.randint(0, 2 ** 32, dtype=np.uint64))
    libcutil.calc_rsr(pointer(_rgbimage(planes)), pointer(_rgbimage(out)), int(sprays), int(points),
                      float(radius) if radius else 0.0, c_uint(int(seed) & 0xFFFFFFFF),
                      _spray_workers(workers))
    return _to_image(out)


def race(nimg, sprays=10, points=250, alpha=20.0, regularize=True, radius=None,
         seed=None, workers=None):
    """
    E. Provenzi, C. Gatta, M. Fierro and A. Rizzi, "A Spatially Variant
    White-Patch and Gray-World Method for Color Image Enhancement Driven by
    Local Contrast", IEEE Trans. PAMI 30(10), 2008.

    RACE fuses the local white patch of Random Spray Retinex with a spray
    version of ACE's local grey world: for every spray the lightness is
    1/2 * (I(i)/max(spray) + mean_j s~_alpha(I(i) - I(j))) with the slope
    function s~_alpha of eq. (10) (alpha=20 is the paper's stable choice),
    averaged over ``sprays`` sprays of ``points`` pixels (radial density 1/r,
    radius ``radius``, None: image diagonal).

    regularize=True gives sigma-RACE (eq. 23): every spray term is mixed
    with the original value through (2 sigma_k)^(sigma_min/sigma_max), where
    sigma_k is the standard deviation inside the spray and sigma_min/max the
    smallest/largest global channel standard deviation, so uniform areas keep
    their value instead of turning into chromatic noise.
    The paper linearises sRGB input first; this function works on the values
    it is given.
    """
    if sprays <= 0 or points <= 0:
        raise ValueError("sprays and points must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    planes = _planes(nimg)
    out = np.empty_like(planes)
    if seed is None:
        seed = int(np.random.randint(0, 2 ** 32, dtype=np.uint64))
    libcutil.calc_race(pointer(_rgbimage(planes)), pointer(_rgbimage(out)), int(sprays), int(points),
                       float(radius) if radius else 0.0, float(alpha), 1 if regularize else 0,
                       c_uint(int(seed) & 0xFFFFFFFF), _spray_workers(workers))
    return _to_image(out)


# ---------------------------------------------------------------------------
# grayness index (Qian et al. 2019)
# ---------------------------------------------------------------------------
def _log_kernel(size=5, sigma=0.5):
    """Laplacian of Gaussian filter as MATLAB's fspecial('log', size, sigma)."""
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    g /= g.sum()
    k = (x * x + y * y - 2.0 * sigma * sigma) / sigma ** 4 * g
    return k - k.mean()


def _convolve2d_same(f, kernel):
    """2-D 'same' convolution with reflected borders (small kernels)."""
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    padded = np.pad(f, ((ry, ry), (rx, rx)), mode="reflect")
    out = np.zeros_like(f)
    for i in range(kh):
        for j in range(kw):
            out += kernel[kh - 1 - i, kw - 1 - j] * padded[i:i + f.shape[0], j:j + f.shape[1]]
    return out


def _box_filter(f, size):
    k = np.ones((size, size)) / float(size * size)
    return _convolve2d_same(f, k)


def grayness_index(nimg, epsilon=1e-4, log_size=5, log_sigma=0.5, average_size=7):
    """
    Grayness index map of Y. Qian, J.-K. Kamarainen, J. Nikkanen and J. Matas,
    "On Finding Gray Pixels", CVPR 2019 (eq. 11, 12 and the 7x7 averaging).

    GI = || ( C{log I_R - log |I|}, C{log I_B - log |I|} ) ||  with
    |I| = I_R + I_G + I_B and C a Laplacian of Gaussian filter; pixels whose
    local contrast C{log I_c} is below ``epsilon`` in any channel carry no
    spatial cue and get an infinite index.  Smaller means greyer.
    """
    f = np.asarray(nimg).astype(np.float64)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array")
    f = f + 1.0  # avoid log(0); the paper assumes black-level corrected data
    kernel = _log_kernel(log_size, log_sigma)
    contrast = [_convolve2d_same(np.log(f[..., c]), kernel) for c in range(3)]
    lum = _convolve2d_same(np.log(f.sum(axis=2)), kernel)
    gi = np.hypot(contrast[0] - lum, contrast[2] - lum)
    no_cue = (np.abs(contrast[0]) < epsilon) | (np.abs(contrast[1]) < epsilon) | (np.abs(contrast[2]) < epsilon)
    gi = _box_filter(gi, average_size)
    gi[no_cue] = np.inf
    return gi


def estimate_illuminant_grayness_index(nimg, fraction=0.001, epsilon=1e-4, saturation_threshold=255):
    """
    Illuminant estimate of Qian et al. 2019: the mean colour of the
    ``fraction`` pixels with the smallest grayness index (0.1% in the paper).
    Returns an array (3,).
    """
    nimg = np.asarray(nimg)
    gi = grayness_index(nimg, epsilon=epsilon)
    valid = _valid_mask(nimg, saturation_threshold) & np.isfinite(gi)
    if not valid.any():
        return np.ones(3)
    flat = np.where(valid, gi, np.inf).ravel()
    k = max(int(round(valid.sum() * fraction)), 1)
    idx = np.argpartition(flat, k - 1)[:k]
    e = nimg.reshape(-1, 3)[idx].astype(np.float64).mean(axis=0)
    return e if e.sum() > 0 else np.ones(3)


def gray_pixel(nimg, fraction=0.001, epsilon=1e-4, saturation_threshold=255):
    """White balance with the grayness index illuminant estimate (Qian et al. 2019)."""
    nimg = np.asarray(nimg)
    e = estimate_illuminant_grayness_index(nimg, fraction=fraction, epsilon=epsilon,
                                           saturation_threshold=saturation_threshold)
    return _apply_gains(nimg, _gains_from_illuminant(e))


# ---------------------------------------------------------------------------
# weighted grey edge (Gijsenij, Gevers, van de Weijer 2012)
# ---------------------------------------------------------------------------
def _edge_weights(f, dx, dy, weighting):
    """
    Photometric edge weights of Gijsenij et al. 2012.

    f: (H, W, 3) image, dx/dy: (H, W, 3) Gaussian derivatives.
    specular: energy of the derivative along the white direction (1,1,1)/sqrt(3)
    shadow:   energy along the pixel's own colour direction
    material: energy orthogonal to both (shadow-shading-specular quasi-invariant)
    each divided by the total derivative energy.
    """
    total = np.sqrt((dx * dx).sum(axis=2) + (dy * dy).sum(axis=2))
    white = np.full(3, 1.0 / np.sqrt(3.0))
    if weighting == "specular":
        ox, oy = dx @ white, dy @ white
        energy = np.hypot(ox, oy)
    else:
        norm = np.linalg.norm(f, axis=2, keepdims=True)
        fhat = np.where(norm > 0, f / np.where(norm > 0, norm, 1.0), white)
        sx, sy = (dx * fhat).sum(axis=2), (dy * fhat).sum(axis=2)
        if weighting == "shadow":
            energy = np.hypot(sx, sy)
        elif weighting == "material":
            b = np.cross(fhat, white)
            bnorm = np.linalg.norm(b, axis=2, keepdims=True)
            b = np.where(bnorm > 0, b / np.where(bnorm > 0, bnorm, 1.0), 0.0)
            hx, hy = (dx * b).sum(axis=2), (dy * b).sum(axis=2)
            rest = total * total - hx * hx - hy * hy
            energy = np.sqrt(np.maximum(rest, 0.0))
        else:
            raise ValueError("weighting must be 'specular', 'shadow' or 'material'")
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(total > 0, energy / np.where(total > 0, total, 1.0), 0.0)
    return np.clip(w, 0.0, 1.0)


def estimate_illuminant_weighted_grey_edge(nimg, weighting="specular", kappa=10.0, p=1, sigma=2.0,
                                           iterations=10, tolerance=0.1, saturation_threshold=255):
    """
    A. Gijsenij, T. Gevers and J. van de Weijer, "Improving Color Constancy by
    Photometric Edge Weighting", IEEE Trans. PAMI 34(5), 2012.

    Weighted grey edge, eq. (7):  e_c ~ ( mean | w(f)^kappa  |grad f_c| |^p )^(1/p)
    with w the specular (default), shadow or material edge weight, applied
    iteratively (Algorithm 1): the image is corrected with the current
    estimate, weights and estimate are recomputed and the estimates
    multiplied, until the estimate moves less than ``tolerance`` degrees or
    ``iterations`` passes.  kappa=10, p=1, sigma=2 are the defaults of the
    authors' reference implementation.  Returns an array (3,).
    """
    nimg = np.asarray(nimg)
    f0 = nimg.astype(np.float64)
    if f0.ndim != 3 or f0.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(np.ceil(3.0 * sigma))
    mask = _valid_mask(nimg, saturation_threshold, dilate=radius)
    border = np.zeros_like(mask)
    border[radius:-radius or None, radius:-radius or None] = True
    mask &= border
    if not mask.any():
        mask = np.ones_like(mask)
    g, g1, _ = _gaussian_kernels(sigma)
    estimate = np.ones(3)
    for _ in range(max(int(iterations), 1)):
        f = f0 / (estimate / estimate.mean())
        dx = np.stack([_convolve_separable(f[..., c], g, g1) for c in range(3)], axis=-1)
        dy = np.stack([_convolve_separable(f[..., c], g1, g) for c in range(3)], axis=-1)
        w = _edge_weights(f, dx, dy, weighting) ** kappa
        grad = np.hypot(dx, dy)
        step = np.array([_minkowski_norm((w * grad[..., c])[mask], p) for c in range(3)])
        if not np.all(step > 0):
            break
        new = estimate * step
        new /= np.linalg.norm(new)
        cos = np.clip(np.dot(new, estimate / np.linalg.norm(estimate)), -1.0, 1.0)
        estimate = new
        if np.degrees(np.arccos(cos)) < tolerance:
            break
    return estimate


def weighted_grey_edge(nimg, weighting="specular", kappa=10.0, p=1, sigma=2.0, iterations=10,
                       tolerance=0.1, saturation_threshold=255):
    """White balance with the iterative weighted grey edge estimate (Gijsenij et al. 2012)."""
    nimg = np.asarray(nimg)
    e = estimate_illuminant_weighted_grey_edge(nimg, weighting=weighting, kappa=kappa, p=p, sigma=sigma,
                                               iterations=iterations, tolerance=tolerance,
                                               saturation_threshold=saturation_threshold)
    return _apply_gains(nimg, _gains_from_illuminant(e))


# ---------------------------------------------------------------------------
# colour space helpers (sRGB D65 <-> CIELAB)
# ---------------------------------------------------------------------------
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_D65 = np.array([0.95047, 1.0, 1.08883])


def _srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    s = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return s * 255.0


def rgb_to_lab(nimg):
    """sRGB (uint8, D65) -> CIELAB float array of the same shape."""
    xyz = _srgb_to_linear(nimg) @ _RGB2XYZ.T / _D65
    f = np.where(xyz > 216.0 / 24389.0, np.cbrt(xyz), (24389.0 / 27.0 * xyz + 16.0) / 116.0)
    lab = np.empty(xyz.shape)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb(lab):
    """CIELAB -> sRGB uint8 (values outside the gamut are clipped)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 216.0 / 24389.0, f ** 3, (116.0 * f - 16.0) * 27.0 / 24389.0) * _D65
    rgb = _linear_to_srgb(xyz @ _XYZ2RGB.T)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# simple global tools used by the film restoration papers
# ---------------------------------------------------------------------------
def simplest_color_balance(nimg, clip=0.01):
    """
    N. Limare, J.-L. Lisani, J.-M. Morel, A. B. Petro and C. Sbert, "Simplest
    Color Balance", Image Processing On Line, 2011.

    Saturate ``clip`` of the pixels at both ends of every channel and stretch
    the rest linearly to [0, 255] (what image editors call auto levels or
    white balance by percentile).
    """
    if not 0 <= clip < 0.5:
        raise ValueError("clip must be in [0, 0.5)")
    nimg = np.asarray(nimg)
    return _simplest_color_balance(nimg.astype(np.float64), clip)


def complementary_cast_removal(nimg, k):
    """
    O.-M. Machidon and M. Ivanovici, "Digital color restoration for the
    preservation of reversal film heritage", Journal of Cultural Heritage 33,
    2018, eq. (1):  g = f + k * complement(mean(f)),  complement(c) = 255 - c.

    The complementary of the average colour is added to every pixel with
    weight k in [0, 1).  The paper picks k per film stock by minimising the
    distance of a colour chart slide to its reference colours (k = 0.28 and
    0.36 for its two stocks); without a chart k has to be chosen by eye.
    Note that the mapping also brightens the image (the paper follows it
    with a colour correction / contrast step).
    """
    f = np.asarray(nimg).astype(np.float64)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array")
    mean = f.reshape(-1, 3).mean(axis=0)
    comp = 255.0 - mean
    if not 0 <= k < 1:
        raise ValueError("k must be in [0, 1)")
    return np.clip(np.rint(f + k * comp), 0, 255).astype(np.uint8)


def color_chart_transform(acquired, reference):
    """
    Least squares 3x3 colour transform T with reference = T @ acquired for
    matching sets of colours (rows of RGB), as used by Machidon & Ivanovici
    2018, eq. (4), to correct every slide of a film stock with the colours of
    its colour chart slide.  Returns T (3, 3).
    """
    a = np.asarray(acquired, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if a.shape != r.shape or len(a) < 3:
        raise ValueError("need at least three matching colours")
    t, _, _, _ = np.linalg.lstsq(a, r, rcond=None)
    return t.T


def apply_color_transform(nimg, transform):
    """Apply a 3x3 linear colour transform (see color_chart_transform) to an image."""
    f = np.asarray(nimg).astype(np.float64)
    out = f.reshape(-1, 3) @ np.asarray(transform, dtype=np.float64).T
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(f.shape)


def saturation_enhancement(nimg, factor=1.5):
    """
    Non-uniform saturation enhancement in the spirit of Chambah, Besserer and
    Courtellemont (2002) as summarised by Chambah et al. 2003: the cloud of
    (a*, b*) chroma values is stretched about its own centre along its
    principal axes, so the real colours of a faded image are revived without
    increasing the colour cast (the mean chroma is unchanged).  The 2003
    paper gives no rule for the amount of stretch; ``factor`` (scalar or one
    value per principal axis) is applied to the semi-axes.
    """
    lab = rgb_to_lab(np.asarray(nimg))
    ab = lab[..., 1:].reshape(-1, 2)
    center = ab.mean(axis=0)
    centered = ab - center
    cov = centered.T @ centered / max(len(ab) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    factors = np.broadcast_to(np.asarray(factor, dtype=np.float64), (2,))
    if np.any(factors <= 0):
        raise ValueError("factor must be positive")
    stretched = (centered @ vecs) * factors @ vecs.T + center
    lab[..., 1:] = stretched.reshape(lab.shape[:2] + (2,))
    return lab_to_rgb(lab)


def film_restoration_ace(nimg, saturation=0.2, scaling="keep_gray", enhance_saturation=None,
                         **ace_kwargs):
    """
    M. Chambah, A. Rizzi, C. Gatta, B. Besserer and D. Marini, "Perceptual
    approach for unsupervised digital color restoration of cinematographic
    archives", SPIE Electronic Imaging 5008, 2003.

    ACE tuned for faded film: a saturation shape function saturating at a
    normalised difference of ``saturation`` (0.2 in the paper; smaller values
    add noise), the "WP+GW (black)" scaling with "keep original gray" (the
    channels are centred on the original mean grey level instead of middle
    grey, preserving the key of the picture) and,
    optionally, the non-uniform saturation enhancement of the colours before
    ACE (``enhance_saturation`` = stretch factor, see saturation_enhancement;
    the paper's own rule for the amount is not given).  Remaining keyword
    arguments go to automatic_color_equalization (method, samples, seed, ...).
    """
    if saturation <= 0:
        raise ValueError("saturation must be positive")
    img = np.asarray(nimg)
    if enhance_saturation is not None:
        img = saturation_enhancement(img, enhance_saturation)
    return automatic_color_equalization(img, alpha=1.0 / saturation, scaling=scaling, **ace_kwargs)
