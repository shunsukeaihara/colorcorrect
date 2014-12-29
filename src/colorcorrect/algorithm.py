# -*- coding: utf-8 -*-
import numpy as np
from ctypes import POINTER
from ctypes import pointer
from ctypes import Structure
from ctypes import c_ubyte
from ctypes import c_int
from ctypes import c_double
from ctypes import c_void_p

import os
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
libcutil.calc_ace.argtypes = [POINTER(RGBImage), c_int, c_double, c_double]


def stretch_pre(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0] - nimg[0].min(), 0)
    nimg[1] = np.maximum(nimg[1] - nimg[1].min(), 0)
    nimg[2] = np.maximum(nimg[2] - nimg[2].min(), 0)
    return nimg.transpose(1, 2, 0)


def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0] * (mu_g / np.average(nimg[0])), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / np.average(nimg[2])), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def max_white(nimg):
    if nimg.dtype == np.uint8:
        brightest = float(2 ** 8)
    elif nimg.dtype == np.uint16:
        brightest = float(2 ** 16)
    elif nimg.dtype == np.uint32:
        brightest = float(2 ** 32)
    else:
        brightest == float(2 ** 8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest / float(nimg[0].max())), 255)
    nimg[1] = np.minimum(nimg[1] * (brightest / float(nimg[1].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (brightest / float(nimg[2].max())), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def stretch(nimg):
    return max_white(stretch_pre(nimg))


def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0] * (mu_g / float(nimg[0].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / float(nimg[2].max())), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0] ** 2)
    max_r = nimg[0].max()
    max_r2 = max_r ** 2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2, sum_r], [max_r2, max_r]]),
                                  np.array([sum_g, max_g]))
    nimg[0] = np.minimum((nimg[0] ** 2) * coefficient[0] + nimg[0] * coefficient[1], 255)
    sum_b = np.sum(nimg[2])
    sum_b2 = np.sum(nimg[2] ** 2)
    max_b = nimg[2].max()
    max_b2 = max_b ** 2
    coefficient = np.minimum(np.linalg.solve(np.array([[sum_b2, sum_b], [max_b2, max_b]]),
                                             np.array([sum_g, max_g])), 255)
    nimg[2] = (nimg[2] ** 2) * coefficient[0] + nimg[2] * coefficient[1]
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))


def standard_deviation_weighted_grey_world_python(nimg, subwidth=20, subheight=20):
    nimg = nimg.astype(np.uint32)
    height, width, ch = nimg.shape
    strides = nimg.itemsize * np.array([width * subheight, subwidth, width, 3, 1])
    shape = (height / subheight, width / subwidth, subheight, subwidth, 3)
    blocks = np.lib.stride_tricks.as_strided(nimg, shape=shape, strides=strides)
    y, x = blocks.shape[:2]
    std_r = np.zeros([y, x], dtype=np.float16)
    std_g = np.zeros([y, x], dtype=np.float16)
    std_b = np.zeros([y, x], dtype=np.float16)
    std_r_sum = 0.0
    std_g_sum = 0.0
    std_b_sum = 0.0
    for i in xrange(y):
        for j in xrange(x):
            subblock = blocks[i, j]
            subb = subblock.transpose(2, 0, 1)
            std_r[i, j] = np.std(subb[0])
            std_g[i, j] = np.std(subb[1])
            std_b[i, j] = np.std(subb[2])
            std_r_sum += std_r[i, j]
            std_g_sum += std_g[i, j]
            std_b_sum += std_b[i, j]
    sdwa_r = 0.0
    sdwa_g = 0.0
    sdwa_b = 0.0
    for i in xrange(y):
        for j in xrange(x):
            subblock = blocks[i, j]
            subb = subblock.transpose(2, 0, 1)
            mean_r = np.mean(subb[0])
            mean_g = np.mean(subb[1])
            mean_b = np.mean(subb[2])
            sdwa_r += (std_r[i, j] / std_r_sum) * mean_r
            sdwa_g += (std_g[i, j] / std_g_sum) * mean_g
            sdwa_b += (std_b[i, j] / std_b_sum) * mean_b
    sdwa_avg = (sdwa_r + sdwa_g + sdwa_b) / 3.0
    gain_r = sdwa_avg / sdwa_r
    gain_g = sdwa_avg / sdwa_g
    gain_b = sdwa_avg / sdwa_b
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.minimum(nimg[0] * gain_r, 255)
    nimg[1] = np.minimum(nimg[1] * gain_g, 255)
    nimg[2] = np.minimum(nimg[2] * gain_b, 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def standard_deviation_weighted_grey_world(nimg, subwidth=20, subheight=20):
    nimg = nimg.transpose(2, 0, 1)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)
    img = RGBImage(nimg.shape[2],
                   nimg.shape[1],
                   nimg[0].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[1].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[2].ctypes.data_as(POINTER(c_ubyte)))
    gains = c_double * 3
    ret = libcutil.calc_sdwgw(pointer(img), subwidth, subheight)
    gains = np.ctypeslib.as_array(gains.from_address(ret))
    gains = gains.copy()
    libcutil.delete_doubleptr(ret)
    nimg = nimg.astype(np.uint32)
    nimg[0] = np.minimum(nimg[0] * gains[0], 255)
    nimg[1] = np.minimum(nimg[1] * gains[1], 255)
    nimg[2] = np.minimum(nimg[2] * gains[2], 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def standard_deviation_and_luminance_weighted_gray_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)
    img = RGBImage(nimg.shape[2],
                   nimg.shape[1],
                   nimg[0].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[1].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[2].ctypes.data_as(POINTER(c_ubyte)))
    gains = c_double * 3
    ret = libcutil.calc_sdlwgw(pointer(img), subwidth, subheight)
    gains = np.ctypeslib.as_array(gains.from_address(ret))
    gains = gains.copy()
    libcutil.delete_doubleptr(ret)
    nimg = nimg.astype(np.uint32)
    nimg[0] = np.minimum(nimg[0] * gains[0], 255)
    nimg[1] = np.minimum(nimg[1] * gains[1], 255)
    nimg[2] = np.minimum(nimg[2] * gains[2], 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def luminance_weighted_gray_world(nimg, subwidth=20, subheight=20):
    """
    AUTOMATIC WHITE BALANCING USING LUMINANCE COMPONENT AND STANDARD DEVIATION OF RGB COMPONENTS
    http://www.ece.umassd.edu/faculty/acosta/icassp/icassp_2004/pdfs/0300493.pdf
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)
    img = RGBImage(nimg.shape[2],
                   nimg.shape[1],
                   nimg[0].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[1].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[2].ctypes.data_as(POINTER(c_ubyte)))
    gains = c_double * 3
    ret = libcutil.calc_lwgw(pointer(img), subwidth, subheight)
    gains = np.ctypeslib.as_array(gains.from_address(ret))
    gains = gains.copy()
    libcutil.delete_doubleptr(ret)
    nimg = nimg.astype(np.uint32)
    nimg[0] = np.minimum(nimg[0] * gains[0], 255)
    nimg[1] = np.minimum(nimg[1] * gains[1], 255)
    nimg[2] = np.minimum(nimg[2] * gains[2], 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500):
    """
    A. Rizzi, C. Gatta and D. Marini, "A new algorithm for unsupervised global and local color correction.",
    Pattern Recognition Letters, vol. 24, no. 11, 2003.
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)
    img = RGBImage(nimg.shape[2],
                   nimg.shape[1],
                   nimg[0].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[1].ctypes.data_as(POINTER(c_ubyte)),
                   nimg[2].ctypes.data_as(POINTER(c_ubyte)))
    libcutil.calc_ace(pointer(img), samples, slope, limit)
    return nimg.transpose(1, 2, 0)
