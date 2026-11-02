# -*- coding: utf-8 -*-
import os
from unittest import TestCase

import numpy as np
from PIL import Image

import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(HERE, "test_image.jpg")

ALL_ALGORITHMS = [
    cca.stretch,
    cca.grey_world,
    cca.retinex,
    cca.max_white,
    cca.retinex_with_adjust,
    cca.standard_deviation_weighted_grey_world,
    cca.standard_deviation_weighted_grey_world_python,
    cca.standard_deviation_and_luminance_weighted_gray_world,
    cca.standard_deviation_and_luminance_weighted_gray_world_python,
    cca.luminance_weighted_gray_world,
    cca.luminance_weighted_gray_world_python,
    cca.automatic_color_equalization,
]


def cast_image(rng, shape=(60, 80), gains=(1.3, 1.0, 0.7)):
    """Random scene with a known colour cast applied on top of it."""
    grey = rng.integers(20, 200, shape).astype(np.float64)
    rgb = np.stack([grey * g for g in gains], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def ace_brute_force(img, slope, limit):
    """Literal O(N^2) evaluation of the ACE chromatic/spatial adjustment."""
    h, w, _ = img.shape
    f = img.astype(np.float64)
    ys, xs = np.mgrid[0:h, 0:w]
    ys, xs = ys.ravel(), xs.ravel()
    d = np.sqrt((ys[:, None] - ys[None, :]) ** 2 + (xs[:, None] - xs[None, :]) ** 2)
    with np.errstate(divide="ignore"):
        invd = np.where(d > 0, 1.0 / d, 0.0)
    scores = np.empty((3, h, w))
    for c in range(3):
        v = f[..., c].ravel()
        sat = np.clip((v[:, None] - v[None, :]) * slope, -limit, limit) / limit
        scores[c] = ((sat * invd).sum(axis=1) / invd.sum(axis=1)).reshape(h, w)
    return scores


class CCTestCase(TestCase):
    def setUp(self):
        self.img = from_pil(Image.open(TEST_IMAGE))
        self.rng = np.random.default_rng(1234)

    def test_all_run_and_keep_input(self):
        for func in ALL_ALGORITHMS:
            src = self.img.copy()
            out = func(src)
            self.assertTrue(np.array_equal(src, self.img), "%s modified its input" % func.__name__)
            self.assertEqual(out.shape, self.img.shape, func.__name__)
            self.assertEqual(out.dtype, np.uint8, func.__name__)
            to_pil(out)

    def test_odd_sizes_and_block_sizes(self):
        # sizes that are not multiples of the sub-block size used to read past
        # the pixel buffers in the C code
        img = self.rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        for func in ALL_ALGORITHMS:
            out = func(img)
            self.assertEqual(out.shape, img.shape)
        for func in (cca.standard_deviation_weighted_grey_world,
                     cca.standard_deviation_and_luminance_weighted_gray_world,
                     cca.luminance_weighted_gray_world):
            self.assertEqual(func(img, 7, 13).shape, img.shape)
            self.assertEqual(func(img, 200, 200).shape, img.shape)

    def test_grey_world_removes_cast(self):
        img = cast_image(self.rng)
        out = cca.grey_world(img).reshape(-1, 3).mean(axis=0)
        self.assertLess(np.ptp(out), 1.0)

    def test_max_white_and_stretch(self):
        img = cast_image(self.rng)
        self.assertTrue((cca.max_white(img).reshape(-1, 3).max(axis=0) == 255).all())
        out = cca.stretch(img).reshape(-1, 3)
        self.assertTrue((out.max(axis=0) == 255).all())
        self.assertTrue((out.min(axis=0) == 0).all())
        # 16 bit input scales to 65535, not to 255
        img16 = (img.astype(np.uint16) * 200)
        out16 = cca.max_white(img16)
        self.assertEqual(out16.dtype, np.uint16)
        self.assertTrue((out16.reshape(-1, 3).max(axis=0) == 65535).all())

    def test_retinex_matches_green_maximum(self):
        img = cast_image(self.rng)
        mx = cca.retinex(img).reshape(-1, 3).max(axis=0)
        self.assertEqual(mx[0], mx[1])
        self.assertEqual(mx[2], mx[1])

    def test_retinex_with_adjust_satisfies_both_constraints(self):
        img = cast_image(self.rng, gains=(1.2, 1.0, 0.8))
        out = cca.retinex_with_adjust(img).reshape(-1, 3).astype(np.float64)
        # quadratic mapping fulfils grey world (mean) and retinex (max) w.r.t. G
        self.assertLess(abs(out[:, 0].mean() - out[:, 1].mean()), 1.0)
        self.assertLess(abs(out[:, 2].mean() - out[:, 1].mean()), 1.0)
        self.assertLessEqual(abs(out[:, 0].max() - out[:, 1].max()), 1)
        self.assertLessEqual(abs(out[:, 2].max() - out[:, 1].max()), 1)

    def test_retinex_with_adjust_never_wraps_around(self):
        # a channel whose quadratic fit overshoots 255 must be clipped, not
        # wrapped (the old code clipped the coefficients instead of the result)
        img = self.rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
        img[..., 2] = (img[..., 2] // 4).astype(np.uint8)
        img[0, 0, 2] = 255
        out = cca.retinex_with_adjust(img)
        self.assertEqual(out.dtype, np.uint8)
        f = img.astype(np.float64)
        mu, nu = cca._quadratic_fit(f[..., 2], f[..., 1].sum(), f[..., 1].max())
        expected = np.clip(np.rint(mu * f[..., 2] ** 2 + nu * f[..., 2]), 0, 255)
        self.assertTrue(np.array_equal(out[..., 2], expected.astype(np.uint8)))

    def test_block_algorithms_c_matches_numpy(self):
        for img in (self.img, self.rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)):
            for sw, sh in ((20, 20), (7, 13), (500, 500)):
                for func, lw, sd in ((cca.libcutil.calc_sdwgw, False, True),
                                     (cca.libcutil.calc_sdlwgw, True, True),
                                     (cca.libcutil.calc_lwgw, True, False)):
                    g_c = cca._c_gains(func, img, sw, sh)
                    g_py = cca._block_gains(img, sw, sh, lw, sd)
                    np.testing.assert_allclose(g_c, g_py, rtol=1e-9, atol=1e-12)

    def test_block_algorithms_remove_cast(self):
        img = cast_image(self.rng, shape=(120, 160))
        for func in (cca.standard_deviation_weighted_grey_world,
                     cca.standard_deviation_and_luminance_weighted_gray_world,
                     cca.luminance_weighted_gray_world):
            out = func(img, 20, 20).reshape(-1, 3).mean(axis=0)
            self.assertLess(np.ptp(out), 1.0, func.__name__)

    def test_block_statistics_follow_the_paper(self):
        # explicit per-block loop of Lam et al. 2004 for the SDWGW gains
        img = self.rng.integers(0, 256, (50, 70, 3), dtype=np.uint8)
        f = img.astype(np.float64)
        means, stds = [], []
        for y in range(0, 50, 20):
            for x in range(0, 70, 20):
                blk = f[y:y + 20, x:x + 20].reshape(-1, 3)
                means.append(blk.mean(axis=0))
                stds.append(blk.std(axis=0, ddof=1))
        means, stds = np.array(means), np.array(stds)
        sdwa = ((stds / stds.sum(axis=0)) * means).sum(axis=0)
        expected = sdwa.mean() / sdwa
        np.testing.assert_allclose(cca._c_gains(cca.libcutil.calc_sdwgw, img, 20, 20), expected, rtol=1e-9)

    def test_luminance_weight(self):
        self.assertEqual(cca._triangular_weight(0.0), 0.0)
        self.assertEqual(cca._triangular_weight(160.0), 1.0)
        self.assertEqual(cca._triangular_weight(255.0), 0.0)
        self.assertAlmostEqual(float(cca._triangular_weight(80.0)), 0.5)
        y = cca._luminance(np.array([[0, 255, 0]], dtype=np.float64))
        self.assertAlmostEqual(float(y[0]), 0.586611 * 255)

    def test_flat_images_do_not_crash(self):
        flat = np.full((30, 40, 3), 90, dtype=np.uint8)
        black = np.zeros((30, 40, 3), dtype=np.uint8)
        for func in ALL_ALGORITHMS:
            for img in (flat, black):
                out = func(img)
                self.assertFalse(np.isnan(out.astype(np.float64)).any(), func.__name__)
                self.assertEqual(out.dtype, np.uint8)

    def test_ace_fft_matches_brute_force(self):
        img = np.asarray(Image.fromarray(self.img).resize((48, 36), Image.BILINEAR))
        for slope, limit in ((10, 1000), (5, 1000), (3, 255)):
            ref = cca._ace_scale(ace_brute_force(img, slope, limit))
            exact = cca._ace_scale(cca._ace_fft(cca._planes(img), slope, limit, 256))
            self.assertLessEqual(np.abs(exact.astype(int) - ref.astype(int)).max(), 1)
            approx = cca._ace_scale(cca._ace_fft(cca._planes(img), slope, limit, 16))
            self.assertLessEqual(np.abs(approx.astype(int) - ref.astype(int)).max(), 2)

    def test_ace_methods(self):
        img = np.asarray(Image.fromarray(self.img).resize((64, 48), Image.BILINEAR))
        fft1 = cca.automatic_color_equalization(img)
        fft2 = cca.automatic_color_equalization(img, workers=1)
        self.assertTrue(np.array_equal(fft1, fft2))
        s1 = cca.automatic_color_equalization(img, method="sampling", seed=7, samples=300)
        s2 = cca.automatic_color_equalization(img, method="sampling", seed=7, samples=300)
        self.assertTrue(np.array_equal(s1, s2))
        # sampling approximates the exact sum
        big = cca.automatic_color_equalization(img, method="sampling", seed=7, samples=3000)
        self.assertLess(np.abs(big.astype(int) - fft1.astype(int)).mean(), 4.0)
        with self.assertRaises(ValueError):
            cca.automatic_color_equalization(img, method="nope")

    def test_ace_increases_contrast_and_removes_cast(self):
        img = cast_image(self.rng, shape=(40, 60))
        out = cca.automatic_color_equalization(img).reshape(-1, 3)
        self.assertTrue((out.max(axis=0) == 255).all())
        self.assertTrue((out.min(axis=0) == 0).all())
        self.assertLess(np.ptp(out.mean(axis=0)), 2.0)
