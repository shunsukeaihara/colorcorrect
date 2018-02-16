from PIL import Image
from colorcorrect.algorithm import stretch, grey_world, retinex, retinex_with_adjust, max_white
from colorcorrect.algorithm import standard_deviation_weighted_grey_world
from colorcorrect.algorithm import standard_deviation_and_luminance_weighted_gray_world
from colorcorrect.algorithm import automatic_color_equalization
from colorcorrect.algorithm import luminance_weighted_gray_world
from colorcorrect.util import from_pil, to_pil
from unittest import TestCase


class CCTestCase(TestCase):
    def setUp(self):
        self.img = Image.open("test/test_image.jpg")

    def tearDown(self):
        pass

    def test_all(self):
        to_pil(stretch(from_pil(self.img)))
        to_pil(grey_world(from_pil(self.img)))
        to_pil(retinex(from_pil(self.img)))
        to_pil(max_white(from_pil(self.img)))
        to_pil(retinex_with_adjust(retinex(from_pil(self.img))))
        to_pil(standard_deviation_weighted_grey_world(from_pil(self.img), 20, 20))
        to_pil(standard_deviation_and_luminance_weighted_gray_world(from_pil(self.img), 20, 20))
        to_pil(luminance_weighted_gray_world(from_pil(self.img), 20, 20))
        to_pil(automatic_color_equalization(from_pil(self.img)))
