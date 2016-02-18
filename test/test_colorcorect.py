# -*- coding: utf-8 -*-
import sys
from PIL import Image
from colorcorrect.algorithm import stretch, grey_world, retinex, retinex_with_adjust, max_white
from colorcorrect.algorithm import standard_deviation_weighted_grey_world
from colorcorrect.algorithm import standard_deviation_and_luminance_weighted_gray_world
from colorcorrect.algorithm import automatic_color_equalization
from colorcorrect.algorithm import luminance_weighted_gray_world
from colorcorrect.util import from_pil, to_pil


if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    # img.show()
    to_pil(stretch(from_pil(img)))
    to_pil(grey_world(from_pil(img)))
    to_pil(retinex(from_pil(img)))
    to_pil(max_white(from_pil(img)))
    to_pil(retinex_with_adjust(retinex(from_pil(img))))
    to_pil(standard_deviation_weighted_grey_world(from_pil(img), 20, 20))
    to_pil(standard_deviation_and_luminance_weighted_gray_world(from_pil(img), 20, 20))
    to_pil(luminance_weighted_gray_world(from_pil(img), 20, 20))
    to_pil(automatic_color_equalization(from_pil(img)))
