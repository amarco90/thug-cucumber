__author__ = 'tbeltramelli'

import cv2
import numpy as np


class Filtering:
    @staticmethod
    def get_gray_scale_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def apply_box_filter(image, n=3):
        box = np.ones((n, n), np.float32) / (n * n)

        return cv2.filter2D(image, -1, box)

    @staticmethod
    def sharpen(image, n=5):
        box = np.zeros((3, 3), np.float32)
        box[0][1] = -1
        box[1][0] = -1
        box[1][1] = n
        box[1][2] = -1
        box[2][1] = -1

        return cv2.filter2D(image, -1, box)

    @staticmethod
    def negative(image):
        image = (255 - image)
        return image
