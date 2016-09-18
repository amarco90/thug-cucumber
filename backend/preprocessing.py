__author__ = 'tbeltramelli'

import base64
import cv2
import numpy as np
import re


class Preprocessing:
    @staticmethod
    def preprocess(base64img):
        path = "temp.png"
        with open(path, "wb") as img_file:
            b64 = re.sub('^data:image/.+;base64,', '', base64img)
            img_file.write(base64.decodestring(b64))

        img = cv2.imread(path)
        img = Preprocessing.get_gray_scale_rgb(img)
        img = Preprocessing.sharpen(img, 5)
        thr, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(path, img)

        with open(path, "rb") as img_file:
            base64img = base64.b64encode(img_file.read())
        return base64img

    @staticmethod
    def get_gray_scale_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def sharpen(image, n=5):
        box = np.zeros((3, 3), np.float32)
        box[0][1] = -1
        box[1][0] = -1
        box[1][1] = n
        box[1][2] = -1
        box[2][1] = -1

        return cv2.filter2D(image, -1, box)
