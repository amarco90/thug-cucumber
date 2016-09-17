__author__ = 'tbeltramelli'

import cv2


class Utils:
    @staticmethod
    def get_image(path):
        return cv2.imread(path)

    @staticmethod
    def save_image(img, path):
        return cv2.imwrite(path, img)

    @staticmethod
    def resize(img, max_size):
        height, width, depth = img.shape

        ratio = float(width) / float(height)

        if width > height:
            height = int(max_size / ratio)
            width = int(max_size)
        else:
            width = int(max_size * ratio)
            height = int(max_size)

        return cv2.resize(img, (width, height))

    @staticmethod
    def show(*images):
        for i, img in enumerate(images):
            cv2.namedWindow(("image %d" % i), cv2.WINDOW_AUTOSIZE)
            cv2.imshow(("image %d" % i), img)
            cv2.moveWindow(("image %d" % i), 0, 0)
            cv2.waitKey(0)
            cv2.destroyWindow(("image %d" % i))
