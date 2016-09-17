#!/usr/bin/env python
__author__ = 'tbeltramelli'

import argparse
from modules.Utils import *
from modules.Filtering import *


def main():
    images = ["assets/IMG_2560.jpg", "assets/IMG_0258.jpg", "assets/IMG_0257.jpg", "assets/IMG_0256.jpg",
              "assets/IMG_0253.jpg", "assets/IMG_20160916_220816.jpg"]

    for image_path in images:
        get_url_segment(image_path)


def get_url_segment(image_path):
    original_image = Utils.get_image(image_path)

    kernels = ["assets/kernel.png", "assets/kernel2.png", "assets/kernel3.png"]

    result = search_for_template(original_image, kernels)
    if result is None:
        result = search_for_template(original_image, kernels, use_negative=True)

    if result is not None:
        Utils.show(Utils.resize(original_image, 600))
        Utils.show(result)

    return result


def search_for_template(original_image, kernels, use_negative=False):
    best = 0
    url_image = None

    for kernel_path in kernels:
        kernel = Utils.get_image(kernel_path)
        kernel = Filtering.get_gray_scale_rgb(kernel)
        if use_negative:
            kernel = Filtering.negative(kernel)

        result, score = detect_url(original_image, kernel)

        if result is None:
            continue

        if score > best:
            best = score
            url_image = result

    if url_image is not None:
        return url_image

    return None


def detect_url(original_image, template, score_threshold=0.4, to_show=False):
    template_width, template_height = template.shape[::-1]

    color_image = np.copy(original_image)

    scores = []
    coordinates = []
    for i in range(0, 5):
        height, width, depth = color_image.shape
        target_width = width / 1.25
        image, color_image = preprocess(color_image, target_width)

        response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
        top_left = max_loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

        scores.append(np.amax(response))
        coordinates.append((top_left, bottom_right, target_width))

    best_score = np.max(scores)
    if best_score < score_threshold:
        return None, best_score

    best_candidate = coordinates[np.argmax(scores)]

    result, img = preprocess(np.copy(original_image), best_candidate[2])
    result_height, result_width = result.shape

    sample_width = best_candidate[1][0] - best_candidate[0][0]
    sample_height = best_candidate[1][1] - best_candidate[0][1]

    start_x = best_candidate[0][0] - sample_width
    start_y = best_candidate[0][1] - (sample_height / 2)
    end_x = result_width - sample_width
    end_y = best_candidate[1][1] + (sample_height / 2)

    resulting_img = np.zeros((result_height, result_width), np.float32)
    resulting_img[start_y:end_y, start_x:end_x] = result[start_y:end_y, start_x:end_x]

    if to_show:
        image_to_show = np.copy(original_image)
        image_to_show = Utils.resize(image_to_show, best_candidate[2])
        cv2.rectangle(image_to_show, (start_x, start_y), (end_x, end_y), [0, 0, 255], 2)
        Utils.show(Utils.resize(image_to_show, 600))
        Utils.show(Utils.resize(Filtering.get_rgb_scale_gray(resulting_img), 600))

    return resulting_img, best_score


def preprocess(color_image, target_width):
    color_image = Utils.resize(color_image, target_width)
    image = Filtering.get_gray_scale_rgb(color_image)
    image = Filtering.sharpen(image, 5)
    thr, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image, color_image

if __name__ == "__main__":
    main()
