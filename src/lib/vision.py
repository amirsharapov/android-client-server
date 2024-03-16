import cv2
import numpy as np


def crop(image, x, y, w, h):
    return image[y:y + h, x:x + w]


def get_alpha_mask(template: np.ndarray):
    assert template.shape[2] == 4, 'Template must have 4 channels.'

    mask = template[:, :, 3]
    mask = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY
    )[1]

    return mask
