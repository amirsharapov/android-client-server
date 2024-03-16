import cv2
import numpy as np

from src.lib.templates import base


def match_easy_marker(image: np.ndarray):
    yield from base.match_template_from_path(
        image,
        'assets/components/easy_marker.png',
        0.5,
        cv2.TM_CCOEFF_NORMED,
        True
    )


def match_hatchet(image: np.ndarray):
    yield from base.match_template_from_path(
        image,
        'assets/components/hatchet.png',
        0.9,
    )


def match_close_with_x(image: np.ndarray):
    yield from base.match_template_from_path(
        image,
        'assets/components/close_with_x.png',
        0.9,
    )


def match_mini_map_frame(image: np.ndarray):
    yield from base.match_template_from_path(
        image,
        'assets/components/mini_map_frame.png',
        0.7,
    )
