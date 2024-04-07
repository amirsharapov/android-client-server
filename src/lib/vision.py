from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np


@dataclass
class Rectangle:
    x: int
    y: int
    w: int
    h: int

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __getitem__(self, item: Literal['x', 'y', 'w', 'h'] | int):
        if isinstance(item, int):
            return (self.x, self.y, self.w, self.h)[item]

        if item == 'x':
            return self.x

        if item == 'y':
            return self.y

        if item == 'w':
            return self.w

        if item == 'h':
            return self.h

        raise KeyError(f'Invalid key: {item}')

    @property
    def top_left(self):
        return Point(self.x, self.y)

    @property
    def bottom_right(self):
        return Point(self.x + self.w, self.y + self.h)

    @property
    def center(self):
        return Point(
            (self.x + self.x + self.w) // 2,
            (self.y + self.y + self.h) // 2
        )

    def is_above(self, other):
        return self.bottom_right.y < other.top_left.y

    def is_below(self, other):
        return self.top_left.y > other.bottom_right.y

    def is_left_of(self, other):
        return self.bottom_right.x < other.top_left.x

    def is_right_of(self, other):
        return self.top_left.x > other.bottom_right.x


@dataclass
class Point:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))

    def __getitem__(self, item: Literal['x', 'y'] | int):
        if isinstance(item, int):
            return (self.x, self.y)[item]

        if item == 'x':
            return self.x

        if item == 'y':
            return self.y

        raise KeyError(f'Invalid key: {item}')


@dataclass
class Match:
    top_left: Point
    bottom_right: Point
    confidence: float | None

    @classmethod
    def from_xywh(cls, x, y, w, h, confidence):
        return cls(
            Point(x, y),
            Point(x + w, y + h),
            confidence
        )

    def __iter__(self):
        return iter((self.top_left, self.bottom_right))

    @property
    def rectangle(self):
        return Rectangle(
            self.top_left.x,
            self.top_left.y,
            self.bottom_right.x - self.top_left.x,
            self.bottom_right.y - self.top_left.y
        )


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


def match_template(
        image,
        template,
        threshold: float = 0.9,
        method: int = cv2.TM_CCOEFF_NORMED,
        mask: np.ndarray = None,
        filter_inf: bool = True
) -> Iterator[Match]:
    """
    Match a template in an image and return an iterator of top left and bottom right points of the matches.
    """
    w, h = template.shape[::-1]

    result = cv2.matchTemplate(
        image,
        template,
        method,
        None,
        mask
    )

    if filter_inf:
        result = np.where(np.isfinite(result), result, 0)

    locations = np.where(result >= threshold)

    for point in zip(*locations[::-1]):
        confidence = result[point[1], point[0]]

        yield Match(
            Point(*point),
            Point(point[0] + w, point[1] + h),
            float(confidence)
        )


def match_template_from_path(
        image: np.ndarray,
        path: str | Path,
        threshold: float = 0.8,
        method: int = cv2.TM_CCOEFF_NORMED,
        use_mask: bool = False,
        filter_inf: bool = True
):
    path = Path(path).as_posix()

    template = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )

    assert template is not None, f'Could not read template from path: {path}'

    template_mask = get_alpha_mask(template) if use_mask else None

    template = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY
    )

    yield from match_template(
        image,
        template,
        threshold,
        method,
        template_mask,
        filter_inf
    )


def match_templates_from_paths(
        image: np.ndarray,
        paths: list[str | Path],
        threshold: float = 0.8,
        method: int = cv2.TM_CCOEFF_NORMED,
        use_mask: bool = False,
        filter_inf: bool = True
):
    for path in paths:
        yield from match_template_from_path(
            image,
            path,
            threshold,
            method,
            use_mask,
            filter_inf
        )
