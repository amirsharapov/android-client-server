from typing import Iterable

import cv2
import numpy as np

from src.lib import vision
from src.lib.vision import Rectangle, Match


def merge_matches(matches: Iterable[Match]):
    rects = [tuple(match.rectangle) for match in matches]
    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)

    for rect in rects:
        yield Rectangle(*rect)


def merge_rectangles(rectangles: list[Rectangle]):
    rects = [tuple(rectangle) for rectangle in rectangles]
    rects, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)

    for rect in rects:
        yield Rectangle(*rect)


def match_x_button(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/x_button.png',
            use_mask=True,
        )
    )


def match_roadside_shop(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/farm/roadside_shop.png',
            threshold=0.65,
            use_mask=True,
        )
    )


def match_open_roadside_shop_slot(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/create_new_sale.png',
            use_mask=False,
        )
    )


def match_sold_roadside_shop_slot(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sold.png',
            use_mask=True,
        )
    )


def match_purchase_new_roadside_shop_slot(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/purchase_new_slot.png',
            use_mask=True,
        )
    )


def match_roadside_shop_layout(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/layout.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_wheat_icon(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/wheat_icon.png',
            use_mask=True,
        )
    )


def match_roadside_shop_silo_storage_text(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/silo_storage.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_silo_icon(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/silo_icon.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_plus_icon(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/plus.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_plus_disabled_icon(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/plus_disabled.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_plus_max_button(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/plus_max.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_sell_icon(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/sell_icon.png',
            use_mask=True,
        )
    )


def match_roadside_shop_sale_preview_put_on_sale_button(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/sale_preview/put_on_sale.png',
            use_mask=True,
        )
    )


def match_roadside_shop_advertise_now_text(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/advertise_now.png',
            use_mask=True,
        )
    )


def match_roadside_shop_occupied_by_wheat(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/occupied_by_wheat.png',
            use_mask=True,
        )
    )


def create_advertisement_button(image: np.ndarray):
    yield from merge_matches(
        vision.match_template_from_path(
            image=image,
            path='public/images/hay_day/templates/roadside_shop/create_advertisement.png',
            use_mask=True,
        )
    )
