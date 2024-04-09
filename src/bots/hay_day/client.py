import asyncio
from dataclasses import dataclass, field
from typing import Literal

import cv2

from src.bots.hay_day import templates
from src.bots.hay_day.templates import merge_rectangles
from src.lib import android
from src.lib.commons import flatten
from src.lib.vision import Rectangle


SlotType = Literal[
    'sold',
    'open',
    'occupied_by_wheat'
]


@dataclass
class HayDayClient:
    current_view: str

    farm: 'Farm' = field(init=False)
    roadside_shop: 'RoadsideShop' = field(init=False)

    def __post_init__(self):
        self.roadside_shop = RoadsideShop(self)
        self.farm = Farm(self)


@dataclass
class Farm:
    client: 'HayDayClient'

    @staticmethod
    async def click_roadside_shop():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop(image)
        rects = list(rects)

        assert len(rects) == 1

        match = rects[0]

        await android.tap(
            match.center.x,
            match.center.y,
        )


@dataclass
class RoadsideShopSlot:
    type: SlotType
    rectangle: Rectangle
    item: str = None


@dataclass
class RoadsideShop:
    client: 'HayDayClient'

    sale_preview: 'SalePreview' = field(init=False)

    def __post_init__(self):
        self.sale_preview = SalePreview(self.client)

    @staticmethod
    async def click_x_button():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_x_button(image)
        rects = list(rects)

        assert len(rects) == 1

        match = rects[0]

        await android.tap(
            match.center.x,
            match.center.y,
        )

    @staticmethod
    async def click_advertise_now_button():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop_advertise_now_text(image)
        rects = list(rects)

        assert len(rects) == 1

        rect = rects[0]

        x = rect.bottom_right.x + 40
        y = rect.center.y

        await android.tap(x, y)

    @staticmethod
    async def click_create_advertisement_button():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.create_advertisement_button(image)
        rects = list(rects)

        assert len(rects) == 1

        rect = rects[0]

        await android.tap(
            rect.center.x,
            rect.center.y
        )

    @staticmethod
    async def scroll_through_shop(reverse: bool = False):
        direction = 'backward' if reverse else 'forward'

        while True:
            image_bgr = await android.take_screenshot_with_api()
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            layout = templates.match_roadside_shop_layout(image)
            layout = next(layout, None)

            assert layout, 'Could not find layout while attempting to scroll through shop.'

            yield image_bgr

            if direction == 'forward':
                purchase_new_slot = templates.match_purchase_new_roadside_shop_slot(image)
                purchase_new_slot = next(purchase_new_slot, None)

                if purchase_new_slot:
                    break

            padding = 180

            x1, y1 = layout.top_left
            x2, y2 = layout.bottom_right

            x1 += padding
            y1 += padding
            x2 -= padding
            y2 -= padding

            start_x = x2
            start_y = (y1 + y2) // 2

            end_x = x1
            end_y = (y1 + y2) // 2

            n = 25

            x_spacing = (start_x - end_x) // n
            y_spacing = (start_y - end_y) // n

            points = [
                (start_x - i * x_spacing, start_y - i * y_spacing)
                for i in range(n)
            ]

            if direction == 'forward':
                points = sorted(points, key=lambda p: -p[0])

            else:
                points = sorted(points, key=lambda p: p[0])

            await android.press(points[0][0], points[0][1])

            for point in points[1:-1]:
                await android.move(point[0], point[1])

            await android.release(points[-1][0], points[-1][1])
            await asyncio.sleep(.33)

    async def iterate_slots(self, slot_types: list[SlotType] = None, reverse: bool = False):
        match_fns = {
            'sold': templates.match_sold_roadside_shop_slot,
            'open': templates.match_open_roadside_shop_slot,
            'occupied_by_wheat': templates.match_roadside_shop_occupied_by_wheat,
        }

        async for _ in self.scroll_through_shop(reverse=reverse):
            for slot_type in slot_types:
                image = await android.take_screenshot_with_api()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                match_fn = match_fns[slot_type]
                rects = match_fn(image)

                for rect in rects:
                    yield RoadsideShopSlot(
                        slot_type,
                        rect
                    )


@dataclass
class SalePreview:
    client: 'HayDayClient'

    @staticmethod
    async def ensure_silo_inventory_is_toggled():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop_silo_storage_text(image)
        rects = list(rects)

        if len(rects) == 1:
            return

        rects = templates.match_roadside_shop_sale_preview_silo_icon(image)
        rects = list(rects)

        assert len(rects) == 1

        match = rects[0]

        await android.tap(
            match.center.x,
            match.center.y,
        )

    @staticmethod
    async def click_wheat_icon():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop_sale_preview_wheat_icon(image)
        rects = list(rects)

        assert len(rects) == 1

        match = rects[0]

        await android.tap(
            match.center.x,
            match.center.y,
        )

    @staticmethod
    async def click_price_plus_max_button():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop_sale_preview_plus_max_button(image)
        rects = list(rects)

        assert len(rects) == 1

        match = rects[0]

        await android.tap(
            match.center.x,
            match.center.y,
        )

    @staticmethod
    async def click_put_on_sale_button():
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop_sale_preview_put_on_sale_button(image)
        rects = list(rects)

        assert len(rects) == 1

        match = rects[0]

        await android.tap(
            match.center.x,
            match.center.y,
        )

    @staticmethod
    async def click_quantity_plus_button(times: int):
        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = list(
            merge_rectangles(
                flatten([
                    templates.match_roadside_shop_sale_preview_plus_icon(image),
                    templates.match_roadside_shop_sale_preview_plus_disabled_icon(image)
                ])
            )
        )

        rects = sorted(rects, key=lambda r: r.center.y)

        assert len(rects) == 2

        quantity_plus = rects[0]

        for _ in range(times):
            await android.tap(
                quantity_plus.center.x,
                quantity_plus.center.y,
            )

            await asyncio.sleep(.1)

    async def is_wheat_icon_visible(self):
        await self.ensure_silo_inventory_is_toggled()
        await asyncio.sleep(1)

        image = await android.take_screenshot_with_api()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = templates.match_roadside_shop_sale_preview_wheat_icon(image)
        rects = list(rects)

        return len(rects) == 1
