import asyncio
from dataclasses import dataclass
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import cv2
import dotenv
import numpy as np

from src.bots.hay_day import templates
from src.bots.hay_day.client import HayDayClient
from src.lib import android, vision
from src.lib.vision import Rectangle


@dataclass
class Image:
    name: str
    path: str | Path

    valid_detect_color_params: list['DetectColorParams'] = None


@dataclass
class DetectColorParams:
    seed: int
    n: int


@dataclass
class DetectColorsResponse:
    colors: list['DetectedColor']
    seed: int


@dataclass
class DetectedColor:
    lower: list[int]
    upper: list[int]


images = {
    'plus': Image(
        name='plus',
        path='public/images/hay_day/templates/roadside_shop/sale_preview/plus.png',
        valid_detect_color_params=[
            DetectColorParams(seed=507, n=3),
            DetectColorParams(seed=507, n=5),
            DetectColorParams(seed=239, n=3)
        ]
    )
}


def detect_colors(image: np.ndarray, n: int, seed: int = None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape((-1, 3))

    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(hsv)

    colors = []

    for center in kmeans.cluster_centers_:
        lower = [max(0, center[0] - 10), 100, 100]
        upper = [min(179, center[0] + 10), 255, 255]

        colors.append(
            DetectedColor(
                lower=lower,
                upper=upper
            )
        )

    return DetectColorsResponse(
        colors=colors,
        seed=seed
    )


async def main_3():
    image = cv2.imread('public/images/hay_day/templates/roadside_shop/sale_preview/plus.png')
    response = detect_colors(image, 3, seed=507)
    visualize_color_ranges(image, response.colors, 'viz_1.png')


def calculate_area_of_color(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    color_area = np.sum(mask > 0)
    total_area = image.shape[0] * image.shape[1]

    percentage = (color_area / total_area) * 100

    return percentage


def visualize_color_ranges(image, colors: list[DetectedColor], output_file):
    areas = [
        calculate_area_of_color(
            image,
            color.lower,
            color.upper
        ) for color in colors
    ]

    colors = [cv2.cvtColor(np.uint8([[color.upper]]), cv2.COLOR_HSV2RGB)[0][0] / 255 for color in colors]
    labels = [f"Color range {i + 1}" for i in range(len(colors))]

    plt.pie(areas, colors=colors, labels=labels)

    plt.savefig(output_file)


async def main_v1():
    image = await android.take_screenshot_with_api()
    cv2.imwrite('test.png', image)

    image = cv2.imread('test.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canvas = cv2.imread('test.png')

    template = Path('public/images/hay_day/templates/roadside_shop/sale_preview/layout.png')

    matches = vision.match_template_from_path(
        image,
        template,
        use_mask=True,
        threshold=0.9
    )

    rectangles = [(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]) for tl, br in matches]
    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)

    for (x, y, w, h) in rectangles:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        midpoint = (x + w // 2, y + h // 2)
        cv2.circle(canvas, midpoint, 5, (0, 255, 0), -1)

    cv2.imwrite('test_post.png', canvas)

    # await android.tap(*midpoint)


async def get_current_state():
    image = await android.take_screenshot_with_api()
    cv2.imwrite('test.png', image)

    image = cv2.imread('test.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canvas = cv2.imread('test.png')

    state = {
        'is_quantity_maxed': False,
        'is_price_maxed': False
    }

    roadside_shop_templates = Path('public/images/hay_day/templates/roadside_shop')
    roadside_shop_layout = roadside_shop_templates / 'layout.png'

    matches = list(
        vision.match_template_from_path(
            image,
            roadside_shop_layout,
            use_mask=True,
            threshold=0.9
        )
    )

    assert matches, 'Roadside Shop layout not found'

    sale_preview_templates = roadside_shop_templates / 'sale_preview'
    no_item_selected = sale_preview_templates / 'no_item_selected.png'

    matches = list(
        vision.match_template_from_path(
            image,
            no_item_selected,
            use_mask=True,
            threshold=0.9
        )
    )

    assert not matches, 'No item selected'

    matches = np.array(list(
        vision.match_templates_from_paths(
            image=image,
            paths=[
                sale_preview_templates / 'plus.png',
                sale_preview_templates / 'plus_disabled.png'
            ],
            use_mask=True,
            threshold=0.9
        )
    )).flatten()

    rectangles = [tuple(match.rectangle) for match in matches]
    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)

    assert len(rectangles) == 2, 'Expected 2 plus signs'

    rectangles = [Rectangle(*rectangle) for rectangle in rectangles]
    rectangles.sort(key=lambda rect: rect.y)

    plus_colors = detect_colors(
        cv2.imread(images['plus'].path),
        images['plus'].valid_detect_color_params[0].n,
        images['plus'].valid_detect_color_params[0].seed
    ).colors

    quantity_plus = rectangles[0]
    quantity_plus_area_of_color = 0

    for color in plus_colors:
        quantity_plus_area_of_color += calculate_area_of_color(
            vision.crop(canvas, *quantity_plus),
            color.lower,
            color.upper
        )

    price_plus = rectangles[1]
    price_plus_area_of_color = 0

    for color in plus_colors:
        price_plus_area_of_color += calculate_area_of_color(
            vision.crop(canvas, *price_plus),
            color.lower,
            color.upper
        )

    state['is_quantity_maxed'] = quantity_plus_area_of_color < 90
    state['is_price_maxed'] = price_plus_area_of_color < 90

    print(state)


async def main_v2():
    image = await android.take_screenshot_with_api()
    cv2.imwrite('test.png', image)

    image = cv2.imread('test.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match = next(
        vision.match_template_from_path(
            image,
            Path('public/images/hay_day/templates/roadside_shop/create_new_sale.png'),
            use_mask=True,
            threshold=0.9
        ),
        None
    )

    if not match:
        return

    await android.tap(*match.rectangle.center)
    await asyncio.sleep(.4)

    image = await android.take_screenshot_with_api()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match = next(
        vision.match_template_from_path(
            image,
            Path('public/images/hay_day/templates/roadside_shop/sale_preview/wheat_icon.png'),
            use_mask=True,
            threshold=0.7
        ),
        None
    )

    if not match:
        print('No match')
        return

    await android.tap(*match.rectangle.center)
    await asyncio.sleep(.4)

    image = await android.take_screenshot_with_api()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match = next(
        vision.match_template_from_path(
            image,
            Path('public/images/hay_day/templates/roadside_shop/sale_preview/plus_max.png'),
            use_mask=True,
            threshold=0.9
        ),
        None
    )

    if not match:
        print('No match')
        return

    await android.tap(*match.rectangle.center)
    await asyncio.sleep(.4)


async def sell_all_pending_wheat():
    """
    - Check color of plus signs
    - Use memory to store the state of the device without needing to constantly check screenshots.
    """
    client = HayDayClient('farm')

    await client.farm.click_roadside_shop()
    await asyncio.sleep(.2)

    has_wheat_left = True

    async for slot in client.roadside_shop.iterate_slots(slot_types=['sold', 'open']):
        if slot.type == 'sold':
            await android.tap(*slot.rectangle.center)
            await asyncio.sleep(.5)
            continue

        if slot.type == 'open':
            if not has_wheat_left:
                continue

            await android.tap(*slot.rectangle.center)
            await asyncio.sleep(.5)

            if not await client.roadside_shop.sale_preview.is_wheat_icon_visible():
                has_wheat_left = False
                await client.roadside_shop.click_x_button()
                await asyncio.sleep(.5)
                continue

            await client.roadside_shop.sale_preview.click_wheat_icon()
            await client.roadside_shop.sale_preview.click_quantity_plus_button(5)
            await client.roadside_shop.sale_preview.click_price_plus_max_button()
            await client.roadside_shop.sale_preview.click_put_on_sale_button()
            await asyncio.sleep(.5)

    async for slot in client.roadside_shop.iterate_slots(slot_types=['occupied_by_wheat'], reverse=True):
        if slot.type == 'occupied_by_wheat':
            await android.tap(*slot.rectangle.center)
            await asyncio.sleep(.5)

            await client.roadside_shop.click_advertise_now_button()
            await asyncio.sleep(.5)

            await client.roadside_shop.click_create_advertisement_button()
            await asyncio.sleep(.5)

            break

    await client.roadside_shop.click_x_button()


async def main():
    image = await android.take_screenshot_with_api()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = templates.match_roadside_shop(image)
    rects = list(rects)

    print(len(rects))

    for rect in rects:
        cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)

    cv2.imwrite('test_post_1.png', image)


if __name__ == '__main__':
    dotenv.load_dotenv()
    asyncio.run(main())
