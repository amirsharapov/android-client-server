import asyncio
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import requests

from src.lib import adb
from src.lib.adb import screencap, execute_command, execute_command_async


def take_screenshot_with_adb(save_path: str = None) -> np.ndarray:
    path = screencap()

    screenshot = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    Path(path).unlink()

    if save_path:
        cv2.imwrite(
            save_path,
            screenshot
        )

    return screenshot


async def take_screenshot_with_api() -> np.ndarray:
    response = requests.get('http://127.0.0.1:8080')
    response.raise_for_status()

    screenshot = cv2.imdecode(
        np.frombuffer(
            response.content,
            np.uint8
        ),
        cv2.IMREAD_UNCHANGED
    )

    return screenshot


async def take_screenshots_with_api():
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get('http://127.0.0.1:8080') as response:
                response.raise_for_status()
                content = await response.read()

                yield cv2.imdecode(
                    np.frombuffer(content, np.uint8),
                    cv2.IMREAD_UNCHANGED
                )


async def tap(x: int, y: int):
    await adb.motionevent('down', x, y)
    await asyncio.sleep(.02)
    await adb.motionevent('up', x, y)
