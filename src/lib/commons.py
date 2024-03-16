from pathlib import Path

import cv2
import numpy as np

from src.lib import adb


def take_screenshot(save_path: str = None) -> np.ndarray:
    path = adb.screencap()

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
