import time
from dataclasses import dataclass, field

import cv2
import requests
# import uvicorn
# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from starlette.middleware.cors import CORSMiddleware

from src.get_objects import get_objects
from src.lib import android_client
from src.lib.adb import tap
from src.lib.adb.commons import swipe
from src.lib.commons import take_screenshot


def main():
    get_objects()

    response = requests.get('http://localhost:8080')
    response.raise_for_status()

    with open('screenshot.png', 'wb') as file:
        file.write(response.content)


def main_v2():
    screenshot = android_client.take_screenshot()

    # use cv2 to draw lines in a grid 100px apart.

    for x in range(0, screenshot.shape[1], 100):
        cv2.line(screenshot, (x, 0), (x, screenshot.shape[0]), (0, 0, 255), 1)
        cv2.putText(screenshot, str(x), (x, 50), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 2)

    for y in range(0, screenshot.shape[0], 100):
        cv2.line(screenshot, (0, y), (screenshot.shape[1], y), (0, 0, 255), 1)
        cv2.putText(screenshot, str(y), (50, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 2)

    cv2.imwrite('screenshot.png', screenshot)

    # screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # tap(100, 100)
    # swipe(600, 600, 1000, 1000)


# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*'],
# )
#
#
# @app.get('/images/test')
# def get_images():
#     return FileResponse(
#         'screenshot.png',
#         media_type='image/png'
#     )
#
#
if __name__ == '__main__':
#     uvicorn.run(
#         'main:app',
#         host='localhost',
#         port=8080,
#     )
#     main()
    main_v2()
