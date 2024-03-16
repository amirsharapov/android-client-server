import time

import cv2
import numpy as np
import requests


def take_screenshot():
    start = time.time()
    response = requests.get('http://127.0.0.1:8080/screenshot')

    print(f'Elapsed time: {time.time() - start} seconds')
    response.raise_for_status()

    assert response.headers['Content-Type'] == 'image/jpeg'

    image = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image
