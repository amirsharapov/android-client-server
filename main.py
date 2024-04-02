from dataclasses import dataclass

import cv2
import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.websockets import WebSocket

from src.lib.android import take_screenshot_with_api, take_screenshots_with_api

_mouse_state = None


def get_mouse_state():
    global _mouse_state

    if _mouse_state is None:
        _mouse_state = MouseState(
            x=0,
            y=0,
            is_mouse_down=False
        )

    return _mouse_state


@dataclass
class MouseState:
    x: float
    y: float
    is_mouse_down: bool = False

    def handle_state_update(self, x: float, y: float, is_mouse_down: bool):
        self.x = x
        self.y = y
        self.is_mouse_down = is_mouse_down


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/v1/images/latest')
def get_image():
    screenshot = take_screenshot_with_api()
    _, buffer = cv2.imencode('.jpeg', screenshot)
    return Response(
        content=buffer.tobytes(),
        media_type='image/jpeg'
    )


@app.websocket('/api/v1/ws/image-stream')
async def ws_image_stream(websocket: WebSocket):
    await websocket.accept()

    async for screenshot in take_screenshots_with_api():
        _, buffer = cv2.imencode('.jpeg', screenshot)
        await websocket.send_bytes(buffer.tobytes())


@app.websocket('/api/v1/ws/mouse-state')
async def ws_mouse_state(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        print(f"Message text was: {data}")


@app.get('/')
def index():
    return FileResponse(
        path='public/index.html'
    )


if __name__ == '__main__':
    dotenv.load_dotenv()
    # turn_on('0309')
    uvicorn.run(
        'main:app',
        host='localhost',
        port=8000,
    )
    # main()
    # main_v2()
