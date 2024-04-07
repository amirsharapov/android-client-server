import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.websockets import WebSocket

from src.lib import adb
from src.lib.android import take_screenshot_with_api, take_screenshots_with_api

_mouse_events_handler = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    handler = get_mouse_event_handler()
    await handler.start_task()

    adb.forward_port(8080)
    adb.start_server()

    yield

    await handler.stop_task()


@dataclass
class MouseEventHandler:
    events: list[dict] = field(default_factory=list)
    events_to_write: list[dict] = field(default_factory=list)
    mouse_state: dict = field(default_factory=dict)
    task: asyncio.Task = None
    add_events_count: int = 0
    log_path: Path = None

    def __post_init__(self):
        if self.log_path is None:
            self.log_path = Path(f'local/events_{int(time.time())}.txt')

    async def start_task(self):
        self.task = asyncio.create_task(self.run())

    async def stop_task(self):
        self.task.cancel()
        await self.task

    async def run(self):
        while True:
            res = await self.handle_next_event()

            if not res:
                await asyncio.sleep(.1)

    def add_events(self, events: list[dict]):
        self.events.extend(events)
        self.events_to_write.extend(events)
        self.add_events_count += 1

        if self.add_events_count and self.add_events_count % 10 == 0:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with self.log_path.open('a') as f:
                f.write(json.dumps(self.events_to_write) + '\n')

            self.events_to_write.clear()

    async def handle_next_event(self):
        if not self.events:
            return False

        event = self.events.pop(0)

        if event['type'] == 'mousedown':
            self.mouse_state['mousedown'] = True
            await adb.motionevent(
                'down',
                int(event['x'] * 1380),
                int(event['y'] * 800),
                do_async=False
            )

        if event['type'] == 'mouseup':
            self.mouse_state['mousedown'] = False
            await adb.motionevent(
                'up',
                int(event['x'] * 1376),
                int(event['y'] * 800),
                do_async=False
            )

        if event['type'] == 'mousemove' and self.mouse_state.get('mousedown'):
            await adb.motionevent(
                'move',
                int(event['x'] * 1376),
                int(event['y'] * 800),
                do_async=False
            )

        return True


def get_mouse_event_handler():
    global _mouse_events_handler

    if not _mouse_events_handler:
        _mouse_events_handler = MouseEventHandler()

    return _mouse_events_handler


app = FastAPI(
    lifespan=lifespan
)
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
    handler = get_mouse_event_handler()
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        handler.add_events(data)


@app.get('/api/v1/images/clicks-and-drags-overlay')
def get_clicks_and_drags_overlay():
    path = Path('clicks_and_drags_overlay.png')
    return FileResponse(
        path=path,
        media_type='image/png'
    )


@app.get('/')
def index():
    return FileResponse(
        path='public/index.html'
    )


if __name__ == '__main__':
    dotenv.load_dotenv()
    uvicorn.run(
        'main:app',
        host='localhost',
        port=8000,
    )
