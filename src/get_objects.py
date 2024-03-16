import time

import cv2
import numpy as np
import requests
import torch
import supervision as sv

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from src import paths
from src.json import JSONFile

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = 'vit_h'
CHECKPOINT_PATH = paths.get_sam_checkpoints() / 'sam_vit_h_4b8939.pth'


def get_model():
    model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    model = model.to(DEVICE)

    return model


def download_checkpoints():
    response = requests.get('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', stream=True)

    if CHECKPOINT_PATH.exists():
        return

    with open(CHECKPOINT_PATH, 'wb') as file:
        with response:
            response.raise_for_status()
            total = int(response.headers.get('content-length'))

            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
                print(f'Downloading: {file.tell() / total:.2%}', end='\r')


def get_objects():
    download_checkpoints()

    model = get_model()
    image_bgr = cv2.imread('screenshot.png')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    generator = SamAutomaticMaskGenerator(model)

    start = time.time()
    result = generator.generate(image_rgb)
    elapsed = time.time() - start

    print(f'Elapsed: {elapsed:.2f}s')

    annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(result)

    annotated_image = annotator.annotate(
        scene=image_rgb.copy(),
        detections=detections
    )

    sv.plot_images_grid(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    print('done')

    # Convert to polygons:

    polygons = []
