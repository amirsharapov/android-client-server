import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import dotenv
import matplotlib.pyplot as plt
import numpy as np

from src.bots.hay_day.client import HayDayClient
from src.lib import adb, android

MANUAL_ASSOCIATIONS = {
    'harvest_crops': [0, 2],
    'plant_crops': [1, 3]
}

SEGMENT_ASSOCIATIONS_BY_INDEX = [
    'harvest_crops',
    'plant_crops',
    'harvest_crops',
    'plant_crops'
]


def get_events():
    file = Path('local/events/events_1712080057.txt')
    events = []

    for line in file.read_text().splitlines():
        event_batch = json.loads(line)
        events.extend(event_batch)

    return events


def segment_drag_events(events):
    drag_events = []
    current_drag = []
    dragging = False

    for event in events:
        if event['type'] == 'mousedown':
            dragging = True
            current_drag = [event]
        elif event['type'] == 'mouseup' and dragging:
            if len(current_drag) > 10:  # Ensure at least 10 mousemove events
                drag_events.append(current_drag)
            current_drag = []
            dragging = False
        elif event['type'] == 'mousemove' and dragging:
            current_drag.append(event)

    return drag_events


def segment_events(events):
    all_events = []
    current_segment = []

    for event in events:
        if event['type'] == 'mousedown':
            # Start of a new segment
            current_segment = [event]

        elif event['type'] == 'mousemove':
            # Continue the current segment
            current_segment.append(event)

        elif event['type'] == 'mouseup':
            # End of the current segment
            current_segment.append(event)

            # Apply the minimum move events filter for drags here if necessary
            if len([e for e in current_segment if e['type'] == 'mousemove']) > 10 or len(current_segment) <= 7:
                all_events.append(current_segment)

            # Reset for the next segment
            current_segment = []

    return all_events


def filter_drag_and_click(segmented_events):
    filtered_events = []
    previous_segment = None

    for segment in segmented_events:
        # Determine if current segment is a drag
        is_drag = len(segment) > 7

        if is_drag:
            # If the previous segment is a click (7 or fewer events), include it
            if previous_segment and len(previous_segment) <= 7:
                filtered_events.append(previous_segment)

            # Keep the first and last events, and only 1 in every 3 events in between
            # segment_local = [segment[0], segment[-1]]
            # segment_local.extend(segment[1:-1][::2])

            # Include the current drag segment
            filtered_events.append(segment)

        # Update previous_segment for the next iteration
        previous_segment = segment

    return filtered_events


def plot_drag_events(drag_events, filename='drag_events_plot.png', dpi=300):
    # Determine the number of drag event segments
    num_segments = len(drag_events)

    # Calculate the number of rows and columns for the subplot grid
    num_cols = int(num_segments ** 0.5) + 1
    num_rows = (num_segments + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for idx, segment in enumerate(drag_events):
        x = [event['x'] for event in segment]
        y = [event['y'] for event in segment]
        axs[idx].plot(x, y)
        axs[idx].set_title(f'Segment {idx}')
        axs[idx].set_xlabel('X')
        axs[idx].set_ylabel('Y')

        # Set the x and y limits to start at 0 and end at 1
        axs[idx].set_xlim([0, 1])
        axs[idx].set_ylim([0, 1])

        # Invert the y-axis to move the origin to the top-left
        axs[idx].invert_yaxis()

    # Hide any unused subplots
    for idx in range(len(drag_events), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()  # Adjust the layout to make room for titles, etc.
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)  # Close the figure to free memory


def plot_clicks_and_drags_together(filtered_events, filename='clicks_and_drags_plot.png', dpi=300):
    # Prepare to iterate through filtered_events while checking for clicks followed by drags
    segments_to_plot = []
    i = 0
    while i < len(filtered_events):
        # Check if the current segment is a click and followed by a drag
        if len(filtered_events[i]) <= 7 and i+1 < len(filtered_events) and len(filtered_events[i+1]) > 7:
            # Pair click with its following drag
            segments_to_plot.append((filtered_events[i], filtered_events[i+1]))
            i += 2  # Skip the next segment since it's already paired with the click
        else:
            # If not a click followed by a drag, treat it as a standalone segment
            segments_to_plot.append((filtered_events[i],))
            i += 1

    # Determine the number of plots
    num_plots = len(segments_to_plot)

    # Calculate the number of rows and columns for the subplot grid
    num_cols = int(num_plots ** 0.5) + 1
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the axs array for easy iteration

    for idx, segments in enumerate(segments_to_plot):
        for segment in segments:
            x = [event['x'] for event in segment]
            y = [event['y'] for event in segment]
            color = 'red' if len(segment) <= 7 else 'blue'
            if color == 'red':
                axs[idx].plot(x, y, color=color, marker='o', markersize=5)
            else:
                axs[idx].plot(x, y, color=color)
            axs[idx].set_xlim([0, 1])
            axs[idx].set_ylim([0, 1])
            axs[idx].invert_yaxis()  # Invert the y-axis

        axs[idx].set_title(f'Plot {idx}')
        axs[idx].set_xlabel('X')
        axs[idx].set_ylabel('Y')

    # Hide any unused subplots
    for idx in range(num_plots, len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()  # Adjust the layout
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)  # Close the figure to free memory


def label_event_segments(drag_events):
    labeled_data = defaultdict(list)

    for i, segment in enumerate(drag_events):
        label = SEGMENT_ASSOCIATIONS_BY_INDEX[i]

        labeled_data[label].append(segment)

    return labeled_data


def label_event_segments_v2(filtered_events):
    labeled_segments = []

    current_labeled_segment_index = 0
    current_labeled_segment = {
        'label': SEGMENT_ASSOCIATIONS_BY_INDEX[current_labeled_segment_index],
        'clicks': [],
        'drag_events': []
    }

    for segment in filtered_events:
        is_click = len(segment) <= 7

        if is_click:
            current_labeled_segment['clicks'].append(segment)
            continue

        current_labeled_segment['drag_events'].append(segment)

        labeled_segments.append(current_labeled_segment)

        current_labeled_segment_index += 1
        current_labeled_segment = {
            'label': (
                SEGMENT_ASSOCIATIONS_BY_INDEX[current_labeled_segment_index]
                if current_labeled_segment_index < len(SEGMENT_ASSOCIATIONS_BY_INDEX)
                else None
            ),
            'clicks': [],
            'drag_events': []
        }

    return labeled_segments


def create_clicks_and_drags_overlay(filtered_events, width=1380, height=800, filename='clicks_and_drags_overlay.png'):
    # Create a blank, fully transparent image
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    drags_printed = 0

    for segment in filtered_events:
        if drags_printed >= 2:
            break

        # Determine segment type (click or drag) based on its length
        if len(segment) <= 7:
            color = (255, 0, 0, 255)

        else:
            if drags_printed == 0:
                color = (0, 0, 255, 255)
            else:
                color = (0, 255, 0, 255)

        thickness = 1 if len(segment) <= 7 else 2  # Thinner for click, thicker for drag

        # Convert segment coordinates to pixel positions
        points = np.array([[int(event['x'] * width), int(event['y'] * height)] for event in segment], np.int32)

        # Draw the segment as a polyline or circles for clicks
        if len(segment) <= 7:  # Click
            for point in points:
                cv2.circle(overlay, tuple(point), 5, color, -1)  # Draw filled circles for clicks
        else:  # Drag
            cv2.polylines(overlay, [points], isClosed=False, color=color, thickness=thickness)
            drags_printed += 1

    # Save the image
    cv2.imwrite(filename, overlay)


def main():
    events = get_events()

    # Segment and filter the drag events
    filtered_drag_events = segment_drag_events(events)

    # Label the event segments
    labeled_data = label_event_segments(filtered_drag_events)

    # Save the labeled data to a file
    with open('local/labeled_data_v1.json', 'w') as f:
        json.dump(labeled_data, f, indent=4)

    # Plot the drag events
    plot_drag_events(filtered_drag_events, 'drag_events_plot_v1.png')


def main_v2():
    events = get_events()

    # Segment all events
    segmented_events = segment_events(events)

    # Filter drag and click events
    filtered_events = filter_drag_and_click(segmented_events)

    create_clicks_and_drags_overlay(filtered_events)

    # Label the event segments
    labeled_data = label_event_segments_v2(filtered_events)

    # Save the labeled data to a file
    with open('local/labeled_data_v2.json', 'w') as f:
        json.dump(labeled_data, f, indent=4)

    # Plot the events
    plot_clicks_and_drags_together(filtered_events, 'filtered_events_plot_v2.png')


async def run(label: str):
    with open('local/labeled_data_v2.json') as f:
        labeled_data = json.load(f)

    adb_motionevent_map = {
        'mousedown': 'down',
        'mouseup': 'up',
        'mousemove': 'move'
    }

    segment = None

    for segment in labeled_data:
        if segment['label'] == label:
            break

    check_error = False
    delay_between_events = [0.03, 0.05]

    for click in segment['clicks']:
        for event in click:
            adb_motionevent = adb_motionevent_map[event['type']]
            await adb.motionevent(
                adb_motionevent,
                int(event['x'] * 1376),
                int(event['y'] * 800),
                do_async=check_error
            )

            if not check_error:
                await asyncio.sleep(random.uniform(*delay_between_events), 3)

    await asyncio.sleep(random.uniform(0.8, 1.2), 3)

    for drag in segment['drag_events']:
        for event in drag:
            adb_motionevent = adb_motionevent_map[event['type']]
            await adb.motionevent(
                adb_motionevent,
                int(event['x'] * 1376),
                int(event['y'] * 800),
                do_async=check_error
            )

            if not check_error:
                await asyncio.sleep(random.uniform(*delay_between_events), 3)


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
            await asyncio.sleep(.15)
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


async def run_async():
    while True:
        await run('plant_crops')
        start = time.time()

        await asyncio.sleep(1)
        await sell_all_pending_wheat()
        elapsed = time.time() - start

        if elapsed <= 125:
            sleep_time = random.uniform(123 - elapsed, 125 - elapsed)
            await asyncio.sleep(sleep_time)

        await run('harvest_crops')
        await asyncio.sleep(random.uniform(3, 5))

        for _ in range(2):
            await run('plant_crops')
            await asyncio.sleep(random.uniform(123, 125))

            await run('harvest_crops')
            await asyncio.sleep(random.uniform(3, 5))


if __name__ == '__main__':
    dotenv.load_dotenv()
    main_v2()
    asyncio.run(run_async())
