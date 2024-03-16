import time
from pathlib import Path

from src.lib.adb.commands import send_command
from src.lib.adb.enums import WakefulnessStates


def screencap(path: str = None):
    path = path or './temp/screenshot.png'
    path = Path(path).resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    send_command(f'adb exec-out screencap -p > {path}')
    return path


def swipe(x1: int, y1: int, x2: int, y2: int, duration: int = 100):
    send_command(f'adb shell input swipe {x1} {y1} {x2} {y2} {duration}')


def tap(x: int, y: int):
    send_command(f'adb shell input tap {x} {y}')


def press_power_button():
    send_command('adb shell input keyevent 26')


def get_wakefulness_state():
    state = send_command('adb shell dumpsys power | find "mWakefulness="')
    state = state.decode().split('=')[1].strip()

    return WakefulnessStates.from_string(state)


def turn_on(passcode: str = None):
    state = get_wakefulness_state()

    if state == WakefulnessStates.ASLEEP:
        press_power_button()

    elif state == WakefulnessStates.DREAMING:
        tap(100, 100)

    elif state == WakefulnessStates.DOZING:
        tap(100, 100)

    elif state == WakefulnessStates.AWAKE:
        pass

    else:
        raise Exception(f'Invalid wakefulness state: {state}')

    time.sleep(2)

    swipe(100, 100, 500, 500)

    if passcode:
        for digit in passcode:
            send_command(f'adb shell input text {digit}')

    send_command('adb shell input keyevent 66')

    return get_wakefulness_state()
