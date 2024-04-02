import subprocess
import time
from pathlib import Path

from src.lib.env import get_adb_alias, get_adb_exe_dir

from enum import Enum

ADB_ALIAS = get_adb_alias()


class KeyCodes(Enum):
    POWER = 'KEYCODE_POWER'


class KeyEvents(Enum):
    LOCK_SCREEN_OK = 66


class WakefulnessStates(Enum):
    ASLEEP = 'Asleep'
    AWAKE = 'Awake'
    DREAMING = 'Dreaming'
    DOZING = 'Dozing'
    UNKNOWN = ''

    @classmethod
    def from_string(cls, string):
        for state in cls:
            if state.value == string:
                return state
        raise ValueError(f'Invalid wakefulness state: {string}')


def screencap(path: str = None):
    path = path or './temp/screenshot.png'
    path = Path(path).resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    execute_command(f'{ADB_ALIAS} exec-out screencap -p > {path}')
    return path


def swipe(x1: int, y1: int, x2: int, y2: int, duration: int = 100):
    execute_command(f'{ADB_ALIAS} shell input swipe {x1} {y1} {x2} {y2} {duration}')


def tap(x: int, y: int):
    execute_command(f'{ADB_ALIAS} shell input tap {x} {y}')


def press_power_button():
    execute_command(f'{ADB_ALIAS} shell input keyevent 26')


def get_wakefulness_state():
    state = execute_command(f'{ADB_ALIAS} shell dumpsys power | grep "mWakefulness="')
    state = state.decode().split('=')[1].strip()

    return WakefulnessStates.from_string(state)


def turn_on(passcode: str = None):
    state = get_wakefulness_state()

    if state == WakefulnessStates.ASLEEP:
        press_power_button()

    elif state == WakefulnessStates.DREAMING or state == WakefulnessStates.DOZING:
        tap(100, 100)

    elif state == WakefulnessStates.AWAKE:
        pass

    else:
        raise Exception(f'Invalid wakefulness state: {state}')

    time.sleep(2)

    swipe(100, 100, 500, 500)

    if passcode:
        for digit in passcode:
            execute_command(f'{ADB_ALIAS} shell input text {digit}')

    execute_command(f'{ADB_ALIAS} shell input keyevent 66')

    return get_wakefulness_state()


def execute_command(
        command: str,
        wait: bool = True
) -> bytes:
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=get_adb_exe_dir(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if wait:
        process.wait()

    stdout, stderr = process.communicate()

    if stderr or process.returncode != 0:
        msg = stderr or f'Error executing command. Args: {process.args}. Return code: {process.returncode}.'
        raise Exception(msg)

    return stdout
