from enum import Enum


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
