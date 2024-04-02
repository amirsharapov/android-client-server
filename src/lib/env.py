import os

from src.lib.commons import undefined


def get_env(key: str, default: str = undefined):
    if key in os.environ:
        return os.getenv(key)

    if default is not undefined:
        return default

    raise KeyError(f'Environment variable {key} not found.')


def get_adb_exe_dir():
    return get_env('ADB_EXE_DIR')


def get_adb_alias():
    return get_env(
        'ADB_ALIAS',
        './adb'
    )
