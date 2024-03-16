import os
import subprocess

ADB_DIR = os.getenv(
    'ADB_EXE',
    'C:/Users/amir/Code/android-client/bin/platform-tools'
)


def send_command(
        command: str,
        wait: bool = True
) -> bytes:
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=ADB_DIR,
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
