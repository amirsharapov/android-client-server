from pathlib import Path

def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_root():
    return mkdir(Path('local'))


def get_sam_checkpoints():
    return mkdir(_get_root() / 'checkpoints')
