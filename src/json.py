import json
from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, Path):
            return str(o.absolute())
        if isinstance(o, set):
            return list(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, bytes):
            return o.decode('utf-8')
        if isinstance(o, Enum):
            return o.name
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


@dataclass
class JSONFile:
    path: Path

    @classmethod
    def create(cls, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        return cls(path)

    def __post_init__(self):
        self.touch()

    def touch(self, exist_ok=True):
        if self.path.exists():
            return

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self.path.touch(
            exist_ok=exist_ok
        )
        self.path.write_text(
            data='{}',
            encoding='utf-8'
        )

    def has_key(self, key: str):
        data = self.read()
        return key in data

    def read(self):
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))

        except json.JSONDecodeError:
            return {}

    def read_key(self, key, default=None):
        data = self.read()
        return data.get(key, default)

    def write(self, data):
        self.path.write_text(
            json.dumps(
                data,
                indent=4,
                cls=JSONEncoder
            )
        )

    def write_key(self, key, value):
        data = self.read()
        data[key] = value
        self.write(data)
