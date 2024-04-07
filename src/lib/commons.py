from typing import TypeVar, Iterable

import numpy as np

_T = TypeVar('_T')

undefined = object()


def flatten(list_of_lists: list[Iterable[_T]]) -> list[_T]:
    return np.array([list(list_) for list_ in list_of_lists]).flatten().tolist()
