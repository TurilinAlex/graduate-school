from functools import wraps
from typing import Protocol

import numpy as np

__all__ = [
    "min_extremum",
    "max_extremum",
    "merge_extremum",
    "TypeCoincident",
    "Coincident",
    "CoincidentNew",
]


def coincident(max_coincident=1):
    def coincident_decorator(extremum):
        @wraps(extremum)
        def wrapper(index, eps):
            coincident_num = 1
            extreme = extremum(index=index, eps=eps)
            eps += 1
            while coincident_num < max_coincident:
                recalculated_extreme = extremum(index=index, eps=eps)
                if len(extreme) == len(recalculated_extreme):
                    coincident_num += 1
                else:
                    extreme = recalculated_extreme
                    coincident_num = 1
                    eps += 1
            return np.array(extreme), eps

        return wrapper

    return coincident_decorator


def coincident_new(max_coincident=1):
    def coincident_decorator(extremum):
        @wraps(extremum)
        def wrapper(index, eps):
            coincident_num = 1
            extreme = extremum(index=index, eps=eps)
            eps += 1
            while coincident_num < max_coincident:
                recalculated_extreme = extremum(index=extreme, eps=eps)
                if len(extreme) == len(recalculated_extreme):
                    coincident_num += 1
                else:
                    coincident_num = 1
                    eps += 1
                extreme = recalculated_extreme
                print(eps, len(extreme))
            return np.array(extreme), eps

        return wrapper

    return coincident_decorator


def min_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
    n, extreme_min = len(index), []
    for i in range(n):
        for j in range(1, i + 1):
            if abs(index[i] - index[i - j]) <= eps:
                break
        else:
            extreme_min.append(index[i])
    return np.array(extreme_min)


def max_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
    n, extreme_max = len(index), []
    for i in range(n):
        for j in range(1, (n - i)):
            if abs(index[i] - index[i + j]) <= eps:
                break
        else:
            extreme_max.append(index[i])
    return np.array(extreme_max)


def merge_extremum(
        extr_min_index: np.ndarray[np.uint32],
        extr_max_index: np.ndarray[np.uint32],
        values: np.ndarray[np.float32]
):  # sourcery skip: low-code-quality
    extr, extr_min_new, extr_max_new = [], [], []
    i = j = 0
    status = 0
    i_min = j_max = None
    min_over, max_over = max(values) + 1, min(values) - 1
    value_min, value_max = min_over, max_over

    while i + j < len(extr_min_index) + len(extr_max_index):
        if i < len(extr_min_index) and j < len(extr_max_index):
            if extr_max_index[j] < extr_min_index[i]:
                status = -1
            if extr_max_index[j] > extr_min_index[i]:
                status = 1
            if extr_max_index[j] == extr_min_index[i]:
                status = 0

        if i >= len(extr_min_index):
            status = -1
        if j >= len(extr_max_index):
            status = 1

        if status >= 0:
            if values[extr_min_index[i]] < value_min:
                value_min = values[extr_min_index[i]]
                i_min = extr_min_index[i]
            if j_max is not None:
                extr_max_new.append(j_max)
                extr.append(j_max)
                j_max = None
            value_max = max_over
            i += 1
        else:
            if values[extr_max_index[j]] >= value_max:
                value_max = values[extr_max_index[j]]
                j_max = extr_max_index[j]
            if i_min is not None:
                extr_min_new.append(i_min)
                extr.append(i_min)
                i_min = None
            value_min = min_over
            j += 1

    if status < 0:
        extr.append(j_max)
        extr_max_new.append(j_max)
    else:
        extr.append(i_min)
        extr_min_new.append(i_min)

    return np.array(extr), np.array(extr_min_new), np.array(extr_max_new)


class TypeCoincident(Protocol):

    @staticmethod
    def __call__(
            extremum,
            index: np.ndarray[np.uint32],
            max_coincident: int = 1,
            eps: int = 1,
    ):
        pass


class Coincident:

    @staticmethod
    def __call__(
            extremum,
            index: np.ndarray[np.uint32],
            max_coincident=1,
            eps: int = 1,
    ):
        coincident_num = 1
        extreme = extremum(index, eps)
        while coincident_num < max_coincident:
            eps += 1
            recalculated_extreme = extremum(index, eps)
            if len(extreme) == len(recalculated_extreme):
                coincident_num += 1
            else:
                extreme = recalculated_extreme
                coincident_num = 1
        return np.sort(extreme), eps


class CoincidentNew:

    @staticmethod
    def __call__(
            extremum,
            index: np.ndarray[np.uint32],
            max_coincident=1,
            eps: int = 1,
    ):
        coincident_num = 1
        extreme = extremum(index, eps)
        while coincident_num < max_coincident:
            eps += 1
            recalculated_extreme = extremum(extreme, eps)
            if len(extreme) == len(recalculated_extreme):
                coincident_num += 1
            else:
                extreme = recalculated_extreme
                coincident_num = 1
        return np.sort(extreme), eps
