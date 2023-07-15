from functools import wraps

import numpy as np

__all__ = [
    'coincident',
    'min_extremum',
    'max_extremum',
    'merge_extremum',
]


def coincident(max_coincident=1):
    def coincident_decorator(extremum):
        @wraps(extremum)
        def wrapper(*args, **kwargs):
            coincident_num = 1
            extreme = extremum(*args, **kwargs)
            while coincident_num < max_coincident:
                kwargs['eps'] += 1
                recalculated_extreme = extremum(*args, **kwargs)
                if extreme == recalculated_extreme:
                    coincident_num += 1
                else:
                    extreme = recalculated_extreme
                    coincident_num = 1
            return extreme, kwargs['eps']

        return wrapper

    return coincident_decorator


def min_extremum(*, index: np.ndarray, eps: int) -> list[int]:
    n, extreme_min = len(index), []
    for k in range(n):
        for l in range(1, k + 1):
            if abs(index[k] - index[k - l]) <= eps:
                break
        else:
            extreme_min.append(index[k])
    return extreme_min


def max_extremum(*, index: np.ndarray, eps: int) -> list[int]:
    n, extreme_max = len(index), []
    for k in range(n):
        for l in range(1, (n - k)):
            if abs(index[k] - index[k + l]) <= eps:
                break
        else:
            extreme_max.append(index[k])
    return extreme_max


def merge_extremum(extr_min, extr_max, value):
    extr = []
    i = j = 0
    status = 0
    i_min = j_max = None
    min_over, max_over = max(value) + 1, min(value) - 1
    value_min, value_max = min_over, max_over

    while i + j < len(extr_min) + len(extr_max):
        if i < len(extr_min) and j < len(extr_max):
            if extr_max[j] < extr_min[i]: status = -1
            if extr_max[j] > extr_min[i]: status = 1
            if extr_max[j] == extr_min[i]: status = 0

        if i >= len(extr_min): status = -1
        if j >= len(extr_max): status = 1

        if status >= 0:
            if value[extr_min[i]] < value_min:
                value_min = value[extr_min[i]]
                i_min = extr_min[i]
            if j_max is not None:
                extr.append(j_max)
            value_max = max_over
            i += 1
        else:
            if value[extr_max[j]] > value_max:
                value_max = value[extr_max[j]]
                j_max = extr_max[j]
            if i_min is not None:
                extr.append(i_min)
            value_min = min_over
            j += 1

    if status < 0:
        extr.append(j_max)
    else:
        extr.append(i_min)

    return extr
