from functools import wraps

import numpy as np

__all__ = [
    'coincident',
    'min_extremum',
    'max_extremum',
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
