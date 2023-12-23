import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from core.extremum import min_extremum, max_extremum
from core.matches_extremum import BaseMatchesOnArray
from core.sort import argsort
from core.trend import CombinedTrendDetection, SplitTrendDetection


class MatchesOnInputArray(BaseMatchesOnArray):
    @staticmethod
    def __call__(
        extremum: Callable[[np.ndarray[np.uint32], int], np.ndarray[np.uint32]],
        index: np.ndarray[np.uint32],
        max_coincident=1,
        eps: int = 1,
    ) -> tuple[np.ndarray[np.uint32], int]:
        """
        Выделяет экстремумы при увеличении радиусе локализации `всегда` во входном массиве

        :param extremum: Оператор выделения экстремумов из последовательности индексов
        :param index: Массив индексов
        :param max_coincident: Требуемое количество последовательно равных массивов экстремумов,
        что бы их считать существенными
        :param eps: Начальный радиус локализации ``eps > 0``
        :return: Возвращает массив индексов существенных экстремумов и радиус локализации при котором он был выделен
        """

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


def search_main_extremum(index: np.ndarray[np.uint32], coincident: int, eps: int = 1, debug=False):
    n = len(index)
    min_extr_diff, max_extr_diff = np.empty_like(index, dtype=np.uint32), np.empty_like(
        index, dtype=np.uint32
    )
    min_diff_count, max_diff_count = np.zeros((n + 1,), dtype=np.int32), np.zeros(
        (n + 1,), dtype=np.int32
    )

    for_print = np.zeros((n, n), dtype=np.int32)
    for_print[0, 0] = n
    for_print[n - 1, n - 1] = n

    # region Вычисление разницы между индексами

    for i in range(n):
        min_v = n
        max_v = n
        # min
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            for_print[i, i - j] = diff
            if diff < min_v:
                min_v = diff
            if min_v <= 1:
                break

        min_extr_diff[i] = min_v
        min_diff_count[min_v] += 1

        # max
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            for_print[i, j + i] = diff
            if diff < max_v:
                max_v = diff
            if max_v <= 1:
                break
        max_extr_diff[i] = max_v
        max_diff_count[max_v] += 1

    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных минимумов по заданному числу совпадений

    count_zero = 0
    __i_last = eps
    for i in range(eps + 1, n):
        if count_zero >= coincident - 1:
            eps_min = __i_last + coincident - 1
            break
        if min_diff_count[i] == 0:
            count_zero += 1
        else:
            count_zero = 0
            __i_last = i
    else:
        eps_min = __i_last + coincident - 1

    if eps_min >= min_extr_diff[0]:
        __extr_min = np.array([index[0]])
    else:
        k = 0
        __extr_min = np.empty_like(index)
        for i in range(n):
            if min_extr_diff[i] > eps_min:
                __extr_min[k] = index[i]
                k += 1
        __extr_min = __extr_min[:k]

    # endregion Поиск главных локальных минимумов по заданному числу совпадений

    # region Поиск главных локальных максимумов по заданному числу совпадений

    count_zero = 0
    __i_last = eps
    for i in range(eps + 1, n):
        if count_zero >= coincident - 1:
            eps_max = __i_last + coincident - 1
            break
        if max_diff_count[i] == 0:
            count_zero += 1
        else:
            count_zero = 0
            __i_last = i
    else:
        eps_max = __i_last + coincident - 1

    if eps_max >= max_extr_diff[n - 1]:
        __extr_max = np.array([index[n - 1]])
    else:
        k = 0
        __extr_max = np.empty_like(index)
        for i in range(n):
            if max_extr_diff[i] > eps_max:
                __extr_max[k] = index[i]
                k += 1
        __extr_max = __extr_max[:k]

    # endregion Поиск главных локальных максимумов по заданному числу совпадений

    if debug:
        print(f"################## {coincident=:5d}, {eps=:5d} ##################")

        print(f"\t   ", end="")
        for i in index:
            print(f"{i:5d}", end="")
        print("\t\t Min Diff\t\tMax Diff")
        for i in range(n):
            print(f"{index[i]:5d}: ", end="")
            for j in range(n):
                if i == j == 0:
                    print(f"{for_print[i, j]:5d}", end="")
                    continue
                if i == j == n - 1:
                    print(f"{for_print[i, j]:5d}", end="")
                    continue
                if i == j:
                    print("    _", end="")
                    continue
                print(f"{for_print[i, j]:5d}", end="")
            print(f"\t\t\t{min_extr_diff[i]:5d}\t\t   {max_extr_diff[i]:5d}")
        print()

        print()
        print(f"\t\t\t", end="  ")
        for i in range(n + 1):
            print(f"{i:5d}", end="")
        print()
        print(f"min_extr_diff ", end="")
        for i in range(n):
            print(f"{min_extr_diff[i]:5d}", end="")
        print()
        print(f"min_diff_count", end="")
        for i in range(n + 1):
            print(f"{min_diff_count[i]:5d}", end="")
        print()
        print()
        print()
        print(f"\t\t\t", end="  ")
        for i in range(n + 1):
            print(f"{i:5d}", end="")
        print()
        print(f"max_extr_diff ", end="")
        for i in range(n):
            print(f"{max_extr_diff[i]:5d}", end="")
        print()
        print(f"max_diff_count", end="")
        for i in range(n + 1):
            print(f"{max_diff_count[i]:5d}", end="")
        print()
        print()
        print()
        print(
            f"Min extremum index: {__extr_min}\t\tMin extremum eps: {eps_min} Coincident: {coincident}"
        )
        print(
            f"Max extremum index: {__extr_max}\t\tMax extremum eps: {eps_max} Coincident: {coincident}"
        )

    return np.sort(__extr_min), np.sort(__extr_max), eps_min, eps_max


if __name__ == "__main__":
    # e = np.array([3, 6, 0, 2, 13, 5, 8, 10, 4, 16, 17, 19, 14, 11, 9, 18, 12, 1, 7, 15])
    np.random.seed(1273)
    size = 50
    a = np.array([np.random.randint(0, 20) for _ in range(size)])
    e = argsort(a)
    #
    # print(f"a={np.arange(size, dtype=np.int32)}")
    # print(f"a={a}")
    # print(f"e={e}")
    # e = np.array([3, 5, 7, 9, 1, 0, 6, 4, 2, 8])
    # e = np.array([8, 12, 14, 19, 1, 0, 13, 9, 5, 16])
    repeat = 1
    start_eps = 2

    match = MatchesOnInputArray()
    s1 = time.monotonic()
    __extr_min_1, _eps_min_1 = match(min_extremum, e, repeat, start_eps)
    __extr_max_1, _eps_max_1 = match(max_extremum, e, repeat, start_eps)
    e1 = time.monotonic() - s1

    s2 = time.monotonic()
    __extr_min_2, __extr_max_2, _eps_min_2, _eps_max_2 = search_main_extremum(
        e, repeat, start_eps, True
    )
    e2 = time.monotonic() - s2

    assert _eps_min_1 == _eps_min_2, print(f"{_eps_min_1=}, {_eps_min_2=} | {repeat=} {start_eps=}")
    assert _eps_max_1 == _eps_max_2, print(f"{_eps_max_1=}, {_eps_max_2=} | {repeat=} {start_eps=}")

    assert np.all(__extr_min_1 == __extr_min_2), print(f"{__extr_min_1=}\n\t{__extr_min_2}")
    assert np.all(__extr_max_1 == __extr_max_2), print(f"{__extr_max_1=}\n\t{__extr_max_2}")

    trend = CombinedTrendDetection(a, size, match)
    trend.search_extremum(repeat, start_eps).search_extremum(repeat, start_eps)

    print(trend.get_min_indexes(1))
    print(trend.get_min_indexes(2))
