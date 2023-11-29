import time
from typing import Callable

import numpy as np
from TradingMath.extremal import (
    extremal_min,
    extremal_max,
    search_main_extremum,
    argsort,
)

from core.matches_extremum import BaseMatchesOnArray
from core.trend import SplitTrendDetection

# from core.sort import argsort


np.random.seed(9)


def min_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
    """
    Оператор идентификации экстремумов (локальных минимумов) по массиву индексов с заданным радиусом локализации

    :param index: Входной массив индексов
    :param eps: Радиус локализации ``eps > 0``
    :return: Возвращает массив индексов экстремумов (локальных минимумов) с данным радиусом локализации ``eps``
    """
    n = len(index)
    extreme_min = []
    for i in range(n):
        for j in range(1, i + 1):
            if abs(index[i] - index[i - j]) <= eps:
                break
        else:
            extreme_min.append(index[i])
    return np.array(extreme_min)


def max_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
    """
    Оператор идентификации экстремумов (локальных максимумов) по массиву индексов с заданным радиусом локализации

    :param index: Входной массив индексов
    :param eps: Радиус локализации ``eps > 0``
    :return: Возвращает массив индексов экстремумов (локальных минимумов) с данным радиусом локализации ``eps``
    """
    n, extreme_max = len(index), []
    for i in range(n):
        for j in range(1, (n - i)):
            if abs(index[i] - index[i + j]) <= eps:
                break
        else:
            extreme_max.append(index[i])
    return np.array(extreme_max)


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
        extreme = extremum(index, eps, parallel=True)
        while coincident_num < max_coincident:
            eps += 1
            recalculated_extreme = extremum(index, eps, parallel=True)
            if len(extreme) == len(recalculated_extreme):
                coincident_num += 1
            else:
                extreme = recalculated_extreme
                coincident_num = 1
        return np.sort(extreme), eps


def search_main_extremum_1(index: np.ndarray[np.uint32], coincident: int, eps: int = 1):
    n = len(index)
    min_extr_diff, max_extr_diff = np.zeros((n,), dtype=np.uint32), np.zeros((n,), dtype=np.uint32)

    min_v, max_v = n, n

    # region Вычисление разницы между индексами
    _s1 = time.monotonic()

    for i in range(n):
        # min
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            if diff < min_v:
                min_v = diff
            if min_v == 1:
                break

        min_extr_diff[i] = min_v
        min_v = n

        # max
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v == 1:
                break

        max_extr_diff[i] = max_v
        max_v = n
    print(time.monotonic() - _s1)
    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных минимумов по заданному числу совпадений

    min_split_index = 1
    min_extr_diff_index = np.argsort(min_extr_diff, kind="mergesort")[::-1]
    for i in range(len(min_extr_diff_index) - 1, 0, -1):
        if min_extr_diff[min_extr_diff_index[i - 1]] <= eps:
            continue
        if (
            min_extr_diff[min_extr_diff_index[i - 1]] - min_extr_diff[min_extr_diff_index[i]]
            >= coincident
        ):
            min_split_index = i
            min_eps = min_extr_diff[min_extr_diff_index[i]] + coincident - 1
            print(f"Min split index: {min_split_index}, Min eps: {min_eps}")
            break

    min_main_index = np.zeros((min_split_index,), dtype=np.uint32)
    for i in range(min_split_index):
        min_main_index[i] = index[min_extr_diff_index[i]]

    # endregion Поиск главных локальных минимумов по заданному числу совпадений

    # region Поиск главных локальных максимумов по заданному числу совпадений

    max_split_index = len(max_extr_diff) - 1
    max_extr_diff_index = np.argsort(max_extr_diff, kind="mergesort")
    for i in range(len(max_extr_diff_index) - 1):
        if max_extr_diff[max_extr_diff_index[i + 1]] <= eps:
            continue
        if (
            max_extr_diff[max_extr_diff_index[i + 1]] - max_extr_diff[max_extr_diff_index[i]]
            >= coincident
        ):
            max_split_index = i + 1
            max_eps = max_extr_diff[max_extr_diff_index[i]] + coincident - 1
            print(f"Max split index: {max_split_index}, Max eps: {max_eps}")
            break

    max_main_index = np.zeros((len(max_extr_diff_index) - max_split_index,), dtype=np.uint32)
    for i in range(max_split_index, len(max_extr_diff_index)):
        max_main_index[i - max_split_index] = index[max_extr_diff_index[i]]

    # endregion Поиск главных локальных максимумов по заданному числу совпадений

    return np.sort(min_main_index), np.sort(max_main_index)


def main():
    size = 10_000
    data = np.random.uniform(0, 50, size)
    index = np.zeros((size,), dtype=np.int64)
    s0 = time.monotonic()
    argsort(data, index)
    print(time.monotonic() - s0)
    # repeat = 4
    # eps = 10
    # s1 = time.monotonic()
    # min_index, max_index, eps_min, eps_max = search_main_extremum(
    #     index=index,
    #     coincident=repeat,
    #     eps=eps,
    #     parallel=True,
    # )
    # print(time.monotonic() - s1)
    #
    # match = MatchesOnInputArray()
    # s2 = time.monotonic()
    # min_1, min_eps = match(extremal_min, index, repeat, eps)
    # max_1, max_eps = match(extremal_max, index, repeat, eps)
    # print(time.monotonic() - s2)
    #
    # s3 = time.monotonic()
    # trend = SplitTrendDetection(values=data, test_size=size, coincident=match)
    # trend.search_extremum(repeat, eps)
    # print(time.monotonic() - s3)
    #
    # print(sorted(min_1))
    # print(sorted(min_index))
    # print(sorted(trend.get_min_indexes()))
    #
    # print(min_eps, eps_min)
    # print(max_eps, eps_max)
    #
    # print(np.all(sorted(min_1) == sorted(min_index)))
    # print(np.all(sorted(min_index) == sorted(trend.get_min_indexes())))
    # print(np.all(sorted(max_1) == sorted(max_index)))
    # print(np.all(sorted(max_index) == sorted(trend.get_max_indexes())))
    #

def count_sort(value):
    e = [0] * len(value)
    for j in range(len(value)):
        k = len(value) - 1
        for i in range(j):
            if value[j] >= value[i]:
                k -= 1
        for i in range(j + 1, len(value)):
            if value[j] > value[i]:
                k -= 1
        e[k] = j
    return e


if __name__ == "__main__":
    main()
    # np.random.seed(124)
    # size = 20_000
    # data = np.random.uniform(0, 5_000_000, size)
    #
    # index1 = np.zeros((size,), dtype=np.int64)
    # s = time.monotonic()
    # argsort(data, index1)
    # print(time.monotonic() - s)
    #
    # s = time.monotonic()
    # index2 = np.argsort(data, kind="mergesort")
    # print(time.monotonic() - s)
    # print(all(index1 == index2))

    # s = time.monotonic()
    # ext1 = extremal_max(index1, 1, True)
    # print(time.monotonic() - s)
    #
    # s = time.monotonic()
    # ext2 = max_extremum(index=index1, eps=1)
    # print(time.monotonic() - s)
    # print(all(ext1 == ext2))
    # print(ext1)
    # print(ext2)
