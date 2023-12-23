import time
from typing import Callable

import numpy as np
from TradingMath.v1.essential_extremum import localize_extremes as v1_le
from TradingMath.v2.essential_extremum import localize_extremes as v2_le
from TradingMath.sort import argsort
from TradingMath.extremum import localize_minimals, localize_maximals

from core.matches_extremum import BaseMatchesOnArray

# np.random.seed(9)


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
        extremum: Callable[[np.ndarray[np.uint32], int, bool], np.ndarray[np.uint32]],
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
        extreme = extremum(index, eps, False)
        while coincident_num < max_coincident:
            eps += 1
            recalculated_extreme = extremum(index, eps, False)
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
    np.random.seed(23235)
    size = 2_000

    data = np.random.uniform(0, 100000, size)
    index = np.argsort(data, kind="mergesort").astype(np.int32)
    # index = np.array(
    #     [3, 6, 0, 2, 13, 5, 8, 10, 4, 16, 17, 19, 14, 11, 9, 18, 12, 1, 7, 15], dtype=np.int32
    # )
    repeat = 1
    eps = 2

    match = MatchesOnInputArray()
    s1 = time.monotonic()
    min_index_1, max_index_1, eps_min_1, eps_max_1 = v1_le(
        index=index,
        coincident=repeat,
        eps=eps,
        parallel=False,
    )
    # min_index_1, eps_min_1 = match(localize_minimals, index, repeat, eps)
    # max_index_1, eps_max_1 = match(localize_maximals, index, repeat, eps)
    t1 = time.monotonic() - s1
    print(t1)

    s1 = time.monotonic()
    min_index_2, max_index_2, eps_min_2, eps_max_2 = v2_le(
        index=index,
        coincident=repeat,
        eps=eps,
        parallel=False,
    )
    t2 = time.monotonic() - s1
    print(t2)

    # print(np.sort(min_index_1))
    # print(np.sort(min_index_2))

    # print(np.sort(max_index_1))
    # print(np.sort(max_index_2))

    print(eps_min_1, eps_min_2)
    print(eps_max_1, eps_max_2)

    print(np.all(np.sort(min_index_1) == np.sort(min_index_2)))
    print(np.all(np.sort(max_index_1) == np.sort(max_index_2)))

    print(t1 / t2)

    # match = MatchesOnInputArray()
    # s2 = time.monotonic()
    # min_1, min_eps = match(localize_minimals, index, repeat, eps)
    # max_1, max_eps = match(localize_maximals, index, repeat, eps)
    # print(time.monotonic() - s2)

    # s3 = time.monotonic()
    # trend = SplitTrendDetection(values=data, test_size=size, coincident=match)
    # trend.search_extremum(repeat, eps)
    # print(time.monotonic() - s3)

    # print(sorted(min_1))
    # print(sorted(min_index))
    # print(sorted(max_1))
    # print(sorted(max_index))
    # print(sorted(trend.get_min_indexes()))

    # print(min_eps, eps_min)
    # print(max_eps, eps_max)

    # print(np.all(sorted(min_1) == sorted(min_index)))
    # print(np.all(sorted(min_index) == sorted(trend.get_min_indexes())))
    # print(np.all(sorted(max_1) == sorted(max_index)))
    # print(np.all(sorted(max_index) == sorted(trend.get_max_indexes())))


if __name__ == "__main__":
    main()
