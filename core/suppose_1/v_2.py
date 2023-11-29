import time
from typing import Callable

import numpy as np

from core import PlotTrendPoint, SplitTrendDetection
from core.matches_extremum import BaseMatchesOnArray

np.random.seed(9)


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
        return np.array(extreme), eps


def max_extremum(index: np.ndarray[np.uint32]):
    n = len(index)
    im = np.zeros(shape=(len(index), len(index)), dtype=np.int32)
    for i in range(n):
        i_min = 0
        j_min = 0
        v_min = n + 1
        for j in range(1, (n - i)):
            im[i][i + j] = abs(index[i] - index[i + j])
    #         if v_min > abs(index[i] - index[i + j]):
    #             v_min = abs(index[i] - index[i + j])
    #             i_min = i
    #             j_min = i + j
    #             if v_min == 1:
    #                 break
    #     im[i_min][j_min] = v_min
    # im[0][0] = 0
    return im


def main():
    data = np.random.uniform(0, 5, 10)
    index = np.argsort(data, kind="mergesort")
    print(data)
    print(index)
    eps = [1, 1]
    repeat = [2, 2]
    matches_extremum = MatchesOnInputArray()
    trend_points = SplitTrendDetection(
        values=data, test_size=len(data), coincident=matches_extremum
    )
    visualisation = PlotTrendPoint(trend_points)
    visualisation.plot_all_values()
    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_points.search_extremum(num_coincident=r, start_eps=e)
        visualisation.plot_extremum()

    print(trend_points.get_max_indexes(1))
    print(trend_points.get_max_indexes(2))
    print(trend_points.get_max_eps(1))
    print(trend_points.get_max_eps(2))

    im = max_extremum(index=index)
    # im = np.sort(a=im, axis=1)
    #
    # # find the first non-zero element of each row
    # first_nonzero = np.zeros(im.shape[0])
    # for i, row in enumerate(im):
    #     if len(row[np.nonzero(row)]):
    #         first_nonzero[i] = row[np.nonzero(row)][0]
    #
    # # sort the rows based on the first non-zero element
    # indices = np.lexsort((first_nonzero,))
    # im = im[indices]

    print("    :", end=" ")
    for elem in index:
        print(f"{elem:4d}", end=" ")
    print()
    for i, row in enumerate(im):
        # print(f"{index[indices[i]]:4d}", end=": ")
        print(f"{index[i]:4d}", end=": ")
        for elem in row:
            # if elem != 0:
            print(f"{elem:4d}", end=" ")
        print()

    print()
    print()

    data_1 = trend_points.get_max_values(1)
    ii_1 = trend_points.get_max_indexes(1)
    index_1 = np.argsort(data_1, kind="mergesort")
    im = max_extremum(index=index_1)
    # im = np.sort(a=im, axis=1)
    #
    # # find the first non-zero element of each row
    # first_nonzero = np.zeros(im.shape[0])
    # for i, row in enumerate(im):
    #     if len(row[np.nonzero(row)]):
    #         first_nonzero[i] = row[np.nonzero(row)][0]
    #
    # # sort the rows based on the first non-zero element
    # indices = np.lexsort((first_nonzero,))
    # im = im[indices]

    for i, row in enumerate(im):
        # print(f"{ii_1[index_1[indices[i]]]:4d}", end=": ")
        print(f"{index_1[i]:4d}", end=": ")
        for elem in row:
            # if elem != 0:
            print(f"{elem:4d}", end=" ")
        print()

    # visualisation.show(
    #     title="",
    #     from_date="",
    #     to_date="",
    #     split_date="",
    #     timeframe="",
    # )


if __name__ == "__main__":

    def extremum(index: np.ndarray[np.uint32]):
        n, extremes = len(index), []
        min_extr_diff, max_extr_diff = np.zeros((n,), dtype=np.uint32), np.zeros(
            (n,), dtype=np.uint32
        )

        # что бы минимальный элемент оказался в начале массива, разница для глобального минимума будет равна 0
        min_v = 0

        # что бы максимальный элемент оказался в конце массива, разница для глобального максимума будет равна n
        max_v = n

        _start_time = time.monotonic()
        for i in range(n):
            # min
            for j in range(1, i + 1):
                # print(
                #     f"Min: [{index[i]}, {index[i - j]}] ({abs(index[i] - index[i - j])})", end="\t"
                # )
                if abs(index[i] - index[i - j]) < min_v:
                    min_v = abs(index[i] - index[i - j])
                if min_v == 1:
                    break

            min_extr_diff[i] = min_v
            min_v = n
            # print()

            # max
            for j in range(1, (n - i)):
                # print(
                #     f"Max: [{index[i]}, {index[i + j]}] ({abs(index[i] - index[i + j])})", end="\t"
                # )
                if abs(index[i] - index[i + j]) < max_v:
                    max_v = abs(index[i] - index[i + j])
                if max_v == 1:
                    break

            max_extr_diff[i] = max_v
            max_v = n
            # print()
        _end_time = time.monotonic()
        print(_end_time - start_time)

        # print(f"Min: {min_extr_diff}")
        # print(f"Max: {max_extr_diff}")

        coincident = repeat[0]
        min_index_diff = np.argsort(min_extr_diff, kind="mergesort")
        # print(f"Min diff sort: {min_extr_diff[min_index_diff]}")
        start_min_index = 0
        for i in range(len(min_index_diff) - 1):
            if (
                min_extr_diff[min_index_diff[i + 1]] - min_extr_diff[min_index_diff[i]]
                >= coincident
            ):
                start_min_index = i + 1
                print("Main min eps:", min_extr_diff[min_index_diff[i]] + coincident - 1)
                break

        if start_min_index != 0:
            main_extr_min = [index[min_index_diff[0]]]
            for i in range(start_min_index, len(min_index_diff)):
                main_extr_min.append(index[min_index_diff[i]])
            print(f"Main min: {np.sort(np.array(main_extr_min))}")

        return min_extr_diff, max_extr_diff

    def max_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
        n, extreme_max = len(index), []
        for i in range(n):
            for j in range(1, (n - i)):
                # print(f"[{index[i]}, {index[i + j]}] ({abs(index[i] - index[i + j])})", end="\t")
                if abs(index[i] - index[i + j]) <= eps:
                    break
            else:
                extreme_max.append(index[i])
            # print()
        return np.array(extreme_max)

    def min_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
        n, extreme_min = len(index), []
        for i in range(n):
            for j in range(1, i + 1):
                # print(f"[{index[i]}, {index[i - j]}] ({abs(index[i] - index[i - j])})", end="\t")
                if abs(index[i] - index[i - j]) <= eps:
                    break
            else:
                extreme_min.append(index[i])
            # print()
        return np.array(extreme_min)

    data = np.random.uniform(0, 5, 10)
    index = np.argsort(data, kind="mergesort")

    start_time = time.monotonic()
    eps = [1]
    repeat = [2]
    matches_extremum = MatchesOnInputArray()
    trend_points = SplitTrendDetection(
        values=data, test_size=len(data), coincident=matches_extremum
    )
    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_points.search_extremum(num_coincident=r, start_eps=e)
    end_time = time.monotonic()
    print(1, end_time - start_time)
    # print(data)
    # print(index)

    print(f"Main min: {trend_points.get_min_indexes()}")
    print(f"Main max: {trend_points.get_max_indexes()}")
    # print(f"Main min: {np.sort(min_extremum(index=index, eps=19))}")
    # print(max_extremum(index=index, eps=1))

    print(f"Main min eps: {trend_points.get_min_eps()}")
    print(f"Main max eps: {trend_points.get_max_eps()}")
    start_time = time.monotonic()
    extremum(index=index)
    end_time = time.monotonic()
    print(2, end_time - start_time)
