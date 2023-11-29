from typing import Callable

import numpy as np

from core import CombinedTrendDetection, PlotTrendPoint, SplitTrendDetection
from core.matches_extremum import MatchesOnInputArray, MatchesOnRecalculatedArray

np.random.seed(1245)


def max_extremum(index: np.ndarray[np.uint32]):
    n = len(index)
    im = np.zeros(shape=(len(index), len(index)), dtype=np.int32)
    for i in range(n):
        for j in range(1, (n - i)):
            im[i][i + j] = abs(index[i] - index[i + j])

    return im


def main():
    n = 200
    data = np.random.uniform(-100, 100, n)
    index = np.argsort(data, kind="mergesort")
    eps = [1, 1]
    repeat = [2, 2]
    matches_extremum = MatchesOnInputArray()
    trend_points = SplitTrendDetection(values=data, test_size=n, coincident=matches_extremum)
    visualisation = PlotTrendPoint(trend_points)
    visualisation.plot_all_values()
    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_points.search_extremum(num_coincident=r, start_eps=e)
        visualisation.plot_extremum()

    print(trend_points.get_max_indexes(1))
    print(trend_points.get_max_indexes(2))
    print(trend_points.get_max_eps())

    im = max_extremum(index=index)
    # im = np.sort(a=im, axis=1)
    #
    # # find the first non-zero element of each row
    # first_nonzero = np.zeros(im.shape[0])
    # for i, row in enumerate(im):
    #     if len(row[np.nonzero(row)]):
    #         first_nonzero[i] = row[np.nonzero(row)][0]

    # sort the rows based on the first non-zero element
    # indices = np.lexsort((first_nonzero,))
    # im = im[indices]

    for i, row in enumerate(im):
        # print(f"{index[indices[i]]:4d}", end=": ")
        print(f"{index[i]:3d}", end=": ")
        for elem in row:
            if elem != 0:
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
        print(f"{ii_1[index_1[i]]:3d}", end=": ")
        for elem in row:
            if elem != 0:
                print(f"{elem:4d}", end=" ")
        print()

    visualisation.show(
        title="",
        from_date="",
        to_date="",
        split_date="",
        timeframe="",
    )


if __name__ == "__main__":
    main()
