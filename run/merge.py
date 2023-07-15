import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(1234567)


def merge_arg_sort(values: np.ndarray | list) -> np.ndarray:
    n = len(values)
    index_temp = np.zeros((n,), np.int32)
    index = np.arange(n)

    h = 1
    status = 0
    while h <= n:
        for j in range(0, n - h, 2 * h):
            ls = j
            le = j + h - 1
            rs = j + h
            re = min(j + 2 * h - 1, n - 1)
            index_temp[ls:re + 1] = index[ls:re + 1]
            i1 = ls
            j1 = rs
            for k in range(i1, re + 1):

                if i1 <= le and j1 <= re:
                    if values[index_temp[j1]] == values[index_temp[i1]]: status = 0
                    if values[index_temp[j1]] < values[index_temp[i1]]: status = -1
                    if values[index_temp[j1]] > values[index_temp[i1]]: status = 1

                if i1 > le: status = -1
                if j1 > re: status = 1

                if status >= 0:
                    index[k] = index_temp[i1]
                    i1 += 1
                else:
                    index[k] = index_temp[j1]
                    j1 += 1
        h *= 2
    return index


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


def main():
    a = [random.randint(-20, 200) for _ in range(100)]
    e = merge_arg_sort(a)
    e_min = min_extremum(index=e, eps=5)
    e_max = max_extremum(index=e, eps=5)
    print(sorted(e_min), sorted(e_max))
    e_extremum = (merge_extremum(sorted(e_min), sorted(e_max), a))
    print(e_extremum)

    e_min_value = [a[i] for i in e_min]
    e_max_value = [a[i] for i in e_max]
    e_extremum_value = [a[i] for i in e_extremum]

    plt.plot(a)
    plt.plot(e_extremum, e_extremum_value)
    plt.scatter(e_min, e_min_value, c='g')
    plt.scatter(e_max, e_max_value, c='r')
    plt.show()


if __name__ == '__main__':
    main()
