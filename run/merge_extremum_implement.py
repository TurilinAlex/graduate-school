import random

import matplotlib.pyplot as plt

from core_math.extremum import max_extremum, min_extremum
from core_math.merge_arg_sort import merge_arg_sort

random.seed(123)


def merge_extremum(extr_min, extr_max, value):
    extr = []
    i = j = 0
    status = 0
    while i + j < len(extr_min) + len(extr_max):
        if i < len(extr_min) and j < len(extr_max):
            if value[extr_min[i]] == value[extr_max[j]]: status = 0
            if value[extr_min[i]] < value[extr_max[j]]: status = -1
            if value[extr_min[i]] > value[extr_max[j]]: status = 1

        if i >= len(extr_min): status = -1
        if j >= len(extr_max): status = 1

        if status >= 0:
            extr.append(extr_min[i])
            i += 1
        else:
            extr.append(extr_max[j])
            j += 1

    return extr


def main():
    a = [random.randint(-20, 200) for _ in range(100)]
    e = merge_arg_sort(a)
    e_min = min_extremum(index=e, eps=5)
    e_max = max_extremum(index=e, eps=5)
    e_extremum = sorted(merge_extremum(e_min, e_max, a))

    e_min_value = [a[i] for i in e_min]
    e_max_value = [a[i] for i in e_max]
    e_extremum_value = [a[i] for i in e_extremum]

    print(e_min, e_min_value)
    print(e_max, e_max_value)

    plt.plot(a)
    plt.plot(e_extremum, e_extremum_value)
    plt.scatter(e_min, e_min_value, c='g')
    plt.scatter(e_max, e_max_value, c='r')
    plt.show()


if __name__ == '__main__':
    main()
