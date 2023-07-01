import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_math.extremum import coincident, max_extremum, min_extremum
from core_math.merge_arg_sort import merge_arg_sort

NUM_COINCIDENT = [2, 1, 1]
EPS = [14, 1, 1]


def main_min_extremum(input_sort_index: np.ndarray, values: np.ndarray) -> None:
    min_index_first, min_eps_first = coincident(NUM_COINCIDENT[0])(min_extremum)(index=input_sort_index, eps=EPS[0])
    min_values_first = values[min_index_first]

    # second iteration
    index_temp_second = np.array(sorted(min_index_first))
    close_temp_second = values[index_temp_second]
    index_second = merge_arg_sort(close_temp_second)

    min_index_second, min_eps_second = coincident(NUM_COINCIDENT[1])(min_extremum)(index=index_second, eps=EPS[1])
    min_values_second = values[index_temp_second[min_index_second]]

    # third iteration
    index_temp_third = np.array(sorted(min_index_second))
    close_temp_third = values[index_temp_second[index_temp_third]]
    index_third = merge_arg_sort(close_temp_third)

    min_index_third, min_eps_third = coincident(NUM_COINCIDENT[2])(min_extremum)(index=index_third, eps=EPS[2])
    min_values_third = values[index_temp_second[index_temp_third[min_index_third]]]

    plt.plot(df.Close.values, color='black')
    plt.plot(values)

    plt.scatter(
        min_index_first,
        min_values_first,
        color='g',
        s=10,
        label=f'min_eps:{min_eps_first} len:{len(min_index_first)}'
    )

    plt.scatter(
        index_temp_second[min_index_second],
        min_values_second,
        color='g',
        s=60,
        label=f'min_eps:{min_eps_second} len:{len(min_index_second)}'
    )

    plt.scatter(
        index_temp_second[index_temp_third[min_index_third]],
        min_values_third,
        color='g',
        s=150,
        label=f'min_eps={min_eps_third} len:{len(min_index_third)}'
    )
    plt.legend()
    plt.show()


def main_max_extremum(input_sort_index: np.ndarray, values: np.ndarray) -> None:
    max_index_first, max_eps_first = coincident(NUM_COINCIDENT[0])(max_extremum)(index=input_sort_index, eps=EPS[0])
    max_values_first = values[max_index_first]

    # second iteration
    index_temp_second = np.array(sorted(max_index_first))
    close_temp_second = values[index_temp_second]
    index_second = merge_arg_sort(close_temp_second)

    max_index_second, max_eps_second = coincident(NUM_COINCIDENT[1])(max_extremum)(index=index_second, eps=EPS[1])
    max_values_second = values[index_temp_second[max_index_second]]

    # third iteration
    index_temp_third = np.array(sorted(max_index_second))
    close_temp_third = values[index_temp_second[index_temp_third]]
    index_third = merge_arg_sort(close_temp_third)

    max_index_third, max_eps_third = coincident(NUM_COINCIDENT[2])(max_extremum)(index=index_third, eps=EPS[2])
    max_values_third = values[index_temp_second[index_temp_third[max_index_third]]]

    plt.plot(df.Close.values, color='black')
    plt.plot(values)

    plt.scatter(
        max_index_first,
        max_values_first,
        color='r',
        s=10,
        label=f'max_eps:{max_eps_first} len:{len(max_index_first)}'
    )

    plt.scatter(
        index_temp_second[max_index_second],
        max_values_second,
        color='r',
        s=60,
        label=f'max_eps:{max_eps_second} len:{len(max_index_second)}'
    )

    plt.scatter(
        index_temp_second[index_temp_third[max_index_third]],
        max_values_third,
        color='r',
        s=150,
        label=f'max_eps:{max_eps_third} len:{len(max_index_third)}'
    )

    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 10_000
    df = pd.read_csv('../data/EUR_AUD.csv', nrows=n)
    index = df.index[:n // 2]
    close = df.Close.values[:n // 2]

    index_first = merge_arg_sort(close)
    main_max_extremum(index_first, close)
    main_min_extremum(index_first, close)
