import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backup.extremum import coincident, max_extremum, min_extremum
from backup.merge_arg_sort import merge_arg_sort

N_ROW = 20_000
P = 0.9
N = int(N_ROW * P)

if __name__ == '__main__':
    df = pd.read_csv('../data/AUD_CHF.csv', nrows=N_ROW)
    close = df.Close.values[:N]

    # first iteration
    index_first = merge_arg_sort(close)
    max_temp_index_first, max_eps_first = coincident(3)(max_extremum)(index=index_first, eps=50)
    min_temp_index_first, min_eps_first = coincident(3)(min_extremum)(index=index_first, eps=50)
    max_values_first = close[max_temp_index_first]
    min_values_first = close[min_temp_index_first]
    index_temp_first = np.array(sorted(max_temp_index_first + min_temp_index_first))
    close_first = close[index_temp_first]
    index_merge_first = index_temp_first
    max_index_first = max_temp_index_first
    min_index_first = min_temp_index_first

    # second iteration
    index_second = merge_arg_sort(close_first)
    max_temp_index_second, max_eps_second = coincident(2)(max_extremum)(index=index_second, eps=1)
    min_temp_index_second, min_eps_second = coincident(2)(min_extremum)(index=index_second, eps=1)
    max_values_second = close_first[max_temp_index_second]
    min_values_second = close_first[min_temp_index_second]
    index_temp_second = np.array(sorted(max_temp_index_second + min_temp_index_second))
    close_second = close_first[index_temp_second]
    index_merge_second = index_temp_first[index_temp_second]
    max_index_second = index_temp_first[max_temp_index_second]
    min_index_second = index_temp_first[min_temp_index_second]

    # third iteration
    index_third = merge_arg_sort(close_second)
    max_temp_index_third, max_eps_third = coincident(2)(max_extremum)(index=index_third, eps=1)
    min_temp_index_third, min_eps_third = coincident(2)(min_extremum)(index=index_third, eps=1)
    max_values_third = close_second[max_temp_index_third]
    min_values_third = close_second[min_temp_index_third]
    index_temp_third = np.array(sorted(max_temp_index_third + min_temp_index_third))
    close_third = close_second[index_temp_third]
    index_merge_third = index_merge_second[index_temp_third]
    max_index_third = index_merge_second[max_temp_index_third]
    min_index_third = index_merge_second[min_temp_index_third]

    # fourth iteration
    index_fourth = merge_arg_sort(close_third)
    max_temp_index_fourth, max_eps_fourth = coincident(1)(max_extremum)(index=index_fourth, eps=2)
    min_temp_index_fourth, min_eps_fourth = coincident(1)(min_extremum)(index=index_fourth, eps=2)
    max_values_fourth = close_third[max_temp_index_fourth]
    min_values_fourth = close_third[min_temp_index_fourth]
    index_temp_fourth = np.array(sorted(max_temp_index_fourth + min_temp_index_fourth))
    close_fourth = close_third[index_temp_fourth]
    index_merge_fourth = index_merge_third[index_temp_fourth]
    max_index_fourth = index_merge_third[max_temp_index_fourth]
    min_index_fourth = index_merge_third[min_temp_index_fourth]

    plt.plot(df.Close.values, color='black')
    plt.plot(close)

    plt.scatter(
        max_index_first,
        max_values_first,
        color='r',
        s=20,
        label=f'max_eps:{max_eps_first} len:{len(max_index_first)}'
    )
    plt.scatter(
        min_index_first,
        min_values_first,
        color='g',
        s=20,
        label=f'min_eps:{min_eps_first} len:{len(min_index_first)}'
    )

    plt.scatter(
        max_index_second,
        max_values_second,
        color='r',
        s=60,
        label=f'max_eps:{max_eps_second} len:{len(max_index_second)}'
    )
    plt.scatter(
        min_index_second,
        min_values_second,
        color='g',
        s=60,
        label=f'min_eps:{min_eps_second} len:{len(min_index_second)}'
    )

    plt.scatter(
        max_index_third,
        max_values_third,
        color='r',
        s=150,
        label=f'max_eps:{max_eps_third} len:{len(max_index_third)}'
    )
    plt.scatter(
        min_index_third,
        min_values_third,
        color='g',
        s=150,
        label=f'min_eps={min_eps_third} len:{len(min_index_third)}'
    )

    plt.scatter(
        max_index_fourth,
        max_values_fourth,
        color='y',
        s=150,
        label=f'max_eps:{max_eps_fourth} len:{len(max_index_fourth)}'
    )
    plt.scatter(
        min_index_fourth,
        min_values_fourth,
        color='m',
        s=150,
        label=f'min_eps={min_eps_fourth} len:{len(min_index_fourth)}'
    )

    plt.grid()
    plt.legend()
    plt.show()
