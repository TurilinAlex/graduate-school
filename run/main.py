import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_math.extremum import coincident, max_extremum, min_extremum
from core_math.merge_arg_sort import merge_arg_sort

if __name__ == '__main__':
    df = pd.read_csv('../data/EUR_AUD.csv', nrows=10_000)
    index = df.index[:5_000]
    close = df.Close.values[:5_000]

    # first iteration
    index_first = merge_arg_sort(close)
    max_index_first, max_eps_first = coincident(4)(max_extremum)(index=index_first, eps=34)
    min_index_first, min_eps_first = coincident(4)(min_extremum)(index=index_first, eps=34)
    max_values_first = close[max_index_first]
    min_values_first = close[min_index_first]

    # second iteration
    index_temp_second = np.array(sorted(min_index_first + max_index_first))
    close_temp_second = close[index_temp_second]
    index_second = merge_arg_sort(close_temp_second)

    max_index_second, max_eps_second = coincident(2)(max_extremum)(index=index_second, eps=1)
    min_index_second, min_eps_second = coincident(2)(min_extremum)(index=index_second, eps=1)
    max_values_second = close[index_temp_second[max_index_second]]
    min_values_second = close[index_temp_second[min_index_second]]

    # third iteration
    index_temp_third = np.array(sorted(min_index_second + max_index_second))
    close_temp_third = close[index_temp_second[index_temp_third]]
    index_third = merge_arg_sort(close_temp_third)

    max_index_third, max_eps_third = coincident(2)(max_extremum)(index=index_third, eps=1)
    min_index_third, min_eps_third = coincident(2)(min_extremum)(index=index_third, eps=1)
    max_values_third = close[index_temp_second[index_temp_third[max_index_third]]]
    min_values_third = close[index_temp_second[index_temp_third[min_index_third]]]

    plt.plot(df.Close.values, color='black')
    plt.plot(close)

    plt.scatter(
        max_index_first,
        max_values_first,
        color='r',
        s=10,
        label=f'max_eps:{max_eps_first} len:{len(max_index_first)}'
    )
    plt.scatter(
        min_index_first,
        min_values_first,
        color='g',
        s=10,
        label=f'min_eps:{min_eps_first} len:{len(min_index_first)}'
    )

    plt.scatter(
        index_temp_second[max_index_second],
        max_values_second,
        color='r',
        s=60,
        label=f'max_eps:{max_eps_second} len:{len(max_index_second)}'
    )
    plt.scatter(
        index_temp_second[min_index_second],
        min_values_second,
        color='g',
        s=60,
        label=f'min_eps:{min_eps_second} len:{len(min_index_second)}'
    )

    plt.scatter(
        index_temp_second[index_temp_third[max_index_third]],
        max_values_third,
        color='r',
        s=150,
        label=f'max_eps:{max_eps_third} len:{len(max_index_third)}'
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
