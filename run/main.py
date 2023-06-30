import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core_math.merge_arg_sort import merge_arg_sort
from core_math.extremum import coincident, max_extremum, min_extremum

if __name__ == '__main__':
    df = pd.read_csv('../data/AUD_CAD.csv', nrows=10000)
    date = df.Date.values[:5000]
    close = df.Close.values[:5000]

    # first iteration
    sort_index = merge_arg_sort(close)
    max_index, max_eps = coincident(4)(max_extremum)(index=sort_index, eps=10)
    min_index, min_eps = coincident(4)(min_extremum)(index=sort_index, eps=10)
    max_values = close[max_index]
    min_values = close[min_index]

    # second iteration
    index_second = np.array(sorted(max_index + min_index))
    close_second = close[index_second]
    sort_index_second = merge_arg_sort(close_second)
    max_index_second, max_eps_second = coincident(2)(max_extremum)(index=sort_index_second, eps=5)
    min_index_second, min_eps_second = coincident(2)(min_extremum)(index=sort_index_second, eps=5)
    max_values_second = close[index_second[max_index_second]]
    min_values_second = close[index_second[min_index_second]]

    # third iteration
    index_third = np.array(sorted(index_second[max_index_second] + index_second[min_index_second]))
    close_third = close[index_third]
    sort_index_third = merge_arg_sort(close_third)
    max_index_third, max_eps_third = coincident(2)(max_extremum)(index=sort_index_third, eps=1)
    min_index_third, min_eps_third = coincident(2)(min_extremum)(index=sort_index_third, eps=1)
    max_values_third = close[index_third[max_index_third]]
    min_values_third = close[index_third[min_index_third]]

    plt.plot(df.Close.values, color='black')
    plt.plot(close)
    plt.scatter(max_index, max_values, color='r', s=20, label=f'max_eps={max_eps}')
    plt.scatter(min_index, min_values, color='g', s=20, label=f'min_eps={min_eps}')

    plt.scatter(index_second[max_index_second], max_values_second, color='r', s=60, label=f'max_eps={max_eps_second}')
    plt.scatter(index_second[min_index_second], min_values_second, color='g', s=60, label=f'min_eps={min_eps_second}')

    plt.scatter(index_third[max_index_third], max_values_third, color='r', s=100, label=f'max_eps={max_eps_third}')
    plt.scatter(index_third[min_index_third], min_values_third, color='g', s=100, label=f'min_eps={min_eps_third}')
    plt.legend()
    plt.show()
