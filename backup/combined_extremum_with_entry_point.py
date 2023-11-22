import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backup.extremum import coincident, max_extremum, min_extremum
from backup.merge_arg_sort import merge_arg_sort

N_ROW = 10_000
P = 0.9
N = int(N_ROW * P)

if __name__ == '__main__':
    df = pd.read_csv('../data/AUD_CAD.csv', nrows=N_ROW)
    index = df.index[:N]
    close_1 = df.Close.values[:N]

    # first iteration
    index_1 = merge_arg_sort(close_1)
    max_index_1, max_eps_1 = coincident(3)(max_extremum)(index=index_1, eps=30)
    min_index_1, min_eps_1 = coincident(3)(min_extremum)(index=index_1, eps=30)

    max_values_1 = close_1[max_index_1]
    min_values_1 = close_1[min_index_1]

    # second iteration
    index_temp_2 = np.sort(np.hstack((min_index_1, max_index_1)), kind="mergesort")
    close_2 = close_1[index_temp_2]
    index_2 = merge_arg_sort(close_2)

    max_index_2, max_eps_2 = coincident(2)(max_extremum)(index=index_2, eps=1)
    min_index_2, min_eps_2 = coincident(2)(min_extremum)(index=index_2, eps=1)
    max_values_2 = close_2[max_index_2]
    min_values_2 = close_2[min_index_2]

    # third iteration
    index_temp_3 = np.sort(np.hstack((min_index_2, max_index_2)), kind="mergesort")
    close_3 = close_2[index_temp_3]
    index_3 = merge_arg_sort(close_3)

    max_index_3, max_eps_3 = coincident(2)(max_extremum)(index=index_3, eps=1)
    min_index_3, min_eps_3 = coincident(2)(min_extremum)(index=index_3, eps=1)
    max_values_3 = close_3[max_index_3]
    min_values_3 = close_3[min_index_3]

    # Downtrend->Uptrend
    min_eps_trend = 2
    min_index_temp_trend = np.sort(min_index_3, kind="mergesort")
    min_close_temp_trend = close_3[min_index_temp_trend]
    min_index_trend = merge_arg_sort(min_close_temp_trend)

    min_index_trend_point = min_extremum(index=min_index_trend, eps=min_eps_trend)
    min_values_fourth = min_close_temp_trend[min_index_trend_point]

    # Uptrend->Downtrend
    max_eps_trend = 2
    max_index_temp_trend = np.sort(max_index_3, kind="mergesort")
    max_close_temp_trend = close_3[max_index_temp_trend]
    max_index_trend = merge_arg_sort(max_close_temp_trend)

    max_index_trend_point = max_extremum(index=max_index_trend, eps=max_eps_trend)
    max_values_trend_point = max_close_temp_trend[max_index_trend_point]

    #############################################

    plt.plot(df.Close.values, color='black')
    plt.plot(close_1)

    plt.scatter(
        max_index_1,
        max_values_1,
        color='r',
        s=10,
        label=f'max_eps:{max_eps_1} len:{len(max_index_1)}'
    )
    plt.scatter(
        min_index_1,
        min_values_1,
        color='g',
        s=10,
        label=f'min_eps:{min_eps_1} len:{len(min_index_1)}'
    )

    plt.scatter(
        index_temp_2[max_index_2],
        max_values_2,
        color='r',
        s=60,
        label=f'max_eps:{max_eps_2} len:{len(max_index_2)}'
    )
    plt.scatter(
        index_temp_2[min_index_2],
        min_values_2,
        color='g',
        s=60,
        label=f'min_eps:{min_eps_2} len:{len(min_index_2)}'
    )

    plt.scatter(
        index_temp_2[index_temp_3[max_index_3]],
        max_values_3,
        color='r',
        s=150,
        label=f'max_eps:{max_eps_3} len:{len(max_index_3)}'
    )
    plt.scatter(
        index_temp_2[index_temp_3[min_index_3]],
        min_values_3,
        color='g',
        s=150,
        label=f'min_eps={min_eps_3} len:{len(min_index_3)}'
    )

    plt.scatter(
        index_temp_2[index_temp_3[max_index_temp_trend[max_index_trend_point]]],
        max_values_trend_point,
        color='y',
        s=200,
        label=f'max_eps:{max_eps_trend} len:{len(max_index_trend_point)}'
    )
    plt.scatter(
        index_temp_2[index_temp_3[min_index_temp_trend[min_index_trend_point]]],
        min_values_fourth,
        color='m',
        s=200,
        label=f'min_eps={min_eps_trend} len:{len(min_index_trend_point)}'
    )

    plt.grid()
    plt.legend()
    plt.show()
