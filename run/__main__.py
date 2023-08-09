import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from TradingMath.extremal import extremal_max, extremal_min
from core_math.extremum import coincident, max_extremum, min_extremum
from core_math.merge_arg_sort import merge_arg_sort
from core_math.trend_detection import CombinedExtremum, MergeExtremum
from matplotlib import pyplot as plt

P = 0.9
N_ROW = 10_000


def main_merge():
    df = pd.read_csv("../data/AUD_USD.csv", nrows=N_ROW)
    close = df.Close.values
    trend_merge = MergeExtremum(close, P)
    trend_merge.plot_values()

    trend_merge.main_extremum(num_coincident=3, start_eps=50)
    trend_merge.plot_extremum(1)

    trend_merge.main_extremum(num_coincident=2, start_eps=1)
    trend_merge.plot_extremum(2)

    trend_merge.main_extremum(num_coincident=2, start_eps=1)
    trend_merge.plot_extremum(3)
    trend_merge.trend_entry_point(3, 2)
    trend_merge.plot_trend(3)

    plt.grid()
    plt.legend()
    plt.show()


def main_combined():
    df = pd.read_csv("../data/AUD_USD.csv", nrows=N_ROW)
    close = df.Close.values
    trend_combined = CombinedExtremum(close, P)
    trend_combined.plot_values()

    trend_combined.main_extremum(num_coincident=3, start_eps=50)
    trend_combined.plot_extremum(1)

    trend_combined.main_extremum(num_coincident=2, start_eps=1)
    trend_combined.plot_extremum(2)

    trend_combined.main_extremum(num_coincident=2, start_eps=1)
    trend_combined.plot_extremum(3)
    trend_combined.trend_entry_point(3, 2)
    trend_combined.plot_trend(3)

    plt.grid()
    plt.legend()
    plt.show()


def main_correlation():
    directory_path = '../data/'
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            f = os.path.join(directory_path, file)
            print(f)
            df_all = pd.read_csv(f, chunksize=N_ROW)
            statistic = defaultdict(list)
            df: pd.DataFrame
            for df in df_all:
                close = df.Close.values
                index = merge_arg_sort(close)
                if len(index) != N_ROW:
                    break
                for i in range(1, 4):

                    max_index = extremal_max(index=index, eps=i)
                    min_index = extremal_min(index=index, eps=i)

                    statistic[f"max_len_eps_{i}"].append(len(max_index))
                    statistic[f"min_len_eps_{i}"].append(len(min_index))

                max_index, eps_max = coincident(3)(extremal_max)(index=index, eps=50)
                min_index, eps_min = coincident(3)(extremal_min)(index=index, eps=50)

                statistic["max_len_eps"].append(len(max_index))
                statistic["min_len_eps"].append(len(min_index))

                statistic["element"].append(len(index))
                statistic["max_eps_label"].append(eps_max)
                statistic["min_eps_label"].append(eps_min)

            st = pd.DataFrame(statistic)
            st.to_csv(f"eps_{file}", index=False)


if __name__ == "__main__":
    main_correlation()