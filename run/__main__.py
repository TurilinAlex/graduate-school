import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import core_math.extremum as ext_coin
from core_math.trend_detection import CombinedExtremum, MergeExtremum, Visualisation, SplitExtremum


def main_merge():
    close = df.Close.values
    dates = df.Date.values

    trend_combined = MergeExtremum(close, P, coincident_num)
    viz = Visualisation(trend_combined)
    viz.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_combined.search_extremum(num_coincident=r, start_eps=e)
        trend_combined.search_change_trend_point(eps=2)
        viz.plot_extremum()
        viz.plot_combined_extremum()

    viz.plot_change_trend()
    viz.show(title=file, from_date=dates[0], to_date=dates[-1], split_date=dates[P], timeframe="1 minute")


def main_combined():
    close = df.Close.values
    dates = df.Date.values
    trend_combined = CombinedExtremum(close, P, coincident_num)
    viz = Visualisation(trend_combined)
    viz.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_combined.search_extremum(num_coincident=r, start_eps=e)
        trend_combined.search_change_trend_point(eps=2)
        viz.plot_extremum()
        viz.plot_combined_extremum()

    viz.plot_change_trend()
    viz.show(title=file, from_date=dates[0], to_date=dates[-1], split_date=dates[P], timeframe="1 minute")


def main_split():
    close = df.Close.values
    dates = df.Date.values
    trend_combined = SplitExtremum(close, P, coincident_num)
    viz = Visualisation(trend_combined)
    viz.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_combined.search_extremum(num_coincident=r, start_eps=e)
        trend_combined.search_change_trend_point(eps=2)
        viz.plot_extremum()
        viz.plot_combined_extremum()

    viz.plot_change_trend()
    viz.show(title=file, from_date=dates[0], to_date=dates[-1], split_date=dates[P], timeframe="1 minute")


def calculate_up_entry_point(trend):
    last_value = trend.get_all_values()[-1]
    iter_trend = trend.get_current_iter()
    all_values = trend.get_all_values()

    trend.search_down2up_trend_point(iter_trend, 2)
    trend.plot_down2up_trend_point(iter_trend)

    index = trend.get_down2up_trend_indexes(iter_trend)
    value = trend.get_down2up_trend_values(iter_trend)
    diff = value - last_value
    negative_diff = diff[diff < 0]
    negative_index = index[diff < 0]
    negative_value = value[diff < 0]
    print(list(zip(value, index)))
    print(list(zip(diff, index)))
    print(list(zip(negative_value, negative_index)))
    # plt.scatter(negative_index, negative_value, c="red", s=50)

    index_pivot_point = len(all_values) - 1
    if len(negative_index) > 0:
        # находим индекс максимального отрицательного элемента
        min_negative_index = np.argmax(negative_diff)
        index_pivot_point = negative_index[min_negative_index]
        print(min_negative_index)
        print(index_pivot_point)

    pivot_point = all_values[index_pivot_point]

    plt.scatter(index_pivot_point, pivot_point, c="red", s=240)


def calculate_down_entry_point(trend):
    last_value = trend.get_all_values()[-1]
    iter_trend = trend.get_current_iter()
    all_values = trend.get_all_values()

    trend.search_up2down_trend_point(iter_trend, 2)
    trend.plot_up2down_trend_point(iter_trend)

    index = trend.get_up2down_trend_indexes(iter_trend)
    value = trend.get_up2down_trend_values(iter_trend)
    diff = value - last_value
    positive_diff = diff[diff > 0]
    positive_index = index[diff > 0]
    positive_value = value[diff > 0]
    print(list(zip(value, index)))
    print(list(zip(diff, index)))
    print(list(zip(positive_value, positive_index)))
    # plt.scatter(positive_index, positive_value, c="green", s=50)

    index_pivot_point = len(all_values) - 1
    if len(positive_index) > 0:
        # находим индекс максимального отрицательного элемента
        min_positive_index = np.argmin(positive_diff)
        index_pivot_point = positive_index[min_positive_index]
        print(min_positive_index)
        print(index_pivot_point)

    pivot_point = all_values[index_pivot_point]

    plt.scatter(index_pivot_point, pivot_point, c="green", s=240)


if __name__ == "__main__":
    N_ROW = 15_000
    P = 10_000

    path = os.path.abspath("../data/AUD_CAD.csv")
    file = os.path.basename(path).split(".")[0]

    eps = [30, 1, 1]
    repeat = [4, 3, 3]

    df = pd.read_csv(
        path,
        nrows=N_ROW
    )
    coincident_num = ext_coin.Coincident()
    main_combined()
