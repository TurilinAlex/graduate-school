import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import core.matches_extremum as matches
from core.trend.combined_trend_point import CombinedTrendDetection
from core.trend.merge_trend_point import MergeTrendDetection
from core.trend.split_trend_point import SplitTrendDetection
from core.visualization.plot_trend_point import PlotTrendPoint


def main_merge(data: np.ndarray[np.float32]):
    trend_merge = MergeTrendDetection(data, P, coincident_num)
    viz = PlotTrendPoint(trend_merge)
    viz.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_merge.search_extremum(num_coincident=r, start_eps=e)
        trend_merge.search_change_trend_point(eps=2)
        viz.plot_extremum()
        viz.plot_combined_extremum()

    viz.plot_change_trend()

    return trend_merge, viz


def main_combined(data: np.ndarray[np.float32]):
    trend_combined = CombinedTrendDetection(data, P, coincident_num)
    viz = PlotTrendPoint(trend_combined)
    viz.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_combined.search_extremum(num_coincident=r, start_eps=e)
        trend_combined.search_change_trend_point(eps=2)
        viz.plot_extremum()
        viz.plot_combined_extremum()

    viz.plot_change_trend()

    return trend_combined, viz


def main_split(data: np.ndarray[np.float32]):
    trend_split = SplitTrendDetection(data, P, coincident_num)
    viz = PlotTrendPoint(trend_split)
    viz.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_split.search_extremum(num_coincident=r, start_eps=e)
        trend_split.search_change_trend_point(eps=2)
        viz.plot_extremum()
        viz.plot_combined_extremum()

    viz.plot_change_trend()

    return trend_split, viz


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

    path = os.path.abspath("../../data/AUD_CAD.csv")
    file = os.path.basename(path).split(".")[0]

    eps = [30, 1, 1]
    repeat = [4, 2, 2]

    df = pd.read_csv(
        path,
        nrows=N_ROW
    )
    close = df.Close.values
    dates = df.Date.values

    coincident_num = matches.MatchesOnInputArray()
    detection, visualisation = main_split(data=close)
    visualisation.show(
        title=file,
        from_date=dates[0],
        to_date=dates[-1],
        split_date=dates[P],
        timeframe="1 minute",
    )

    # N_ROW = 15_000
    # P = 10_000
    #
    # path = os.path.abspath("../data/USD_CHF.csv")
    # file = os.path.basename(path).split(".")[0]
    #
    # eps = [30, 1, 1]
    # repeat = [3, 2, 2]
    #
    # dfs = list(pd.read_csv(
    #     path,
    #     chunksize=P + 1
    # ))
    #
    # for ii in range(len(list(dfs))):
    #
    #     print(ii, "===================")
    #     df = dfs[ii]
    #     close = df.Close.values
    #     dates = df.Date.values
    #
    #     if len(close) < P:
    #         continue
    #
    #     coincident_num = ext_coin.Coincident()
    #     detection, visualisation = main_split(data=close)
    #
    #     up2down_trend_indexes = detection.get_up2down_trend_indexes()
    #     max_indexes = detection.get_max_indexes()
    #
    #     down2up_trend_indexes = detection.get_down2up_trend_indexes()
    #     min_indexes = detection.get_min_indexes()
    #
    #     if up2down_trend_indexes[-1] < down2up_trend_indexes[-1]:
    #         print("Detect DOWN last trend")
    #     else:
    #
    #         index_min = np.where(min_indexes == down2up_trend_indexes[-1])[0][0]
    #         index_max = np.where(max_indexes == up2down_trend_indexes[-1])[0][0]
    #
    #         if index_max - 2 >= 0 and index_min + 2 <= len(min_indexes) - 1:
    #
    #             a_1 = min_indexes[index_min:index_min + 2 + 1]
    #             a_2 = max_indexes[index_max - 2:index_max + 1]
    #
    #             a_3, _, _ = merge_extremum(
    #                 extr_min_index=a_1,
    #                 extr_max_index=a_2,
    #                 values=close
    #             )
    #
    #             if len(a_3) == len(a_1) + len(a_2):
    #                 print("Detect UP last trend", down2up_trend_indexes, up2down_trend_indexes)
    #
    #                 print(max_indexes)
    #                 print(up2down_trend_indexes)
    #                 print(min_indexes)
    #                 print(down2up_trend_indexes)
    #
    #                 print(
    #                     index_min,
    #                     index_max,
    #                 )
    #
    #                 print(
    #                     min_indexes[index_min:index_min + 2 + 1],
    #                     max_indexes[index_max - 2:index_max + 1]
    #                 )
    #
    #                 print(a_3)
    #                 visualisation._ax_plot.scatter(a_1, close[a_1], s=30, c="red")
    #                 visualisation._ax_plot.scatter(a_2, close[a_2], s=30, c="green")
    #                 visualisation.save(
    #                     title=file,
    #                     from_date=dates[0],
    #                     to_date=dates[-1],
    #                     split_date=dates[P],
    #                     timeframe="1 minute",
    #                     name=f"{ii}.jpg"
    #                 )
    #         else:
    #             print("Don`t detect reversal trend")
    #     print()
