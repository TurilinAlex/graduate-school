import pandas as pd
from matplotlib import pyplot as plt

from core_math.trend_detection import CombinedExtremum, MergeExtremum

P = 0.9
N_ROW = 10_000


def main_merge():
    df = pd.read_csv('../data/AUD_USD.csv', nrows=N_ROW)
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
    df = pd.read_csv('../data/AUD_USD.csv', nrows=N_ROW)
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
    df = pd.read_csv('../data/AUD_USD.csv', nrows=N_ROW)
    close = df.Close.values

    for i in range(1, 3):
        trend_combined = CombinedExtremum(close, P)
        trend_combined.main_extremum(num_coincident=1, start_eps=i)

        print(len(trend_combined.get_combined_values(1)))


if __name__ == '__main__':
    main_correlation()
