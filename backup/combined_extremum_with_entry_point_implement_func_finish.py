from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from backup.extremum import coincident, max_extremum, min_extremum
from backup.merge_arg_sort import merge_arg_sort

P = 0.9
N_ROW = 15_000


class Trend:
    class ExtrName(Enum):
        min_indexes = 'min_indexes_iter_{num_iter}_extr'
        max_indexes = 'max_indexes_iter_{num_iter}_extr'
        max_values = 'max_values_iter_{num_iter}_extr'
        min_values = 'min_values_iter_{num_iter}_extr'
        max_eps = 'max_eps_iter_{num_iter}_extr'
        min_eps = 'min_eps_iter_{num_iter}_extr'
        combined_indexes = 'combined_indexes_iter_{num_iter}_extr'
        combined_values = 'combined_values_iter_{num_iter}_extr'

    class TrendName(Enum):
        min_indexes = 'min_indexes_iter_{num_iter}_trend'
        max_indexes = 'max_indexes_iter_{num_iter}_trend'
        max_values = 'max_values_iter_{num_iter}_trend'
        min_values = 'min_values_iter_{num_iter}_trend'
        max_eps = 'max_eps_iter_{num_iter}_trend'
        min_eps = 'min_eps_iter_{num_iter}_trend'
        combined_indexes = 'combined_indexes_iter_{num_iter}_trend'
        combined_values = 'combined_values_iter_{num_iter}_trend'

    def __init__(self, values: list[float] | np.ndarray[float], test_size: float):
        self._all_values = values
        self._values = values[:int(len(values) * test_size)]
        self._current_iter: int = 0

        self._last_values = None
        self._last_indexes = None

    # region Extremum Parameters
    def get_min_indexes(self, num_iter: int):
        result = self._getter(self.ExtrName.min_indexes, num_iter)
        return result

    def get_max_indexes(self, num_iter: int):
        result = self._getter(self.ExtrName.max_indexes, num_iter)
        return result

    def get_min_values(self, num_iter: int):
        result = self._getter(self.ExtrName.min_values, num_iter)
        return result

    def get_max_values(self, num_iter: int):
        result = self._getter(self.ExtrName.max_values, num_iter)
        return result

    def get_min_eps(self, num_iter: int):
        result = self._getter(self.ExtrName.min_eps, num_iter)
        return result

    def get_max_eps(self, num_iter: int):
        result = self._getter(self.ExtrName.max_eps, num_iter)
        return result

    def get_combined_indexes(self, num_iter: int):
        result = self._getter(self.ExtrName.combined_indexes, num_iter)
        return result

    def get_combined_values(self, num_iter: int):
        result = self._getter(self.ExtrName.combined_values, num_iter)
        return result

    # endregion Extremum Parameters

    # region Trend Parameters

    def get_trend_min_indexes(self, num_iter: int):
        result = self._getter(self.TrendName.min_indexes, num_iter)
        return result

    def get_trend_max_indexes(self, num_iter: int):
        result = self._getter(self.TrendName.max_indexes, num_iter)
        return result

    def get_trend_min_values(self, num_iter: int):
        result = self._getter(self.TrendName.min_values, num_iter)
        return result

    def get_trend_max_values(self, num_iter: int):
        result = self._getter(self.TrendName.max_values, num_iter)
        return result

    def get_trend_min_eps(self, num_iter: int):
        result = self._getter(self.TrendName.min_eps, num_iter)
        return result

    def get_trend_max_eps(self, num_iter: int):
        result = self._getter(self.TrendName.max_eps, num_iter)
        return result

    def get_trend_combined_indexes(self, num_iter: int):
        result = self._getter(self.TrendName.combined_indexes, num_iter)
        return result

    def get_trend_combined_values(self, num_iter: int):
        result = self._getter(self.TrendName.combined_values, num_iter)
        return result

    # endregion Trend Parameters

    # region Calculation Function
    def get_num_iteration(self) -> int:
        return self._current_iter

    def trend_at_extremum(self, num_coincident: int, start_eps: int):
        self._current_iter += 1
        if self._last_values is None and self._last_indexes is None:
            self._first_iteration(num_coincident, start_eps)
        else:
            self._further_iterations(num_coincident, start_eps)

        return self

    def trend_entry_point(self, after_iter: int, eps: int):
        values = np.array(self.get_combined_values(after_iter))
        indexes = np.array(self.get_combined_indexes(after_iter))
        if values is not None and indexes is not None:
            _index = merge_arg_sort(values)
            __max_index = max_extremum(index=_index, eps=eps)
            __min_index = min_extremum(index=_index, eps=eps)
            _max_values = values[__max_index]
            _min_values = values[__min_index]

            _index_combined = np.array(sorted(__max_index + __min_index))

            _min_indexes = indexes[__min_index]
            _max_indexes = indexes[__max_index]
            _combined_values = values[_index_combined]
            _combined_indexes = indexes[_index_combined]

            setattr(self, self.TrendName.min_indexes.value.format(num_iter=after_iter), _min_indexes)
            setattr(self, self.TrendName.max_indexes.value.format(num_iter=after_iter), _max_indexes)

            setattr(self, self.TrendName.min_values.value.format(num_iter=after_iter), _min_values)
            setattr(self, self.TrendName.max_values.value.format(num_iter=after_iter), _max_values)

            setattr(self, self.TrendName.min_eps.value.format(num_iter=after_iter), eps)
            setattr(self, self.TrendName.max_eps.value.format(num_iter=after_iter), eps)

            setattr(self, self.TrendName.combined_values.value.format(num_iter=after_iter), _combined_values)
            setattr(self, self.TrendName.combined_indexes.value.format(num_iter=after_iter), _combined_indexes)

    def _first_iteration(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._values)
        __max_index, _max_eps = coincident(num_coincident)(max_extremum)(index=_index, eps=start_eps)
        __min_index, _min_eps = coincident(num_coincident)(min_extremum)(index=_index, eps=start_eps)
        _max_values = self._values[__max_index]
        _min_values = self._values[__min_index]

        _index_combined = np.array(sorted(__max_index + __min_index))
        _values_combined = self._values[_index_combined]

        setattr(self, self.ExtrName.min_indexes.value.format(num_iter=self._current_iter), __min_index)
        setattr(self, self.ExtrName.max_indexes.value.format(num_iter=self._current_iter), __max_index)

        setattr(self, self.ExtrName.min_values.value.format(num_iter=self._current_iter), _min_values)
        setattr(self, self.ExtrName.max_values.value.format(num_iter=self._current_iter), _max_values)

        setattr(self, self.ExtrName.min_eps.value.format(num_iter=self._current_iter), _min_eps)
        setattr(self, self.ExtrName.max_eps.value.format(num_iter=self._current_iter), _max_eps)

        setattr(self, self.ExtrName.combined_indexes.value.format(num_iter=self._current_iter), _index_combined)
        setattr(self, self.ExtrName.combined_values.value.format(num_iter=self._current_iter), _values_combined)

        self._last_values = _values_combined
        self._last_indexes = _index_combined

    def _further_iterations(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._last_values)
        __max_index, _max_eps = coincident(num_coincident)(max_extremum)(index=_index, eps=start_eps)
        __min_index, _min_eps = coincident(num_coincident)(min_extremum)(index=_index, eps=start_eps)
        _max_values = self._last_values[__max_index]
        _min_values = self._last_values[__min_index]

        _index_combined = np.array(sorted(__max_index + __min_index))

        _min_indexes = self._last_indexes[__min_index]
        _max_indexes = self._last_indexes[__max_index]
        _combined_values = self._last_values[_index_combined]
        _combined_indexes = self._last_indexes[_index_combined]

        setattr(self, self.ExtrName.min_indexes.value.format(num_iter=self._current_iter), _min_indexes)
        setattr(self, self.ExtrName.max_indexes.value.format(num_iter=self._current_iter), _max_indexes)

        setattr(self, self.ExtrName.min_values.value.format(num_iter=self._current_iter), _min_values)
        setattr(self, self.ExtrName.max_values.value.format(num_iter=self._current_iter), _max_values)

        setattr(self, self.ExtrName.min_eps.value.format(num_iter=self._current_iter), _min_eps)
        setattr(self, self.ExtrName.max_eps.value.format(num_iter=self._current_iter), _max_eps)

        setattr(self, self.ExtrName.combined_values.value.format(num_iter=self._current_iter), _combined_values)
        setattr(self, self.ExtrName.combined_indexes.value.format(num_iter=self._current_iter), _combined_indexes)

        self._last_indexes = self._last_indexes[_index_combined]
        self._last_values = _combined_values

    def _getter(self, name: ExtrName | TrendName, num_iter: int):
        result = None
        if num_iter > self._current_iter:
            return result
        try:
            result = getattr(self, name.value.format(num_iter=num_iter))
        except AttributeError as error:
            print(error)

        return result

    # endregion Calculation Function

    # region Plot

    def plot_values(self):
        plt.plot(self._all_values, color='black')
        plt.plot(self._values)

    def plot_extremum(self, num_iter: int):
        plt.scatter(
            self.get_max_indexes(num_iter),
            self.get_max_values(num_iter),
            color='r',
            s=10 * num_iter * (1.1 * (num_iter - 1) + 1),
            label=f'max_eps:{self.get_max_eps(num_iter)} len:{len(self.get_max_indexes(num_iter))}'
        )
        plt.scatter(
            self.get_min_indexes(num_iter),
            self.get_min_values(num_iter),
            color='g',
            s=10 * num_iter * (1.1 * (num_iter - 1) + 1),
            label=f'max_eps:{self.get_min_eps(num_iter)} len:{len(self.get_min_indexes(num_iter))}'
        )

    def plot_trend(self, after_iter: int):
        plt.scatter(
            self.get_trend_max_indexes(after_iter),
            self.get_trend_max_values(after_iter),
            color='y',
            s=20 * after_iter,
            label=f'max_eps:{self.get_trend_max_eps(after_iter)} len:{len(self.get_trend_max_indexes(after_iter))}'
        )
        plt.scatter(
            self.get_trend_min_indexes(after_iter),
            self.get_trend_min_values(after_iter),
            color='m',
            s=20 * after_iter,
            label=f'max_eps:{self.get_trend_min_eps(after_iter)} len:{len(self.get_trend_min_indexes(after_iter))}'
        )


if __name__ == '__main__':
    df = pd.read_csv('../data/AUD_USD.csv', nrows=N_ROW)
    close = df.Close.values
    trend = Trend(close, P)
    trend.plot_values()

    trend.trend_at_extremum(num_coincident=3, start_eps=50)
    trend.plot_extremum(1)

    trend.trend_at_extremum(num_coincident=2, start_eps=1)
    trend.plot_extremum(2)

    trend.trend_at_extremum(num_coincident=2, start_eps=1)
    trend.plot_extremum(3)

    trend.trend_at_extremum(num_coincident=2, start_eps=1)
    trend.plot_extremum(4)

    plt.grid()
    plt.legend()
    plt.show()
