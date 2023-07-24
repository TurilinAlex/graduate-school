from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from core_math.extremum import coincident, max_extremum, min_extremum
from core_math.merge_arg_sort import merge_arg_sort

N_ROW = 15_000
P = 0.9
N = int(N_ROW * P)


class Trend:
    class ExtrName(Enum):
        min_indexes = 'min_indexes_iter_{num_iter}'
        max_indexes = 'max_indexes_iter_{num_iter}'
        max_values = 'max_values_iter_{num_iter}'
        min_values = 'min_values_iter_{num_iter}'
        max_eps = 'max_eps_iter_{num_iter}'
        min_eps = 'min_eps_iter_{num_iter}'
        combined_indexes = 'combined_indexes_iter_{num_iter}'
        combined_values = 'combined_values_iter_{num_iter}'

    class TrendName(Enum):
        min_indexes = 'min_indexes_iter_{num_iter}'
        max_indexes = 'max_indexes_iter_{num_iter}'
        max_values = 'max_values_iter_{num_iter}'
        min_values = 'min_values_iter_{num_iter}'
        max_eps = 'max_eps_iter_{num_iter}'
        min_eps = 'min_eps_iter_{num_iter}'
        combined_indexes = 'combined_indexes_iter_{num_iter}'
        combined_values = 'combined_values_iter_{num_iter}'

    def __init__(self, values: list[float] | np.ndarray[float]):
        self._values = values
        self._current_iter: int = 0

        self._last_values = None
        self._last_indexes = None

    # region Extremum Parameters
    def get_min_indexes(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.min_indexes, num_iter)
        return result

    def get_max_indexes(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.max_indexes, num_iter)
        return result

    def get_min_values(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.min_values, num_iter)
        return result

    def get_max_values(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.max_values, num_iter)
        return result

    def get_min_eps(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.min_eps, num_iter)
        return result

    def get_max_eps(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.max_eps, num_iter)
        return result

    def get_combined_indexes(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.combined_indexes, num_iter)
        return result

    def get_combined_values(self, num_iter: int) -> Iterable:
        result = self._getter(self.ExtrName.combined_values, num_iter)
        return result

    # endregion Extremum Parameters

    # region Trend Parameters

    def get_trend_min_indexes(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.min_indexes, num_iter)
        return result

    def get_trend_max_indexes(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.max_indexes, num_iter)
        return result

    def get_trend_min_values(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.min_values, num_iter)
        return result

    def get_trend_max_values(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.max_values, num_iter)
        return result

    def get_trend_min_eps(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.min_eps, num_iter)
        return result

    def get_trend_max_eps(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.max_eps, num_iter)
        return result

    def get_trend_combined_indexes(self, num_iter: int) -> Iterable:
        result = self._getter(self.TrendName.combined_indexes, num_iter)
        return result

    def get_trend_combined_values(self, num_iter: int) -> Iterable:
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

    def trend_entry_point(self, num_iter: int, eps: int):
        values = np.array(self.get_combined_values(num_iter))
        indexes = np.array(self.get_combined_indexes(num_iter))
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

            setattr(self, self.TrendName.min_indexes.value.format(num_iter=num_iter), _min_indexes)
            setattr(self, self.TrendName.max_indexes.value.format(num_iter=num_iter), _max_indexes)

            setattr(self, self.TrendName.min_values.value.format(num_iter=num_iter), _min_values)
            setattr(self, self.TrendName.max_values.value.format(num_iter=num_iter), _max_values)

            setattr(self, self.TrendName.min_eps.value.format(num_iter=num_iter), eps)
            setattr(self, self.TrendName.max_eps.value.format(num_iter=num_iter), eps)

            setattr(self, self.TrendName.combined_values.value.format(num_iter=num_iter), _combined_values)
            setattr(self, self.TrendName.combined_indexes.value.format(num_iter=num_iter), _combined_indexes)

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


if __name__ == '__main__':
    df = pd.read_csv('../data/AUD_USD.csv', nrows=N_ROW)
    close = df.Close.values[:N]
    trend = Trend(close)

    # first iteration
    index_first = merge_arg_sort(close)
    _max_temp_index_first, max_eps_first = coincident(3)(max_extremum)(index=index_first, eps=50)
    _min_temp_index_first, min_eps_first = coincident(3)(min_extremum)(index=index_first, eps=50)
    max_values_first = close[_max_temp_index_first]
    min_values_first = close[_min_temp_index_first]
    index_temp_first = np.array(sorted(_max_temp_index_first + _min_temp_index_first))
    close_first = close[index_temp_first]
    index_merge_first = index_temp_first
    max_index_first = np.array(_max_temp_index_first)
    min_index_first = np.array(_min_temp_index_first)

    trend.trend_at_extremum(num_coincident=3, start_eps=50)
    assert all(max_values_first == trend.get_max_values(1)), print(max_values_first, trend.get_max_values(1))
    assert all(min_values_first == trend.get_min_values(1)), print(min_values_first, trend.get_min_values(1))

    assert all(max_index_first == trend.get_max_indexes(1)), print(max_index_first, trend.get_max_indexes(1))
    assert all(min_index_first == trend.get_min_indexes(1)), print(min_index_first, trend.get_min_indexes(1))

    assert all(close_first == trend.get_combined_values(1)), print(close_first, trend.get_combined_values(1))
    assert all(index_merge_first == trend.get_combined_indexes(1)), print(index_merge_first,
                                                                          trend.get_combined_indexes(1))

    assert min_eps_first == trend.get_min_eps(1), print(min_eps_first, trend.get_min_eps(1))
    assert max_eps_first == trend.get_max_eps(1), print(max_eps_first, trend.get_max_eps(1))

    # second iteration
    index_second = merge_arg_sort(close_first)
    _max_temp_index_second, max_eps_second = coincident(2)(max_extremum)(index=index_second, eps=1)
    _min_temp_index_second, min_eps_second = coincident(2)(min_extremum)(index=index_second, eps=1)
    max_values_second = close_first[_max_temp_index_second]
    min_values_second = close_first[_min_temp_index_second]
    index_temp_second = np.array(sorted(_max_temp_index_second + _min_temp_index_second))
    close_second = close_first[index_temp_second]
    index_merge_second = index_temp_first[index_temp_second]
    max_index_second = index_temp_first[_max_temp_index_second]
    min_index_second = index_temp_first[_min_temp_index_second]

    trend.trend_at_extremum(num_coincident=2, start_eps=1)
    assert all(max_values_second == trend.get_max_values(2)), print(max_values_second, trend.get_max_values(2))
    assert all(min_values_second == trend.get_min_values(2)), print(min_values_second, trend.get_min_values(2))

    assert all(max_index_second == trend.get_max_indexes(2)), print(max_index_second, trend.get_max_indexes(2))
    assert all(min_index_second == trend.get_min_indexes(2)), print(min_index_second, trend.get_min_indexes(2))

    assert all(close_second == trend.get_combined_values(2)), print(close_second, trend.get_combined_values(2))
    assert all(index_merge_second == trend.get_combined_indexes(2)), print(index_merge_second,
                                                                           trend.get_combined_indexes(2))

    assert min_eps_second == trend.get_min_eps(2), print(min_eps_second, trend.get_min_eps(2))
    assert max_eps_second == trend.get_max_eps(2), print(max_eps_second, trend.get_max_eps(2))

    # third iteration
    index_third = merge_arg_sort(close_second)
    _max_temp_index_third, max_eps_third = coincident(2)(max_extremum)(index=index_third, eps=1)
    _min_temp_index_third, min_eps_third = coincident(2)(min_extremum)(index=index_third, eps=1)
    max_values_third = close_second[_max_temp_index_third]
    min_values_third = close_second[_min_temp_index_third]
    index_temp_third = np.array(sorted(_max_temp_index_third + _min_temp_index_third))
    close_third = close_second[index_temp_third]
    index_merge_third = index_merge_second[index_temp_third]
    max_index_third = index_merge_second[_max_temp_index_third]
    min_index_third = index_merge_second[_min_temp_index_third]

    trend.trend_at_extremum(num_coincident=2, start_eps=1)
    assert all(max_values_third == trend.get_max_values(3)), print(max_values_third, trend.get_max_values(3))
    assert all(min_values_third == trend.get_min_values(3)), print(min_values_third, trend.get_min_values(3))

    assert all(max_index_third == trend.get_max_indexes(3)), print(max_index_third, trend.get_max_indexes(3))
    assert all(min_index_third == trend.get_min_indexes(3)), print(min_index_third, trend.get_min_indexes(3))

    assert all(close_third == trend.get_combined_values(3)), print(close_third, trend.get_combined_values(3))
    assert all(index_merge_third == trend.get_combined_indexes(3)), print(index_merge_third,
                                                                          trend.get_combined_indexes(3))

    assert min_eps_third == trend.get_min_eps(3), print(min_eps_third, trend.get_min_eps(3))
    assert max_eps_third == trend.get_max_eps(3), print(max_eps_third, trend.get_max_eps(3))

    # fourth iteration
    index_fourth = merge_arg_sort(close_third)
    _max_temp_index_fourth, max_eps_fourth = coincident(1)(max_extremum)(index=index_fourth, eps=2)
    _min_temp_index_fourth, min_eps_fourth = coincident(1)(min_extremum)(index=index_fourth, eps=2)
    max_values_fourth = close_third[_max_temp_index_fourth]
    min_values_fourth = close_third[_min_temp_index_fourth]
    index_temp_fourth = np.array(sorted(_max_temp_index_fourth + _min_temp_index_fourth))
    close_fourth = close_third[index_temp_fourth]
    index_merge_fourth = index_merge_third[index_temp_fourth]
    max_index_fourth = index_merge_third[_max_temp_index_fourth]
    min_index_fourth = index_merge_third[_min_temp_index_fourth]

    trend.trend_entry_point(num_iter=3, eps=2)
    assert all(max_values_fourth == trend.get_trend_max_values(3)), print(max_values_fourth,
                                                                          trend.get_trend_max_values(3))
    assert all(min_values_fourth == trend.get_trend_min_values(3)), print(min_values_fourth,
                                                                          trend.get_trend_min_values(3))

    assert all(max_index_fourth == trend.get_trend_max_indexes(3)), print(max_index_fourth,
                                                                          trend.get_trend_max_indexes(3))
    assert all(min_index_fourth == trend.get_trend_min_indexes(3)), print(min_index_fourth,
                                                                          trend.get_trend_min_indexes(3))

    assert all(close_fourth == trend.get_trend_combined_values(3)), print(close_fourth,
                                                                          trend.get_trend_combined_values(3))
    assert all(index_merge_fourth == trend.get_trend_combined_indexes(3)), print(index_merge_fourth,
                                                                                 trend.get_trend_combined_indexes(3))

    assert min_eps_fourth == trend.get_trend_min_eps(3), print(min_eps_fourth, trend.get_trend_min_eps(3))
    assert max_eps_fourth == trend.get_trend_max_eps(3), print(max_eps_fourth, trend.get_trend_max_eps(3))

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
