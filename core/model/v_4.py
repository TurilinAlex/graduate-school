from operator import le, lt, gt, ge
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from core.sort import argsort

np.random.seed(12222)


class Slice(BaseModel):
    begin: int
    end: int


class ExtremumAll(BaseModel):
    index: np.ndarray = Field(default=np.empty((1,)))
    slice: Slice = Field(default=Slice(begin=0, end=0))

    class Config:
        arbitrary_types_allowed = True


class ExtremumMin(ExtremumAll):
    diff: np.ndarray = Field(default=np.empty((1,)))

    class Config:
        arbitrary_types_allowed = True


class ExtremumMax(ExtremumAll):
    diff: np.ndarray = Field(default=np.empty((1,)))

    class Config:
        arbitrary_types_allowed = True


class ExtremumData(BaseModel):
    extr_all: ExtremumAll
    extr_min: ExtremumMin
    extr_max: ExtremumMax
    eps_min: int = Field(default=1)
    eps_max: int = Field(default=1)


class CombinedExtremum:

    def __init__(self, values: np.ndarray[float], split: int, batch: int = None):
        self._values = values[:split]
        self._batch = len(values) if batch is None else batch
        self._len_values = len(values)
        self._indexes = np.arange(0, self._len_values)
        self._ptr_extr: {int, ExtremumData} = {}
        print(self._values)
        self._num_intervals = 0

        for i in range(0, self._len_values, self._batch):
            self._num_intervals += 1
            end = min(i + self._batch, self._len_values)
            self._ptr_extr[i // self._batch] = ExtremumData(
                extr_all=ExtremumAll(
                    index=np.arange(i, end),
                    slice=Slice(
                        begin=i,
                        end=end,
                    ),
                ),
                extr_min=ExtremumMin(
                    index=np.arange(i, end),
                    slice=Slice(
                        begin=i,
                        end=end,
                    ),
                ),
                extr_max=ExtremumMax(
                    index=np.arange(i, end),
                    slice=Slice(
                        begin=i,
                        end=end,
                    ),
                ),
            )

    def localization_extremes(self, coincident: int = 1, eps: int = 1) -> "CombinedExtremum":

        for sub_interval, data in self._ptr_extr.items():
            print("-------------------")
            begin = data.extr_all.slice.begin
            end = data.extr_all.slice.end

            _values = self._values[begin: end]
            _indexes = argsort(_values)

            _marker_min, _marker_max = self._diff_between_sort_indexes(
                _indexes=_indexes,
                _sub_interval=sub_interval,
                _eps=eps,
            )

            _eps_min = self._select_eps(
                _marker_diff=_marker_min,
                _coincident=coincident,
                _eps=eps,
            )
            _eps_max = self._select_eps(
                _marker_diff=_marker_max,
                _coincident=coincident,
                _eps=eps,
            )

            self._filter_extremes(
                _sub_interval=sub_interval,
                _eps_min=_eps_min,
                _eps_max=_eps_max,
            )
            print("-------------------")
            print()

        for _, data in self._ptr_extr.items():
            print(data.extr_all.index, end=" ")
        print()
        for _, data in self._ptr_extr.items():
            print(data.extr_min.index, end=" ")
        print()
        for _, data in self._ptr_extr.items():
            print(data.extr_max.index, end=" ")
        print()

        return self

    def _diff_between_sort_indexes(
            self,
            _indexes: np.ndarray,
            _sub_interval: int,
            _eps: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(_indexes)
        self._ptr_extr[_sub_interval].extr_min.diff = np.empty_like(_indexes, dtype=np.uint32)
        self._ptr_extr[_sub_interval].extr_max.diff = np.empty_like(_indexes, dtype=np.uint32)

        # Хранит метки о значении разности индексов для локальных минимумов.
        # То есть, abs(_index[i] - _index[i - j]) = _diff -> marker_diff_for_minimal[_diff] = 1
        marker_diff_for_minimal = np.zeros((n + 1,), dtype=np.int32)

        # Хранит метки о значении разности индексов для локальных минимумов.
        # То есть, abs(_index[i] - _index[i + j]) = _diff -> marker_diff_for_maximal[_diff] = 1
        marker_diff_for_maximal = np.zeros((n + 1,), dtype=np.int32)

        # region Вычисление разницы между индексами

        for i in range(n):
            min_diff_for_minimal, min_diff_for_maximal = n, n

            # min
            for j in range(1, i + 1):
                diff = abs(_indexes[i] - _indexes[i - j])
                if diff < min_diff_for_minimal:
                    min_diff_for_minimal = diff
                if min_diff_for_minimal <= _eps:
                    break

            self._ptr_extr[_sub_interval].extr_min.diff[_indexes[i]] = min_diff_for_minimal
            marker_diff_for_minimal[min_diff_for_minimal] = 1

            # max
            for j in range(1, (n - i)):
                diff = abs(_indexes[i] - _indexes[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= _eps:
                    break

            self._ptr_extr[_sub_interval].extr_max.diff[_indexes[i]] = min_diff_for_maximal
            marker_diff_for_maximal[min_diff_for_maximal] = 1

        # endregion Вычисление разницы между индексами

        return marker_diff_for_minimal, marker_diff_for_maximal

    def _filter_extremes(self, _sub_interval: int, _eps_min: int, _eps_max: int):
        _batch = len(self._ptr_extr[_sub_interval].extr_all.index)
        _extremes = np.zeros((len(self._ptr_extr[_sub_interval].extr_all.index),), dtype=int)
        _extremes_min = np.zeros((len(self._ptr_extr[_sub_interval].extr_all.index),), dtype=int)
        _extremes_max = np.zeros((len(self._ptr_extr[_sub_interval].extr_all.index),), dtype=int)

        _k_all = 0
        _k_min = 0
        _k_max = 0

        for i in range(_batch):
            is_min_extr = (self._ptr_extr[_sub_interval].extr_min.diff[i] > _eps_min or
                           self._ptr_extr[_sub_interval].extr_min.diff[i] == _batch)
            is_max_extr = (self._ptr_extr[_sub_interval].extr_max.diff[i] > _eps_max or
                           self._ptr_extr[_sub_interval].extr_max.diff[i] == _batch)

            is_added = False
            if is_min_extr:
                print("Min")
                is_add_index = self._border_check(
                    left=le,
                    right=lt,
                    _extremum_index=i,
                    _sub_interval=_sub_interval,
                    _eps=_eps_min,
                )
                print("Min ----", is_add_index)

                if is_add_index:
                    _extremes[_k_all] = i
                    _extremes_min[_k_min] = i
                    _k_all += 1
                    _k_min += 1
                    is_added = True

            if is_max_extr and not is_added:
                print("Max")
                is_add_index = self._border_check(
                    left=gt,
                    right=ge,
                    _extremum_index=i,
                    _sub_interval=_sub_interval,
                    _eps=_eps_max,
                )
                print("Max ----", is_add_index)

                if is_add_index:
                    _extremes[_k_all] = i
                    _extremes_max[_k_max] = i
                    _k_all += 1
                    _k_max += 1

        self._ptr_extr[_sub_interval].extr_all.index = (
            self._ptr_extr[_sub_interval].extr_all.index[_extremes[:_k_all]]
        )
        self._ptr_extr[_sub_interval].extr_min.index = (
            self._ptr_extr[_sub_interval].extr_min.index[_extremes_min[:_k_min]]
        )
        self._ptr_extr[_sub_interval].extr_max.index = (
            self._ptr_extr[_sub_interval].extr_max.index[_extremes_max[:_k_max]]
        )

    def _border_check(
            self,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            _extremum_index: int, _sub_interval: int, _eps: int
    ) -> bool:
        print("_________", _sub_interval)
        _left = True
        _right = True
        _offset = self._ptr_extr[_sub_interval].extr_all.slice.begin

        # region Gluing Minimals Sub-intervals

        _left_count = self._ptr_extr[_sub_interval].extr_all.slice.begin
        _left_sub_interval = _sub_interval
        if _extremum_index - _eps < 0:
            while _left_sub_interval - 1 >= 0:
                if not len(self._ptr_extr[_left_sub_interval - 1].extr_all.index):
                    _left_count -= 1

                if _left_count < _extremum_index - _eps - 1:
                    break

                for index in self._ptr_extr[_left_sub_interval - 1].extr_all.index[::-1]:
                    _left_count -= 1
                    print(
                        "left",
                        self._values[index],
                        self._values[self._ptr_extr[_sub_interval].extr_all.index[_extremum_index]],
                        self._ptr_extr[_sub_interval].extr_all.index[_extremum_index],
                    )
                    if left(
                            self._values[index],
                            self._values[self._ptr_extr[_sub_interval].extr_all.index[_extremum_index]]
                    ):
                        _left = False
                        break
                    if _left_count < _extremum_index - _eps - 1:
                        break

                else:
                    _left_sub_interval -= 1
                    continue

                break

        _right_count = self._ptr_extr[_sub_interval].extr_all.slice.end
        _right_sub_interval = _sub_interval
        if _extremum_index + _eps >= len(self._ptr_extr[_right_sub_interval].extr_all.index):
            while _right_sub_interval + 1 < self._num_intervals:
                if not len(self._ptr_extr[_right_sub_interval + 1].extr_all.index):
                    _right_count += 1

                if _right_count > _extremum_index + _eps + 1:
                    break

                for index in self._ptr_extr[_right_sub_interval + 1].extr_all.index:
                    print(
                        "right",
                        self._values[index],
                        self._values[self._ptr_extr[_sub_interval].extr_all.index[_extremum_index]],
                        self._ptr_extr[_sub_interval].extr_all.index[_extremum_index],
                        index,
                        _right_sub_interval + 1
                    )
                    _right_count += 1
                    if right(
                            self._values[index],
                            self._values[self._ptr_extr[_sub_interval].extr_all.index[_extremum_index]]
                    ):
                        _right = False
                        break
                    if _right_count > _extremum_index + _eps + 1:
                        break

                else:
                    _right_sub_interval += 1
                    continue

                break

        return _left and _right

    @staticmethod
    def _select_eps(_marker_diff: np.ndarray, _coincident: int, _eps: int) -> int:

        # region Подбор эпсилон окрестности, зависящей от заданного количества совпадений

        count_zero = 0
        last_non_zero_index = _eps
        for i in range(_eps + 1, len(_marker_diff)):
            if count_zero >= _coincident - 1:
                _select_eps = last_non_zero_index + _coincident - 1
                return _select_eps

            if _marker_diff[i] == 0:
                count_zero += 1
            else:
                count_zero = 0
                last_non_zero_index = i

        _select_eps = last_non_zero_index + _coincident - 1
        return _select_eps

        # endregion Подбор эпсилон окрестности, зависящей от заданного количества совпадений


def main():
    size = 13
    closes = np.array([np.random.randint(10, 50) for _ in range(size)])

    eps = 7
    step = 1
    coincident = 1

    trend = CombinedExtremum(values=closes, split=size, batch=step)
    trend.localization_extremes(coincident=coincident, eps=eps)


if __name__ == '__main__':
    main()
