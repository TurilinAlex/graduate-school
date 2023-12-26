from operator import le, lt, ge, gt
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from core.sort import argsort

np.random.seed(12222)


class Slice(BaseModel):
    begin: int
    end: int


class Extremum(BaseModel):
    index: np.ndarray = Field(default=np.empty((1,)))
    slice: Slice = Field(default=Slice(begin=0, end=0))

    class Config:
        arbitrary_types_allowed = True


class ExtremumData(BaseModel):
    extr_all: Extremum
    extr_min: Extremum
    extr_max: Extremum
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

        for i in range(0, self._len_values, self._batch):
            end = min(i + self._batch, self._len_values)
            self._ptr_extr[i // self._batch] = ExtremumData(
                extr_all=Extremum(
                    index=np.arange(i, end),
                    slice=Slice(
                        begin=i,
                        end=end,
                    ),
                ),
                extr_min=Extremum(
                    index=np.arange(i, end),
                    slice=Slice(
                        begin=i,
                        end=end,
                    ),
                ),
                extr_max=Extremum(
                    index=np.arange(i, end),
                    slice=Slice(
                        begin=i,
                        end=end,
                    ),
                ),
            )

    def localization_extremes(self, coincident: int = 1, eps: int = 1) -> "CombinedExtremum":

        extr_all_index = []
        print("---------------------")
        offset_all = 0
        offset_min = 0
        offset_max = 0

        for sub_interval, data in self._ptr_extr.items():
            begin = data.extr_all.slice.begin
            end = data.extr_all.slice.end

            _values = self._values[begin: end]
            _indexes = argsort(_values)

            self._diff_between_sort_indexes(_indexes=_indexes, _eps=eps)

            select_min_eps = self._select_eps(
                _marker_diff=self._marker_diff_for_minimal,
                _coincident=coincident,
                _eps=eps,
            )
            select_max_eps = self._select_eps(
                _marker_diff=self._marker_diff_for_maximal,
                _coincident=coincident,
                _eps=eps,
            )
            _extr_all_index, _extr_min_index, _extr_max_index = self._filter_extremes(
                _offset=begin,
                _eps_min=select_min_eps,
                _eps_max=select_max_eps,
            )
            extr_all_index.extend(_extr_all_index + begin)

            _extr_all_index = data.extr_all.index[_extr_all_index]
            _extr_min_index = data.extr_all.index[_extr_min_index]
            _extr_max_index = data.extr_all.index[_extr_max_index]

            _data = ExtremumData(
                extr_all=Extremum(
                    index=_extr_all_index,
                    slice=Slice(
                        begin=offset_all,
                        end=offset_all + len(_extr_all_index)),
                ),
                extr_min=Extremum(
                    index=_extr_min_index,
                    slice=Slice(
                        begin=offset_min,
                        end=offset_min + len(_extr_min_index)),
                ),
                extr_max=Extremum(
                    index=_extr_max_index,
                    slice=Slice(
                        begin=offset_max,
                        end=offset_max + len(_extr_max_index)),
                ),
                eps_min=select_min_eps,
                eps_max=select_max_eps,
            )

            self._ptr_extr[sub_interval] = _data
            offset_all += len(_extr_all_index)
            offset_min += len(_extr_min_index)
            offset_max += len(_extr_max_index)

        extr_all_index = np.array(extr_all_index)

        self._values = self._values[extr_all_index]
        for key, value in self._ptr_extr.items():
            print(key, value.extr_all.index)
        print()

        return self

    def _diff_between_sort_indexes(self, _indexes: np.ndarray, _eps: int = 1) -> None:
        _n = len(_indexes)
        # Хранит разницу между парами индексов ниже главной диагонали (локальные минимумы)
        self._diff_index_for_minimal = np.empty_like(_indexes, dtype=np.uint32)

        # Хранит разницу между парами индексов выше главной диагонали (локальные максимумы)
        self._diff_index_for_maximal = np.empty_like(_indexes, dtype=np.uint32)

        # Хранит метки о значении разности индексов для локальных минимумов.
        # То есть, abs(_index[i] - _index[i - j]) = _diff -> _marker_diff_for_minimal[_diff] = 1
        self._marker_diff_for_minimal = np.zeros((_n + 1,), dtype=np.int32)

        # Хранит метки о значении разности индексов для локальных минимумов.
        # То есть, abs(_index[i] - _index[i + j]) = _diff -> _marker_diff_for_maximal[_diff] = 1
        self._marker_diff_for_maximal = np.zeros((_n + 1,), dtype=np.int32)

        # region Вычисление разницы между индексами

        for i in range(_n):
            min_diff_for_minimal, min_diff_for_maximal = _n, _n

            # min
            for j in range(1, i + 1):
                diff = abs(_indexes[i] - _indexes[i - j])
                if diff < min_diff_for_minimal:
                    min_diff_for_minimal = diff
                if min_diff_for_minimal <= _eps:
                    break

            self._diff_index_for_minimal[_indexes[i]] = min_diff_for_minimal
            self._marker_diff_for_minimal[min_diff_for_minimal] = 1

            # max
            for j in range(1, (_n - i)):
                diff = abs(_indexes[i] - _indexes[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= _eps:
                    break

            self._diff_index_for_maximal[_indexes[i]] = min_diff_for_maximal
            self._marker_diff_for_maximal[min_diff_for_maximal] = 1

        # endregion Вычисление разницы между индексами

    def _filter_extremes(self, _offset: int, _eps_min: int, _eps_max: int):
        _batch = len(self._diff_index_for_minimal)
        _extremes = np.empty_like(self._diff_index_for_minimal)
        _extremes_min = np.empty_like(self._diff_index_for_minimal)
        _extremes_max = np.empty_like(self._diff_index_for_minimal)
        _k_all = 0
        _k_min = 0
        _k_max = 0

        for i in range(_batch):
            is_min_extr = self._diff_index_for_minimal[i] > _eps_min or self._diff_index_for_minimal[i] == _batch
            is_max_extr = self._diff_index_for_maximal[i] > _eps_max or self._diff_index_for_maximal[i] == _batch

            is_added = False
            if is_min_extr:
                is_add_index = self._border_check(
                    left=le,
                    right=lt,
                    _extremum_index=i,
                    _offset=_offset,
                    _batch=_batch,
                    _eps=_eps_min,
                )

                if is_add_index:
                    _extremes[_k_all] = i
                    _extremes_min[_k_min] = i
                    _k_all += 1
                    _k_min += 1
                    is_added = True

            if is_max_extr and not is_added:
                is_add_index = self._border_check(
                    left=gt,
                    right=ge,
                    _extremum_index=i,
                    _offset=_offset,
                    _batch=_batch,
                    _eps=_eps_max,
                )

                if is_add_index:
                    _extremes[_k_all] = i
                    _extremes_max[_k_max] = i
                    _k_all += 1
                    _k_max += 1

        _extremes = _extremes[:_k_all]
        _extremes_min = _extremes_min[:_k_min]
        _extremes_max = _extremes_max[:_k_max]

        return _extremes, _extremes_min, _extremes_max

    def _border_check(
            self,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            _extremum_index: int, _offset: int, _batch: int, _eps: int
    ) -> bool:
        _left = True
        _right = True

        # region Gluing Minimals Sub-intervals

        if _extremum_index - _eps < 0:
            for j in range(_offset - 1, _extremum_index + _offset - _eps - 1, -1):
                if j < 0:
                    _left = True
                    break

                if left(self._values[j], self._values[_extremum_index + _offset]):
                    _left = False
                    break

        if _extremum_index + _eps >= _batch:
            for j in range(_offset + _batch, _extremum_index + _offset + _eps + 1):
                if j >= len(self._values):
                    _right = True
                    break

                if right(self._values[j], self._values[_extremum_index + _offset]):
                    _right = False
                    break

        return _left and _right

        # endregion Gluing Minimals Sub-intervals

    def _max_border_check(self, _extremum_index: int, _offset: int, _eps: int) -> bool:
        _left = True
        _right = True

        # region Gluing Maximals Sub-intervals

        if _extremum_index - _eps < 0:
            for j in range(_offset - 1, _extremum_index + _offset - _eps - 1, -1):
                if j < 0:
                    _left = True
                    break

                if self._values[j] > self._values[_extremum_index + _offset]:
                    _left = False
                    break

        if _extremum_index + _eps >= self._batch:
            for j in range(_offset + self._batch, _extremum_index + _offset + _eps + 1):
                if j >= len(self._values):
                    _right = True
                    break

                if self._values[j] >= self._values[_extremum_index + _offset]:
                    _right = False
                    break

        return _left and _right

        # endregion Gluing Maximals Sub-intervals

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

    eps = 3
    step = 3
    coincident = 1

    trend = CombinedExtremum(values=closes, split=size, batch=step)
    trend.localization_extremes(coincident=coincident, eps=eps)


if __name__ == '__main__':
    main()
