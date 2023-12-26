from operator import le, lt, ge, gt
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from core import CombinedTrendDetection
from core.matches_extremum import MatchesOnInputArray
from core.sort import argsort

np.random.seed(12222)


class Slice(BaseModel):
    begin: int
    end: int


class ExtremumAll(BaseModel):
    index: np.ndarray = Field(default=np.array((1,)))
    slice: Slice = Field(default=Slice(begin=0, end=0))

    class Config:
        arbitrary_types_allowed = True


class ExtremumMin(ExtremumAll):
    diff: np.ndarray = Field(default=np.array((1,)))


class ExtremumMax(ExtremumAll):
    diff: np.ndarray = Field(default=np.array((1,)))


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
        self._ptr_extr: {int, ExtremumData} = {}

        _len_values = len(values)
        for begin in range(0, _len_values, self._batch):
            end = min(begin + self._batch, _len_values)
            self._ptr_extr[begin // self._batch] = ExtremumData(
                extr_all=ExtremumAll(
                    index=np.arange(begin, end),
                    slice=Slice(
                        begin=begin,
                        end=end,
                    ),
                ),
                extr_min=ExtremumMin(
                    index=np.arange(begin, end),
                    slice=Slice(
                        begin=begin,
                        end=end,
                    ),
                ),
                extr_max=ExtremumMax(
                    index=np.arange(begin, end),
                    slice=Slice(
                        begin=begin,
                        end=end,
                    ),
                ),
            )

    def localization_extremes(self, coincident: int = 1, eps: int = 1) -> "CombinedExtremum":

        offset_all = 0
        offset_min = 0
        offset_max = 0
        extr_all_index = []

        for sub_interval, data in self._ptr_extr.items():
            _offset = data.extr_all.slice.begin
            end = _offset + len(data.extr_all.index)

            _marker_min, _marker_max = self._diff_between_sort_indexes(
                _values=self._values[_offset: end],
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
            _extr_all_index, _extr_min_index, _extr_max_index = self._filter_extremes(
                _diff_min=data.extr_min.diff,
                _diff_max=data.extr_max.diff,
                _offset=_offset,
                _eps_min=_eps_min,
                _eps_max=_eps_max,
            )

            extr_all_index.extend(_extr_all_index + _offset)

            _extr_all_index = data.extr_all.index[_extr_all_index]
            _extr_min_index = data.extr_all.index[_extr_min_index]
            _extr_max_index = data.extr_all.index[_extr_max_index]

            data.extr_all.index = _extr_all_index
            data.extr_all.slice.begin = offset_all
            data.extr_all.slice.end = offset_all + len(_extr_all_index)

            data.extr_min.index = _extr_min_index
            data.extr_min.slice.begin = offset_min
            data.extr_min.slice.end = offset_min + len(_extr_min_index)

            data.extr_max.index = _extr_max_index
            data.extr_max.slice.begin = offset_max
            data.extr_max.slice.end = offset_max + len(_extr_max_index)

            data.eps_min = _eps_min
            data.eps_max = _eps_max

            offset_all = data.extr_all.slice.end
            offset_min = data.extr_min.slice.end
            offset_max = data.extr_max.slice.end

        extr_all_index = np.array(extr_all_index)
        self._values = self._values[extr_all_index]

        return self

    def get_all_extremum(self, interval: int | None = None):

        if interval is None:
            return np.array(
                [elem for _, data in self._ptr_extr.items() for elem in data.extr_all.index]
            )
        if isinstance(interval, int):
            if self._ptr_extr.get(interval) is not None:
                return np.array(
                    [elem for elem in self._ptr_extr[interval].extr_all.index]
                )
            else:
                print("Такого интервала нет!")
        else:
            print("Интервал не является числом!")

    def get_min_extremum(self, interval: int | None = None):

        if interval is None:
            return np.array(
                [elem for _, data in self._ptr_extr.items() for elem in data.extr_min.index]
            )
        if isinstance(interval, int):
            if self._ptr_extr.get(interval) is not None:
                return np.array(
                    [elem for elem in self._ptr_extr[interval].extr_min.index]
                )
            else:
                print("Такого интервала нет!")
        else:
            print("Интервал не является числом!")

    def get_max_extremum(self, interval: int | None = None):

        if interval is None:
            return np.array(
                [elem for _, data in self._ptr_extr.items() for elem in data.extr_max.index]
            )
        if isinstance(interval, int):
            if self._ptr_extr.get(interval) is not None:
                return np.array(
                    [elem for elem in self._ptr_extr[interval].extr_max.index]
                )
            else:
                print("Такого интервала нет!")
        else:
            print("Интервал не является числом!")

    def _diff_between_sort_indexes(
            self,
            _values: np.ndarray,
            _sub_interval: int,
            _eps: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:

        _indexes = argsort(_values)

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

    def _filter_extremes(
            self,
            _diff_min: np.ndarray,
            _diff_max: np.ndarray,
            _offset: int,
            _eps_min: int,
            _eps_max: int,
    ):
        _batch = len(_diff_min)
        _extremes = np.empty_like(_diff_min)
        _extremes_min = np.empty_like(_diff_min)
        _extremes_max = np.empty_like(_diff_max)
        _k_all = 0
        _k_min = 0
        _k_max = 0

        for i in range(_batch):
            is_min_extr = _diff_min[i] > _eps_min or _diff_min[i] == _batch
            is_max_extr = _diff_max[i] > _eps_max or _diff_max[i] == _batch

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
            _extremum_index: int,
            _offset: int,
            _batch: int,
            _eps: int,
    ) -> bool:
        _left = True
        _right = True

        # region Gluing Sub-intervals

        if _extremum_index - _eps < 0:
            for i in range(_offset - 1, _extremum_index + _offset - _eps - 1, -1):
                if i < 0:
                    break

                if left(self._values[i], self._values[_extremum_index + _offset]):
                    _left = False
                    break

        if _extremum_index + _eps >= _batch:
            for i in range(_offset + _batch, _extremum_index + _offset + _eps + 1):
                if i >= len(self._values):
                    break

                if right(self._values[i], self._values[_extremum_index + _offset]):
                    _right = False
                    break

        return _left and _right

        # endregion Gluing Sub-intervals

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
    size = 30000
    closes = np.array([np.random.randint(-10000, 10000) for _ in range(size)])

    eps = 113
    step = 10000
    coincident = 1

    matches = MatchesOnInputArray()
    trend_old = CombinedTrendDetection(closes, size, matches)
    trend_new = CombinedExtremum(values=closes, split=size, batch=step)

    for _ in range(2):
        trend_old.search_extremum(num_coincident=coincident, start_eps=eps)
        trend_new.localization_extremes(coincident=coincident, eps=eps)

        assert len(trend_old.get_max_indexes()) == len(trend_new.get_max_extremum()), (
            print(
                f"Combined: old {len(trend_old.get_max_indexes())}, "
                f"new {len(trend_new.get_max_extremum())}"
            )
        )
        assert np.all(trend_old.get_max_indexes() == trend_new.get_max_extremum()), (
            print(
                f"Combined: old {trend_old.get_max_indexes()}, "
                f"new {trend_new.get_max_extremum()}"
            )
        )

        assert len(trend_old.get_min_indexes()) == len(trend_new.get_min_extremum()), (
            print(
                f"Combined: old {len(trend_old.get_min_indexes())}, "
                f"new {len(trend_new.get_min_extremum())}"
            )
        )
        assert np.all(trend_old.get_min_indexes() == trend_new.get_min_extremum()), (
            print(
                f"Combined: old {trend_old.get_min_indexes()}, "
                f"new {trend_new.get_min_extremum()}"
            )
        )

        assert len(trend_old.get_combined_indexes()) == len(trend_new.get_all_extremum()), (
            print(
                f"Combined: old {len(trend_old.get_combined_indexes())}, "
                f"new {len(trend_new.get_all_extremum())}"
            )
        )
        assert np.all(trend_old.get_combined_indexes() == trend_new.get_all_extremum()), (
            print(
                f"Combined: old {trend_old.get_combined_indexes()}, "
                f"new {trend_new.get_all_extremum()}"
            )
        )


if __name__ == '__main__':
    main()
