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


class ExtremesAll(BaseModel):
    index: np.ndarray[int] = Field(default=np.array((1,)))
    value: np.ndarray[float] = Field(default=np.array((1,)))
    slice: Slice = Field(default=Slice(begin=0, end=0))

    class Config:
        arbitrary_types_allowed = True


class ExtremesMin(BaseModel):
    diff: np.ndarray = Field(default=np.array((1,)))
    index: np.ndarray[int] = Field(default=np.array((1,)))
    slice: Slice = Field(default=Slice(begin=0, end=0))

    class Config:
        arbitrary_types_allowed = True


class ExtremesMax(BaseModel):
    diff: np.ndarray = Field(default=np.array((1,)))
    index: np.ndarray[int] = Field(default=np.array((1,)))
    slice: Slice = Field(default=Slice(begin=0, end=0))

    class Config:
        arbitrary_types_allowed = True


class ExtremesData(BaseModel):
    extr_all: ExtremesAll
    extr_min: ExtremesMin
    extr_max: ExtremesMax
    eps_min: int = Field(default=1)
    eps_max: int = Field(default=1)


def unravel_index(index, array: list[ExtremesData]):
    i = 0
    size = 0
    while size < len(array) - 1:
        index -= len(array[i].extr_all.value)
        if index < 0:
            break
        i += 1
        size += 1
    j = index % len(array[i].extr_all.value) if len(array[i].extr_all.value) else 0
    return i, j


def select_eps(_marker_diff: np.ndarray, _coincident: int, _eps: int) -> int:

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


class CombinedExtremes:

    def __init__(self, values: np.ndarray[float], split: int, batch: int = None):
        self._values = values[:split]
        self._batch = len(self._values) if batch is None else batch
        self._ptr_extr: list[ExtremesData] = []

        self._len = len(self._values)
        for begin in range(0, self._len, self._batch):
            end = min(begin + self._batch, self._len)
            self._ptr_extr.append(
                ExtremesData(
                    extr_all=ExtremesAll(
                        index=np.arange(begin, end),
                        slice=Slice(
                            begin=begin,
                            end=end,
                        ),
                    ),
                    extr_min=ExtremesMin(
                        index=np.arange(begin, end),
                        slice=Slice(
                            begin=begin,
                            end=end,
                        ),
                    ),
                    extr_max=ExtremesMax(
                        index=np.arange(begin, end),
                        slice=Slice(
                            begin=begin,
                            end=end,
                        ),
                    ),
                )
            )

    def localization_extremes(self, coincident: int = 1, eps: int = 1) -> "CombinedExtremes":

        offset_all = self._ptr_extr[0].extr_all.slice.begin
        offset_min = self._ptr_extr[0].extr_min.slice.begin
        offset_max = self._ptr_extr[0].extr_max.slice.begin

        self._update_values()

        for sub_interval, data in enumerate(self._ptr_extr):
            _offset = data.extr_all.slice.begin

            _marker_min, _marker_max = self._diff_between_sort_indexes(
                _sub_interval=sub_interval,
                _eps=eps,
            )

            _eps_min = select_eps(
                _marker_diff=_marker_min,
                _coincident=coincident,
                _eps=eps,
            )
            _eps_max = select_eps(
                _marker_diff=_marker_max,
                _coincident=coincident,
                _eps=eps,
            )
            _extr_all_index, _extr_min_index, _extr_max_index = self._filter_extremes(
                _diff_min=data.extr_min.diff,
                _diff_max=data.extr_max.diff,
                _eps_min=_eps_min,
                _eps_max=_eps_max,
                _offset=_offset,
            )

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

        return self

    def _update_values(self):
        self._len = 0
        for data in self._ptr_extr:
            data.extr_all.value = self._values[data.extr_all.index]
            self._len += len(data.extr_all.value)

    def _diff_between_sort_indexes(
            self,
            _sub_interval: int,
            _eps: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:

        _indexes = argsort(self._ptr_extr[_sub_interval].extr_all.value)

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
            if is_min_extr:
                is_add_index = self._border_check(
                    left=le,
                    right=lt,
                    _extremes_index=i,
                    _offset=_offset,
                    _batch=_batch,
                    _eps=_eps_min,
                )

                if is_add_index:
                    _extremes[_k_all] = i
                    _extremes_min[_k_min] = i
                    _k_all += 1
                    _k_min += 1
                    continue

            is_max_extr = _diff_max[i] > _eps_max or _diff_max[i] == _batch
            if is_max_extr:
                is_add_index = self._border_check(
                    left=gt,
                    right=ge,
                    _extremes_index=i,
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
            _extremes_index: int,
            _offset: int,
            _batch: int,
            _eps: int,
    ) -> bool:
        _left = True
        _right = True

        _i_extr, _j_extr = unravel_index((_extremes_index + _offset), self._ptr_extr)

        # region Gluing Sub-intervals

        if _extremes_index - _eps < 0:
            for i in range(_offset - 1, _extremes_index + _offset - _eps - 1, -1):
                if i < 0:
                    break
                _i, _j = unravel_index(i, self._ptr_extr)
                if left(
                        self._ptr_extr[_i].extr_all.value[_j],
                        self._ptr_extr[_i_extr].extr_all.value[_j_extr]
                ):
                    _left = False
                    break

        if _extremes_index + _eps >= _batch:
            for i in range(_offset + _batch, _extremes_index + _offset + _eps + 1):
                if i >= self._len:
                    break

                _i, _j = unravel_index(i, self._ptr_extr)
                if right(
                        self._ptr_extr[_i].extr_all.value[_j],
                        self._ptr_extr[_i_extr].extr_all.value[_j_extr]
                ):
                    _right = False
                    break

        return _left and _right

        # endregion Gluing Sub-intervals

    def get_all_extremes(self, interval: int | None = None):

        return self.__prepare_extr(attr="extr_all", interval=interval)

    def get_min_extremes(self, interval: int | None = None):

        return self.__prepare_extr(attr="extr_min", interval=interval)

    def get_max_extremes(self, interval: int | None = None):

        return self.__prepare_extr(attr="extr_max", interval=interval)

    def __prepare_extr(self, attr: str, interval: int | None):

        if interval is None:
            return np.array(
                [elem for data in self._ptr_extr for elem in getattr(data, attr).index]
            )
        if isinstance(interval, int):
            try:
                return np.array(
                    [elem for elem in getattr(self._ptr_extr[interval], attr).index]
                )
            except IndexError as error:
                print(f"Такого интервала нет! {error}")
        else:
            print("Интервал не является числом!")


def main():
    size = 13
    closes = np.array([np.random.randint(10, 50) for _ in range(size)])

    eps = 3
    step = 3
    coincident = 1

    matches = MatchesOnInputArray()
    trend_old = CombinedTrendDetection(closes, size, matches)
    trend_new = CombinedExtremes(values=closes, split=size, batch=step)

    for _ in range(4):
        trend_old.search_extremum(num_coincident=coincident, start_eps=eps)
        trend_new.localization_extremes(coincident=coincident, eps=eps)

        assert len(trend_old.get_max_indexes()) == len(trend_new.get_max_extremes()), (
            print(
                f"Combined: old {len(trend_old.get_max_indexes())}, "
                f"new {len(trend_new.get_max_extremes())}"
            )
        )
        assert np.all(trend_old.get_max_indexes() == trend_new.get_max_extremes()), (
            print(
                f"Combined: old {trend_old.get_max_indexes()}, "
                f"new {trend_new.get_max_extremes()}"
            )
        )

        assert len(trend_old.get_min_indexes()) == len(trend_new.get_min_extremes()), (
            print(
                f"Combined: old {len(trend_old.get_min_indexes())}, "
                f"new {len(trend_new.get_min_extremes())}"
            )
        )
        assert np.all(trend_old.get_min_indexes() == trend_new.get_min_extremes()), (
            print(
                f"Combined: old {trend_old.get_min_indexes()}, "
                f"new {trend_new.get_min_extremes()}"
            )
        )

        assert len(trend_old.get_combined_indexes()) == len(trend_new.get_all_extremes()), (
            print(
                f"Combined: old {len(trend_old.get_combined_indexes())}, "
                f"new {len(trend_new.get_all_extremes())}"
            )
        )
        assert np.all(trend_old.get_combined_indexes() == trend_new.get_all_extremes()), (
            print(
                f"Combined: old {trend_old.get_combined_indexes()}, "
                f"new {trend_new.get_all_extremes()}"
            )
        )


if __name__ == '__main__':
    main()

    # def unravel_index(index, array):
    #     i = 0
    #     size = 0
    #     while size < len(array) - 1:
    #         index -= len(array[i])
    #         if index < 0:
    #             break
    #         i += 1
    #         size += 1
    #     j = index % len(array[i]) if len(array[i]) else 0
    #     return i, j
    #
    # a = [
    #     [0, 1],
    #     [2],
    #     [],
    #     [3],
    #     [4],
    # ]
    #
    # b = [i for elem in a for i in elem]
    #
    # for index in b:
    #     i, j = unravel_index(index, a)
    #     print(index, i, j, a[i][j])

    # print(unravel_index(3, a))
