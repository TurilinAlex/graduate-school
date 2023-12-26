from operator import le, lt, ge, gt
from typing import Callable

import numpy as np

from core.sort import argsort

np.random.seed(12222)


class CombinedExtremum:

    def __init__(self, values: np.ndarray[float], split: int, batch: int = None):
        self._values = values[:split]
        self._batch = len(values) if batch is None else batch
        self._len_values = len(values)
        self._ptr_extr: {int, tuple[int, int]} = {}

        for i in range(0, self._len_values, self._batch):
            self._ptr_extr[i // self._batch] = (i, min(i + self._batch, self._len_values))

    def localization_extremes(self, coincident: int = 1, eps: int = 1) -> "CombinedExtremum":

        extr_index = []

        offset = 0
        for sub_interval, (begin, end) in self._ptr_extr.items():
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
            _extr_index = self._filter_extremes(
                _offset=begin,
                _eps_min=select_min_eps,
                _eps_max=select_max_eps,
            )
            extr_index.extend(_extr_index)

            self._ptr_extr[sub_interval] = (offset, offset + len(_extr_index))
            offset += len(_extr_index)

        print(np.array(extr_index))
        print(self._ptr_extr)
        for _, (begin, end) in self._ptr_extr.items():
            print(extr_index[begin: end])

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
        _n = len(self._diff_index_for_minimal)
        _extremes = np.empty_like(self._diff_index_for_minimal)
        k = 0

        for i in range(_n):
            is_min_extr = self._diff_index_for_minimal[i] > _eps_min or self._diff_index_for_minimal[i] == _n
            is_max_extr = self._diff_index_for_maximal[i] > _eps_max or self._diff_index_for_maximal[i] == _n

            is_added = False
            if is_min_extr:
                is_add_index = self._border_check(
                    left=le,
                    right=lt,
                    _extremum_index=i,
                    _offset=_offset,
                    _eps=_eps_min,
                )

                if is_add_index:
                    _extremes[k] = i + _offset
                    k += 1
                    is_added = True

            if is_max_extr and not is_added:
                is_add_index = self._border_check(
                    left=gt,
                    right=ge,
                    _extremum_index=i,
                    _offset=_offset,
                    _eps=_eps_max,
                )

                if is_add_index:
                    _extremes[k] = i + _offset
                    k += 1

        _extremes = _extremes[:k]

        return _extremes

    def _filter_min_extremes(self, _index: np.ndarray[int], _offset: int, _eps: int) -> np.ndarray:
        _n = len(self._diff_index_for_minimal)
        if _eps >= self._diff_index_for_minimal[0]:
            is_add_index = self._border_check(
                left=le,
                right=lt,
                _extremum_index=_index[0],
                _offset=_offset,
                _eps=_eps,
            )

            if is_add_index:
                _extremes = np.array([_index[0] + _offset])
                return _extremes

        k = 0
        _extremes = np.empty_like(_index)
        for i in range(_n):
            if self._diff_index_for_minimal[i] > _eps:
                is_add_index = self._border_check(
                    left=le,
                    right=lt,
                    _extremum_index=_index[i],
                    _offset=_offset,
                    _eps=_eps,
                )

                if is_add_index:
                    _extremes[k] = _index[i] + _offset
                    k += 1
        _extremes = _extremes[:k]

        return _extremes

    def _filter_max_extremes(self, _index: np.ndarray[int], _offset: int, _eps: int) -> np.ndarray:
        _n = len(self._diff_index_for_maximal)
        if _eps >= self._diff_index_for_maximal[_n - 1]:
            is_add_index = self._border_check(
                left=gt,
                right=ge,
                _extremum_index=_index[_n - 1],
                _offset=_offset,
                _eps=_eps,
            )

            if is_add_index:
                _extremes = np.array([_index[_n - 1] + _offset])
                return _extremes

        k = 0
        _extremes = np.empty_like(_index)
        for i in range(_n):
            if self._diff_index_for_maximal[i] > _eps:
                is_add_index = self._border_check(
                    left=gt,
                    right=ge,
                    _extremum_index=_index[i],
                    _offset=_offset,
                    _eps=_eps,
                )

                if is_add_index:
                    _extremes[k] = _index[i] + _offset
                    k += 1
        _extremes = _extremes[:k]

        return _extremes

    def _border_check(
            self,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            _extremum_index: int, _offset: int, _eps: int
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

        if _extremum_index + _eps >= self._batch:
            for j in range(_offset + self._batch, _extremum_index + _offset + _eps + 1):
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
