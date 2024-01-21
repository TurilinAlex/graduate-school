from dataclasses import dataclass
from operator import le, lt, gt, ge
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from core import CombinedTrendDetection
from core.extremum import min_extremum, max_extremum
from core.matches_extremum import MatchesOnInputArray
from core.sort import argsort

np.random.seed(12222)


def unravel_index(index, array: "Extremes"):
    i = 0
    size = 0
    while size < len(array) - 1:
        index -= len(array[i].values)
        if index < 0:
            break
        i += 1
        size += 1
    j = index % len(array[i].values) if len(array[i].values) else 0
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


class Extremes:
    __extremes: list["Extremes"] = []
    __values_all: np.ndarray[float]
    __len_all_values: int = 0
    __current: int = 0
    __current_iter: int = 0
    __split_index: int = 0

    __offset_all: int = 0
    __offset_min: int = 0
    __offset_max: int = 0

    @dataclass
    class ExtremesAll:
        indexes: np.ndarray[int]
        begin: int
        end: int

    @dataclass
    class ExtremesMin(ExtremesAll):
        pass

    @dataclass
    class ExtremesMax(ExtremesAll):
        pass

    def __init__(
            self,
            values: np.ndarray[float],
            indexes: np.ndarray[int],
            begin: int,
            end: int,
            sub_interval: int
    ):
        self._is_update: bool = False
        self._sub_interval: int = sub_interval

        self._values: np.ndarray[float] = values

        self._extr_all: Extremes.ExtremesAll = self.ExtremesAll(indexes=indexes, begin=begin, end=end)
        self._extr_min: Extremes.ExtremesMin = self.ExtremesMin(indexes=indexes, begin=begin, end=end)
        self._extr_max: Extremes.ExtremesMax = self.ExtremesMax(indexes=indexes, begin=begin, end=end)

        self._diff_min: np.ndarray[int] | None = None
        self._diff_max: np.ndarray[int] | None = None

        self._trend_min: np.ndarray[int] | None = None
        self._trend_max: np.ndarray[int] | None = None

        self._eps_min: int | None = None
        self._eps_max: int | None = None

        self.__add(self=self)

    @property
    def values(self) -> np.ndarray[float]:
        return self._values

    @values.setter
    def values(self, values: np.ndarray[float]):
        self._values = values

    @property
    def indexes(self) -> np.ndarray[float]:
        return self._extr_all.indexes

    def localization_extremes(self, item: int | slice | None = None, coincident: int = 1, eps: int = 1) -> "Extremes":

        if item is None:
            Extremes.__offset_all = self[0]._extr_all.begin
            Extremes.__offset_min = self[0]._extr_min.begin
            Extremes.__offset_max = self[0]._extr_max.begin
            _data = self

        elif isinstance(item, int):
            Extremes.__offset_all = self[item]._extr_all.begin
            Extremes.__offset_min = self[item]._extr_min.begin
            Extremes.__offset_max = self[item]._extr_max.begin
            _data = self[item]

        elif isinstance(item, slice):
            Extremes.__offset_all = self[item.start]._extr_all.begin
            Extremes.__offset_min = self[item.start]._extr_min.begin
            Extremes.__offset_max = self[item.start]._extr_max.begin
            _data = self[item]

        else:
            raise IndexError

        self.__update_values()

        for data in _data:
            data.__localization_extremes(coincident, eps)

        return self

    def __localization_extremes(self, coincident: int, eps: int):
        self._is_update = True
        _offset = self._extr_all.begin

        _marker_min, _marker_max = self._diff_between_sort_indexes(eps=eps)

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
            _diff_min=self._diff_min,
            _diff_max=self._diff_max,
            _eps_min=_eps_min,
            _eps_max=_eps_max,
            _offset=_offset,
        )

        _extr_all_index = self._extr_all.indexes[_extr_all_index]
        _extr_min_index = self._extr_all.indexes[_extr_min_index]
        _extr_max_index = self._extr_all.indexes[_extr_max_index]

        self._extr_all.indexes = _extr_all_index
        self._extr_all.begin = Extremes.__offset_all
        self._extr_all.end = Extremes.__offset_all + len(_extr_all_index)

        self._extr_min.indexes = _extr_min_index
        self._extr_min.begin = Extremes.__offset_min
        self._extr_min.end = Extremes.__offset_min + len(_extr_min_index)

        self._extr_max.indexes = _extr_max_index
        self._extr_max.begin = Extremes.__offset_max
        self._extr_max.end = Extremes.__offset_max + len(_extr_max_index)

        self._eps_min = _eps_min
        self._eps_max = _eps_max

        Extremes.__offset_all = self._extr_all.end
        Extremes.__offset_min = self._extr_min.end
        Extremes.__offset_max = self._extr_max.end

    def min_trend_identification(self, eps: int):
        _indexes = argsort(self.get_min_values())
        _min_indexes = np.sort(min_extremum(index=_indexes, eps=eps))

        _min_trend_indexes = self.get_min_indexes()[_min_indexes]
        _min_trend_values = self.get_min_values()[_min_indexes]

        return _min_trend_indexes, _min_trend_values

    def max_trend_identification(self, eps: int):
        _indexes = argsort(self.get_max_values())
        _max_indexes = np.sort(max_extremum(index=_indexes, eps=eps))

        _max_trend_indexes = self.get_max_indexes()[_max_indexes]
        _max_trend_values = self.get_max_values()[_max_indexes]

        return _max_trend_indexes, _max_trend_values

    def _diff_between_sort_indexes(self, eps: int = 1) -> tuple[np.ndarray, np.ndarray]:

        _indexes = argsort(self._values)

        n = len(_indexes)
        self._diff_min = np.empty_like(_indexes, dtype=np.uint32)
        self._diff_max = np.empty_like(_indexes, dtype=np.uint32)

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
                if min_diff_for_minimal <= eps:
                    break

            self._diff_min[_indexes[i]] = min_diff_for_minimal
            marker_diff_for_minimal[min_diff_for_minimal] = 1

            # max
            for j in range(1, (n - i)):
                diff = abs(_indexes[i] - _indexes[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= eps:
                    break

            self._diff_max[_indexes[i]] = min_diff_for_maximal
            marker_diff_for_maximal[min_diff_for_maximal] = 1

        # endregion Вычисление разницы между индексами

        return marker_diff_for_minimal, marker_diff_for_maximal

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

        _i_extr, _j_extr = unravel_index((_extremes_index + _offset), self)

        # region Gluing Sub-intervals

        if _extremes_index - _eps < 0:
            for i in range(_offset - 1, _extremes_index + _offset - _eps - 1, -1):
                if i < 0:
                    break
                _i, _j = unravel_index(i, self)
                if left(
                        self[_i]._values[_j],
                        self[_i_extr]._values[_j_extr]
                ):
                    _left = False
                    break

        if _extremes_index + _eps >= _batch:
            for i in range(_offset + _batch, _extremes_index + _offset + _eps + 1):
                if i >= self.__len_all_values:
                    break

                _i, _j = unravel_index(i, self)
                if right(
                        self[_i]._values[_j],
                        self[_i_extr]._values[_j_extr]
                ):
                    _right = False
                    break

        return _left and _right

        # endregion Gluing Sub-intervals

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

    def get_split_index(self) -> int:
        return self.__split_index

    def get_all_values(self) -> np.ndarray[np.float32]:
        return self.__values_all

    def get_current_iter(self) -> int:
        return self.__current_iter

    def get_combined_indexes(self, interval: int | None = None):

        return self.__prepare_extr(attr="_extr_all", interval=interval)

    def get_combined_values(self, interval: int | None = None):

        _indexes = self.__prepare_extr(attr="_extr_all", interval=interval)

        if len(_indexes):
            return self.__values_all[_indexes]
        else:
            return np.array([], dtype=float)

    def get_min_indexes(self, interval: int | None = None):

        return self.__prepare_extr(attr="_extr_min", interval=interval)

    def get_min_values(self, interval: int | None = None):

        _indexes = self.__prepare_extr(attr="_extr_min", interval=interval)

        if len(_indexes):
            return self.__values_all[_indexes]
        else:
            return np.array([], dtype=float)

    def get_max_indexes(self, interval: int | None = None):

        return self.__prepare_extr(attr="_extr_max", interval=interval)

    def get_max_values(self, interval: int | None = None):

        _indexes = self.__prepare_extr(attr="_extr_max", interval=interval)

        if len(_indexes):
            return self.__values_all[_indexes]
        else:
            return np.array([], dtype=float)

    def __prepare_extr(self, attr: str, interval: int | None):

        if interval is None:
            return np.array(
                [elem for data in self for elem in getattr(data, attr).indexes]
            )
        if isinstance(interval, int):
            try:
                return np.array(
                    [elem for elem in getattr(self[interval], attr).indexes]
                )
            except IndexError as error:
                print(f"Такого интервала нет! {error}")
        else:
            print("Интервал не является числом!")

    def get_min_eps(self, interval: int | None = None):

        return self.__prepare_eps(attr="_eps_min", interval=interval)

    def get_max_eps(self, interval: int | None = None):

        return self.__prepare_eps(attr="_eps_max", interval=interval)

    def __prepare_eps(self, attr: str, interval: int | None):

        if interval is None:
            return np.array(
                [getattr(data, attr) for data in self]
            )
        if isinstance(interval, int):
            try:
                return np.array(
                    [getattr(self[interval], attr)]
                )
            except IndexError as error:
                print(f"Такого интервала нет! {error}")
        else:
            print("Интервал не является числом!")

    def __len__(self):
        return len(Extremes.__extremes)

    def __iter__(self):
        return iter(self.__extremes)

    def __next__(self):
        if self.__current < len(self.__extremes):
            __data = self.__extremes[self.__current]
            self.__current += 1
            return __data
        else:
            raise StopIteration

    def __getitem__(self, item) -> "Extremes":
        return Extremes.__extremes[item]

    def __repr__(self):
        return (
            f"{self._sub_interval}: (V: "
            f"{self._values}; I: "
            f"{self._extr_all.indexes}; ["
            f"{self._extr_all.begin}, "
            f"{self._extr_all.end}])"
        )

    @classmethod
    def build(cls, values: np.ndarray[float], split: int, batch: int):
        cls.__split_index = split
        cls.__values_all = values

        _extr = None
        for begin in range(0, split, batch):
            end = min(begin + batch, split)
            _extr = Extremes(
                values=values[begin: end],
                indexes=np.arange(begin, end, dtype=int),
                begin=begin,
                end=end,
                sub_interval=begin // batch
            )

        return _extr

    @classmethod
    def __add(cls, self: "Extremes"):
        cls.__extremes.append(self)
        cls.__len_all_values += len(self._values)

    @classmethod
    def __update_values(cls):
        cls.__len_all_values = 0
        for data in cls.__extremes:
            if data._is_update:
                data.values = data.__values_all[data.indexes]
            cls.__len_all_values += len(data.values)


def main():
    size = 13
    closes = np.array([np.random.randint(10, 50) for _ in range(size)])

    eps = 1
    step = 3
    coincident = 1

    matches = MatchesOnInputArray()
    trend_old = CombinedTrendDetection(closes, size, matches)
    trend_new: Extremes = Extremes.build(values=closes, split=size, batch=step)
    plt.plot(trend_new.get_all_values(), marker=".")

    for _ in range(1):
        trend_old.search_extremum(num_coincident=coincident, start_eps=eps)
        trend_old.search_change_trend_point(eps=2)
        trend_new.localization_extremes(coincident=coincident, eps=eps)

        min_indexes, min_values = trend_new.min_trend_identification(eps=2)
        max_indexes, max_values = trend_new.max_trend_identification(eps=2)
        print(trend_old.get_combined_indexes(), trend_new.get_combined_indexes())
        plt.scatter(trend_new.get_combined_indexes(), trend_new.get_combined_values(), s=50)
        plt.scatter(min_indexes, min_values, s=150, color="red")
        plt.scatter(max_indexes, max_values, s=150, color="yellow")

        assert len(trend_old.get_max_indexes()) == len(trend_new.get_max_indexes()), (
            print(
                f"Combined: old {len(trend_old.get_max_indexes())}, "
                f"new {len(trend_new.get_max_indexes())}"
            )
        )
        assert np.all(trend_old.get_max_indexes() == trend_new.get_max_indexes()), (
            print(
                f"Combined: old {trend_old.get_max_indexes()}, "
                f"new {trend_new.get_max_indexes()}"
            )
        )

        assert len(trend_old.get_min_indexes()) == len(trend_new.get_min_indexes()), (
            print(
                f"Combined: old {len(trend_old.get_min_indexes())}, "
                f"new {len(trend_new.get_min_indexes())}"
            )
        )
        assert np.all(trend_old.get_min_indexes() == trend_new.get_min_indexes()), (
            print(
                f"Combined: old {trend_old.get_min_indexes()}, "
                f"new {trend_new.get_min_indexes()}"
            )
        )

        assert len(trend_old.get_combined_indexes()) == len(trend_new.get_combined_indexes()), (
            print(
                f"Combined: old {len(trend_old.get_combined_indexes())}, "
                f"new {len(trend_new.get_combined_indexes())}"
            )
        )
        assert np.all(trend_old.get_combined_indexes() == trend_new.get_combined_indexes()), (
            print(
                f"Combined: old {trend_old.get_combined_indexes()}, "
                f"new {trend_new.get_combined_indexes()}"
            )
        )
    # plt.show()


if __name__ == '__main__':
    main()
