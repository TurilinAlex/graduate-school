import copy
from dataclasses import dataclass
from operator import le, lt, gt, ge
from typing import Type, Callable

import numpy as np

from core.extremum import min_extremum, max_extremum
from core.sort import argsort


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


class ExtremesContainer:

    def __init__(
            self,
            values: np.ndarray[float],
            indexes: np.ndarray[int],
            begin: int,
            end: int,
            sub_interval: int,
    ):
        self.is_update: bool = False
        self.sub_interval: int = sub_interval

        self._values: np.ndarray[float] = values

        self.extr_all: ExtremesAll = ExtremesAll(indexes=indexes, begin=begin, end=end)
        self.extr_min: ExtremesMin = ExtremesMin(indexes=indexes, begin=begin, end=end)
        self.extr_max: ExtremesMax = ExtremesMax(indexes=indexes, begin=begin, end=end)

        self._diff_min: np.ndarray[int] | None = None
        self._diff_max: np.ndarray[int] | None = None

        self._trend_min: np.ndarray[int] | None = None
        self._trend_max: np.ndarray[int] | None = None

        self._eps_min: int | None = None
        self._eps_max: int | None = None

    @property
    def values(self) -> np.ndarray[float]:
        return self._values

    @values.setter
    def values(self, values: np.ndarray[float]):
        self._values = values

    @property
    def indexes(self) -> np.ndarray[float]:
        return self.extr_all.indexes

    def search_extremes(self, coincident: int, eps: int):
        self.is_update = True
        _offset = self.extr_all.begin

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
        _extr_all_index, _extr_min_index, _extr_max_index = ExtremesStorage.filter_extremes(
            _diff_min=self._diff_min,
            _diff_max=self._diff_max,
            _eps_min=_eps_min,
            _eps_max=_eps_max,
            _offset=_offset,
        )

        _extr_all_index = self.extr_all.indexes[_extr_all_index]
        _extr_min_index = self.extr_all.indexes[_extr_min_index]
        _extr_max_index = self.extr_all.indexes[_extr_max_index]

        self.extr_all.indexes = _extr_all_index
        self.extr_all.begin = ExtremesStorage.offset_all
        self.extr_all.end = ExtremesStorage.offset_all + len(_extr_all_index)

        self.extr_min.indexes = _extr_min_index
        self.extr_min.begin = ExtremesStorage.offset_min
        self.extr_min.end = ExtremesStorage.offset_min + len(_extr_min_index)

        self.extr_max.indexes = _extr_max_index
        self.extr_max.begin = ExtremesStorage.offset_max
        self.extr_max.end = ExtremesStorage.offset_max + len(_extr_max_index)

        self._eps_min = _eps_min
        self._eps_max = _eps_max

        ExtremesStorage.offset_all = self.extr_all.end
        ExtremesStorage.offset_min = self.extr_min.end
        ExtremesStorage.offset_max = self.extr_max.end

    def _diff_between_sort_indexes(self, eps: int = 1) -> tuple[np.ndarray, np.ndarray]:

        _indexes = argsort(self._values)

        n = len(_indexes)
        self._diff_min = np.empty_like(_indexes, dtype=np.uint32)
        self._diff_max = np.empty_like(_indexes, dtype=np.uint32)

        # Stores labels about the value of the index difference for local minima
        # That is abs(_index[i] - _index[i - j]) = _diff -> marker_diff_for_minimal[_diff] = 1
        marker_diff_for_minimal = np.zeros((n + 1,), dtype=np.int32)

        # Stores labels about the value of the index difference for local maxima
        # That is abs(_index[i] - _index[i + j]) = _diff -> marker_diff_for_maximal[_diff] = 1
        marker_diff_for_maximal = np.zeros((n + 1,), dtype=np.int32)

        # region Calculating the difference between indexes

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

        # endregion Calculating the difference between indexes

        return marker_diff_for_minimal, marker_diff_for_maximal

    def __repr__(self):
        return (
            f"{self.sub_interval}: (V: "
            f"{self._values}; I: "
            f"{self.extr_all.indexes}; ["
            f"{self.extr_all.begin}, "
            f"{self.extr_all.end}])"
        )


class MetaExtremes(type):
    _storage: list[ExtremesContainer] = []
    _current_iter: int = 0

    @classmethod
    def __len__(cls):
        return len(cls._storage)

    @classmethod
    def __iter__(cls):
        return iter(cls._storage)

    @classmethod
    def __next__(cls):
        if cls._current_iter < len(cls._storage):
            __data = cls._storage[cls._current_iter]
            cls._current_iter += 1
            return __data
        else:
            raise StopIteration

    @classmethod
    def __getitem__(
            cls, item: int | slice
    ) -> ExtremesContainer | list[ExtremesContainer]:
        return cls._storage[item]


class ExtremesStorage(metaclass=MetaExtremes):
    offset_all: int = 0
    offset_min: int = 0
    offset_max: int = 0

    split_index: int = 0
    values_all: np.ndarray[float]
    len_all_values: int = 0

    @classmethod
    def add(cls, container: ExtremesContainer):
        cls._storage.append(container)
        cls.len_all_values += len(container.values)

    @classmethod
    def search_extremes(cls, item: int | slice | None = None, coincident: int = 1, eps: int = 1):

        cls.__update_values()
        cls.__set_start_offset(item=item)

        if item is None:
            for container in ExtremesStorage:
                container.search_extremes(coincident=coincident, eps=eps)

            return

        if isinstance(item, int):
            ExtremesStorage[item].search_extremes(coincident=coincident, eps=eps)

            return

        if isinstance(item, slice):
            for container in ExtremesStorage[item]:
                container.search_extremes(coincident=coincident, eps=eps)

            return

    @classmethod
    def filter_extremes(
            cls,
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
                is_add_index = cls.border_check(
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
                is_add_index = cls.border_check(
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

    @classmethod
    def border_check(
            cls,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            _extremes_index: int,
            _offset: int,
            _batch: int,
            _eps: int,
    ) -> bool:
        _left = True
        _right = True

        _i_extr, _j_extr = cls.unravel_index((_extremes_index + _offset))

        # region Gluing Sub-intervals

        if _extremes_index - _eps < 0:
            for i in range(_offset - 1, _extremes_index + _offset - _eps - 1, -1):
                if i < 0:
                    break
                _i, _j = cls.unravel_index(i)
                if left(
                        ExtremesStorage[_i].values[_j],
                        ExtremesStorage[_i_extr].values[_j_extr]
                ):
                    _left = False
                    break

        if _extremes_index + _eps >= _batch:
            for i in range(_offset + _batch, _extremes_index + _offset + _eps + 1):
                if i >= ExtremesStorage.len_all_values:
                    break

                _i, _j = cls.unravel_index(i)
                if right(
                        ExtremesStorage[_i].values[_j],
                        ExtremesStorage[_i_extr].values[_j_extr]
                ):
                    _right = False
                    break

        return _left and _right

        # endregion Gluing Sub-intervals

    @classmethod
    def unravel_index(cls, index):
        i = 0
        size = 0
        while size < len(cls) - 1:
            index -= len(cls[i].values)
            if index < 0:
                break
            i += 1
            size += 1
        j = index % len(cls[i].values) if len(cls[i].values) else 0
        return i, j

    @classmethod
    def __set_start_offset(cls, item: int | slice | None = None) -> None:
        if item is None:
            cls.offset_all = cls[0].extr_all.begin
            cls.offset_min = cls[0].extr_min.begin
            cls.offset_max = cls[0].extr_max.begin
            return

        elif isinstance(item, int):
            cls.offset_all = cls[item].extr_all.begin
            cls.offset_min = cls[item].extr_min.begin
            cls.offset_max = cls[item].extr_max.begin
            return

        elif isinstance(item, slice):
            cls.offset_all = cls[item.start].extr_all.begin
            cls.offset_min = cls[item.start].extr_min.begin
            cls.offset_max = cls[item.start].extr_max.begin
            return

    @classmethod
    def __update_values(cls) -> None:
        cls.len_all_values = 0
        for data in cls._storage:
            if data.is_update:
                data.values = cls.values_all[data.indexes]
            cls.len_all_values += len(data.values)


def build(storage: Type[ExtremesStorage], values: np.ndarray[float], split: int, batch: int):
    storage.split_index = split
    storage.values_all = values

    for begin in range(0, split, batch):
        end = min(begin + batch, split)
        container = ExtremesContainer(
            values=values[begin: end],
            indexes=np.arange(begin, end, dtype=int),
            begin=begin,
            end=end,
            sub_interval=begin // batch
        )

        storage.add(container=container)


def select_eps(_marker_diff: np.ndarray, _coincident: int, _eps: int) -> int:
    # region Selection of an epsilon neighborhood depending on a given number of matches

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

    # endregion Selection of an epsilon neighborhood depending on a given number of matches


class ExtremesSaveState:

    def __init__(self):
        self.__states: {int, Type[ExtremesStorage]} = {}
        self.__states_iter: int = 0

    def save(self, storage: Type[ExtremesStorage]):
        self.__states[self.__states_iter] = copy.deepcopy(storage)
        self.__states_iter += 1

    def search_trend_points(
            self, eps: int, after_iter: int | None = None, item: int | slice | None = None
    ) -> tuple[np.ndarray[np.uint32], np.ndarray[np.uint32]]:

        min_trend_points = self.search_min_trend_points(eps=eps, after_iter=after_iter, item=item)
        max_trend_points = self.search_max_trend_points(eps=eps, after_iter=after_iter, item=item)

        return min_trend_points, max_trend_points

    def search_min_trend_points(
            self, eps: int, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self.__states_iter - 1

        _min_extremes_indexes = self.get_min_indexes(after_iter=after_iter, item=item)
        _temp_min_extremes_indexes = argsort(_min_extremes_indexes)
        _temp_min_trend_points = min_extremum(index=_temp_min_extremes_indexes, eps=eps)

        min_trend_points = _min_extremes_indexes[_temp_min_trend_points]

        return min_trend_points

    def search_max_trend_points(
            self, eps: int, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self.__states_iter - 1

        _max_extremes_indexes = self.get_max_indexes(after_iter=after_iter, item=item)
        _temp_max_extremes_indexes = argsort(_max_extremes_indexes)
        _temp_max_trend_points = max_extremum(index=_temp_max_extremes_indexes, eps=eps)

        max_trend_points = _max_extremes_indexes[_temp_max_trend_points]

        return max_trend_points

        # region Other

    def get_split_index(self) -> int:
        _storage: Type[ExtremesStorage] = self.__states[0]
        return _storage.split_index

    def get_all_values(self) -> np.ndarray[np.float32]:
        _storage: Type[ExtremesStorage] = self.__states[0]
        return _storage.values_all

    def get_current_iter(self) -> int:
        return self.__states_iter

    # endregion Other

    # region Extremum

    def get_combined_indexes(
            self, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        return self.__extract_indexes(attr="extr_all", after_iter=after_iter, item=item)

    def get_combined_values(
            self, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        return self.__extract_values(attr="extr_all", after_iter=after_iter, item=item)

    def get_min_indexes(
            self, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        return self.__extract_indexes(attr="extr_min", after_iter=after_iter, item=item)

    def get_min_values(
            self, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        return self.__extract_values(attr="extr_min", after_iter=after_iter, item=item)

    def get_max_indexes(
            self, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        return self.__extract_indexes(attr="extr_max", after_iter=after_iter, item=item)

    def get_max_values(
            self, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:

        return self.__extract_values(attr="extr_max", after_iter=after_iter, item=item)

    # endregion Extremum

    # region Private

    def __extract_values(
            self, attr: str, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:
        if after_iter is None:
            after_iter = self.__states_iter - 1

        _stages = self.__parse_storage(after_iter, item)

        _temp = []
        for container in _stages:
            _temp.extend(_stages.values_all[getattr(container, attr).indexes])
        return np.array(_temp)

    def __extract_indexes(
            self, attr: str, after_iter: int | None = None, item: int | slice | None = None
    ) -> np.ndarray[np.uint32]:
        if after_iter is None:
            after_iter = self.__states_iter - 1

        _stages = self.__parse_storage(after_iter, item)

        _temp = []
        for container in _stages:
            _temp.extend(getattr(container, attr).indexes)
        return np.array(_temp)

    def __parse_storage(
            self, after_iter: int, item: int | slice | None = None
    ) -> Type[ExtremesStorage] | list[ExtremesStorage]:
        if item is None:
            return self.__states[after_iter]

        if isinstance(item, int):
            return [self.__states[after_iter][item]]

        if isinstance(item, slice):
            return self.__states[after_iter][item]

    # endregion Private


def main():
    size = 13
    step = 3
    values = np.array([np.random.randint(10, 50) for _ in range(size)])
    print(np.arange(size))
    print(values)
    print()

    build(ExtremesStorage, values, size, step)
    extremes_stages = ExtremesSaveState()
    ExtremesStorage.search_extremes(coincident=1, eps=1)
    extremes_stages.save(ExtremesStorage)

    print(extremes_stages.get_combined_indexes())
    print(extremes_stages.get_combined_values())
    print()

    ExtremesStorage.search_extremes(coincident=1, eps=2)
    extremes_stages.save(ExtremesStorage)

    print(extremes_stages.get_combined_indexes())
    print(extremes_stages.get_combined_values())


if __name__ == '__main__':
    np.random.seed(12222)
    main()
