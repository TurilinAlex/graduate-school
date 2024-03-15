from copy import deepcopy
from itertools import zip_longest
from typing import Iterable

import numpy as np

from core.sort import argsort


class CustomList(list):
    def __getitem__(self, index):
        if index is None:
            return self
        elif isinstance(index, (list, np.ndarray)):
            return CustomList([self[i] for i in index])
        elif isinstance(index, slice):
            return CustomList(super().__getitem__(index))
        else:
            return super().__getitem__(index)


class CustomDict(dict):
    __key = [
        "extr_values",
        "extr_indexes",
        "trend_values",
        "trend_indexes",
        "temp_extr",
        "temp_trend",
        "begin",
        "end",
    ]

    def __getitem__(self, key):
        if key not in self.__key:
            raise KeyError(f"Key: {key} not in {self.__key}")
        return deepcopy(super().__getitem__(key))

    def __setitem__(self, key, value):
        if key not in self.__key:
            raise KeyError(f"Key: {key} not in {self.__key}")
        super().__setitem__(key, deepcopy(value))

    def __copy__(self):
        return CustomDict(self)

    def __repr__(self):
        return "\n\t".join(f"{key:<14}: {self.get(key)}" for key in self.__key)


class ExtremesData:
    __key = [
        "all",
        "min",
        "max",
        "extr_eps_min",
        "extr_eps_max",
        "trend_eps_min",
        "trend_eps_max",
    ]

    __temp = {
        "all": CustomDict(
            {
                "extr_indexes": None,
                "extr_values": None,
                "trend_indexes": None,
                "trend_values": None,
                "begin": 0,
                "end": 0,
            }
        ),
        "min": CustomDict(
            {
                "extr_indexes": None,
                "extr_values": None,
                "trend_indexes": None,
                "trend_values": None,
                "temp_extr": CustomList([]),
                "temp_trend": CustomList([]),
                "begin": 0,
                "end": 0,
            }
        ),
        "max": CustomDict(
            {
                "extr_indexes": None,
                "extr_values": None,
                "trend_indexes": None,
                "trend_values": None,
                "temp_extr": CustomList([]),
                "temp_trend": CustomList([]),
                "begin": 0,
                "end": 0,
            }
        ),
        "extr_eps_min": None,
        "extr_eps_max": None,
        "trend_eps_min": None,
        "trend_eps_max": None,
    }

    def __init__(self):
        self.__current_iter: int | None = None
        self.__history_data = {}
        self.__data = None

        self.init()

    def __getitem__(self, key):
        if key not in self.__key:
            raise KeyError(f"Key: {key} not in {self.__key}")
        return self.__data[key]

    def __setitem__(self, key, value):
        if key not in self.__key:
            raise KeyError(f"Key: {key} not in {self.__key}")
        self.__data[key] = deepcopy(value)

    def __repr__(self):
        return (
            f"all:\n\t{self['all']}\n"
            f"min:\n\t{self['min']}\n"
            f"max:\n\t{self['max']}\n"
            f"{'extr_eps_min':<14}: {self['extr_eps_min']}\n"
            f"{'extr_eps_max':<14}: {self['extr_eps_max']}\n"
            f"{'trend_eps_min':<14}: {self['trend_eps_min']}\n"
            f"{'trend_eps_max':<14}: {self['trend_eps_max']}\n"
        )

    def init(self):
        self.__data = deepcopy(self.__temp)

    def save_extremes(self):
        if self.__current_iter is None:
            self.__current_iter = 0
        else:
            self.__current_iter += 1

        self.__history_data[self.__current_iter] = deepcopy(self.__temp)
        self.__history_data[self.__current_iter]["all"]["extr_indexes"] = deepcopy(self.__data["all"]["extr_indexes"])
        self.__history_data[self.__current_iter]["all"]["extr_values"] = deepcopy(self.__data["all"]["extr_values"])
        self.__history_data[self.__current_iter]["all"]["begin"] = self.__data["all"]["begin"]
        self.__history_data[self.__current_iter]["all"]["end"] = self.__data["all"]["end"]

        self.__history_data[self.__current_iter]["min"]["extr_indexes"] = deepcopy(self.__data["min"]["extr_indexes"])
        self.__history_data[self.__current_iter]["min"]["extr_values"] = deepcopy(self.__data["min"]["extr_values"])
        self.__history_data[self.__current_iter]["min"]["begin"] = self.__data["min"]["begin"]
        self.__history_data[self.__current_iter]["min"]["end"] = self.__data["min"]["end"]

        self.__history_data[self.__current_iter]["max"]["extr_indexes"] = deepcopy(self.__data["max"]["extr_indexes"])
        self.__history_data[self.__current_iter]["max"]["extr_values"] = deepcopy(self.__data["max"]["extr_values"])
        self.__history_data[self.__current_iter]["max"]["begin"] = self.__data["max"]["begin"]
        self.__history_data[self.__current_iter]["max"]["end"] = self.__data["max"]["end"]

    def save_extremes_eps(self):
        self.__history_data[self.__current_iter]["extr_eps_min"] = self.__data["extr_eps_min"]
        self.__history_data[self.__current_iter]["extr_eps_max"] = self.__data["extr_eps_max"]

    def save_extremes_temp(self):
        self.__history_data[self.__current_iter]["min"]["temp_extr"] = deepcopy(self.__data["min"]["temp_extr"])
        self.__history_data[self.__current_iter]["max"]["temp_extr"] = deepcopy(self.__data["max"]["temp_extr"])

    def save_trends(self):
        self.__history_data[self.__current_iter]["all"]["trend_indexes"] = deepcopy(self.__data["all"]["trend_indexes"])
        self.__history_data[self.__current_iter]["all"]["trend_values"] = deepcopy(self.__data["all"]["trend_values"])

        self.__history_data[self.__current_iter]["min"]["trend_indexes"] = deepcopy(self.__data["min"]["trend_indexes"])
        self.__history_data[self.__current_iter]["min"]["trend_values"] = deepcopy(self.__data["min"]["trend_values"])

        self.__history_data[self.__current_iter]["max"]["trend_indexes"] = deepcopy(self.__data["max"]["trend_indexes"])
        self.__history_data[self.__current_iter]["max"]["trend_values"] = deepcopy(self.__data["max"]["trend_values"])

    def save_trends_eps(self):
        self.__history_data[self.__current_iter]["trend_eps_min"] = self.__data["trend_eps_min"]
        self.__history_data[self.__current_iter]["trend_eps_max"] = self.__data["trend_eps_max"]

    def save_trends_temp(self):
        self.__history_data[self.__current_iter]["min"]["temp_trend"] = deepcopy(self.__data["min"]["temp_trend"])
        self.__history_data[self.__current_iter]["max"]["temp_trend"] = deepcopy(self.__data["max"]["temp_trend"])

    def restore(self, after_iter: int):
        self.__data = deepcopy(self.__history_data[after_iter])
        self.__current_iter = after_iter

    def clear(self):
        self.init()

        self.__history_data = {}
        self.__current_iter = None

    def get_current_iter(self, after_iter: int | None = None) -> int:
        if after_iter is None:
            return self.__current_iter

        return after_iter

    # region Extremum getter

    def get_extr_min_temp(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["min"]["temp_extr"]

        return self.__history_data[after_iter]["min"]["temp_extr"]

    def get_extr_max_temp(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["max"]["temp_extr"]

        return self.__history_data[after_iter]["max"]["temp_extr"]

    def get_extr_indexes_min(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["min"]["extr_indexes"]

        return self.__history_data[after_iter]["min"]["extr_indexes"]

    def get_extr_indexes_max(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["max"]["extr_indexes"]

        return self.__history_data[after_iter]["max"]["extr_indexes"]

    def get_extr_values_min(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        if after_iter is None:
            return self.__data["min"]["extr_values"]

        return self.__history_data[after_iter]["min"]["extr_values"]

    def get_extr_values_max(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        if after_iter is None:
            return self.__data["max"]["extr_values"]

        return self.__history_data[after_iter]["max"]["extr_values"]

    def get_extr_eps_min(self, after_iter: int | None = None) -> list[int]:
        if after_iter is None:
            return self.__data["extr_eps_min"]

        return self.__history_data[after_iter]["extr_eps_min"]

    def get_extr_eps_max(self, after_iter: int | None = None) -> list[int]:
        if after_iter is None:
            return self.__data["extr_eps_max"]

        return self.__history_data[after_iter]["extr_eps_max"]

    def get_extr_indexes_combined(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["all"]["extr_indexes"]

        return self.__history_data[after_iter]["all"]["extr_indexes"]

    def get_extr_values_combined(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["all"]["extr_values"]

        return self.__history_data[after_iter]["all"]["extr_values"]

    # endregion Extremum getter

    # region Trend getter

    def get_trend_min_temp(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["min"]["temp_trend"]

        return self.__history_data[after_iter]["min"]["temp_trend"]

    def get_trend_max_temp(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["max"]["temp_trend"]

        return self.__history_data[after_iter]["max"]["temp_trend"]

    def get_trend_indexes_min(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["min"]["trend_indexes"]

        return self.__history_data[after_iter]["min"]["trend_indexes"]

    def get_trend_indexes_max(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["max"]["trend_indexes"]

        return self.__history_data[after_iter]["max"]["trend_indexes"]

    def get_trend_values_min(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        if after_iter is None:
            return self.__data["min"]["trend_values"]

        return self.__history_data[after_iter]["min"]["trend_values"]

    def get_trend_values_max(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        if after_iter is None:
            return self.__data["max"]["trend_values"]

        return self.__history_data[after_iter]["max"]["trend_values"]

    def get_trend_eps_min(self, after_iter: int | None = None) -> list[int]:
        if after_iter is None:
            return self.__data["trend_eps_min"]

        return self.__history_data[after_iter]["trend_eps_min"]

    def get_trend_eps_max(self, after_iter: int | None = None) -> list[int]:
        if after_iter is None:
            return self.__data["trend_eps_max"]

        return self.__history_data[after_iter]["trend_eps_max"]

    def get_trend_indexes_combined(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        if after_iter is None:
            return self.__data["all"]["trend_indexes"]

        return self.__history_data[after_iter]["all"]["trend_indexes"]

    def get_trend_values_combined(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        if after_iter is None:
            return self.__data["all"]["trend_values"]

        return self.__history_data[after_iter]["all"]["trend_values"]

    # endregion Trend getter

    # region Interval getter

    def get_extr_begin_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__data["all"]["begin"]

        return self.__history_data[after_iter]["all"]["begin"]

    def get_extr_end_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__data["all"]["end"]

        return self.__history_data[after_iter]["all"]["end"]

    def get_extr_begin_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__data["min"]["begin"]

        return self.__history_data[after_iter]["min"]["begin"]

    def get_extr_end_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__data["min"]["end"]

        return self.__history_data[after_iter]["min"]["end"]

    def get_extr_begin_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__data["max"]["begin"]

        return self.__history_data[after_iter]["max"]["begin"]

    def get_extr_end_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__data["max"]["end"]

        return self.__history_data[after_iter]["max"]["end"]

    # endregion Interval getter

    # region Interval setter

    def set_extr_begin_combined(self, value: int, after_iter: int | None = None):
        if after_iter is None:
            self.__data["all"]["begin"] = value
        else:
            self.__history_data[after_iter]["all"]["begin"] = value

    def set_extr_end_combined(self, value: int, after_iter: int | None = None):
        if after_iter is None:
            self.__data["all"]["end"] = value
        else:
            self.__history_data[after_iter]["all"]["end"] = value

    def set_extr_begin_min(self, value: int, after_iter: int | None = None):
        if after_iter is None:
            self.__data["min"]["begin"] = value
        else:
            self.__history_data[after_iter]["min"]["begin"] = value

    def set_extr_end_min(self, value: int, after_iter: int | None = None):
        if after_iter is None:
            self.__data["min"]["end"] = value
        else:
            self.__history_data[after_iter]["min"]["end"] = value

    def set_extr_begin_max(self, value: int, after_iter: int | None = None):
        if after_iter is None:
            self.__data["max"]["begin"] = value
        else:
            self.__history_data[after_iter]["max"]["begin"] = value

    def set_extr_end_max(self, value: int, after_iter: int | None = None):
        if after_iter is None:
            self.__data["max"]["end"] = value
        else:
            self.__history_data[after_iter]["max"]["end"] = value

    # endregion Interval setter


class ExtremesContainer(ExtremesData):
    __values: CustomList | None = None

    def __init__(
            self,
            values: np.ndarray[np.float32],
            indexes: np.ndarray[np.int32],
            sub_interval: int,
            begin: int,
            end: int,
    ):
        super().__init__()

        self.is_updated_trend = False
        self.is_updated_extr = False

        self["all"]["extr_indexes"] = indexes
        self["min"]["extr_indexes"] = indexes
        self["max"]["extr_indexes"] = indexes
        self["all"]["extr_values"] = values
        self["min"]["extr_values"] = values
        self["max"]["extr_values"] = values
        self["all"]["begin"] = begin
        self["all"]["end"] = end
        self["min"]["begin"] = begin
        self["min"]["end"] = end
        self["max"]["begin"] = begin
        self["max"]["end"] = end

        self.save_extremes()
        self.save_trends()

        self._sub_interval = sub_interval

        self._diff_min_extr: np.ndarray[int] | None = None
        self._diff_max_extr: np.ndarray[int] | None = None

        self._diff_min_trend: np.ndarray[int] | None = None
        self._diff_max_trend: np.ndarray[int] | None = None

    @classmethod
    def set_values(cls, values: np.ndarray[float]):
        cls.__values = CustomList(values)

    @classmethod
    def get_values(cls):
        return cls.__values

    @classmethod
    def add_values(cls, values):
        if isinstance(values, Iterable):
            cls.__values.extend(values)
        else:
            cls.__values.append(values)

    def get_sub_interval(self):
        return self._sub_interval

    def search_extremes(
            self,
            eps: int,
            coincident: int,
            is_set_data: bool = True
    ) -> None:

        self.is_updated_extr = True

        _marker_min, _marker_max = self._diff_between_sort_indexes(eps=eps)
        self["extr_eps_min"] = self._select_eps(
            marker_diff=_marker_min,
            coincident=coincident,
            eps=eps,
        )
        self["extr_eps_max"] = self._select_eps(
            marker_diff=_marker_max,
            coincident=coincident,
            eps=eps,
        )
        self._localize_extremes()

        self.save_extremes_eps()
        self.save_extremes_temp()

        if is_set_data:
            self["min"]["extr_indexes"] = self["all"]["extr_indexes"][self["min"]["temp_extr"]]
            self["max"]["extr_indexes"] = self["all"]["extr_indexes"][self["max"]["temp_extr"]]
            self["all"]["extr_indexes"] = np.sort(
                np.hstack([self["min"]["extr_indexes"], self["max"]["extr_indexes"]])
            )

            self["min"]["extr_values"] = self.__values[self["min"]["extr_indexes"]]
            self["max"]["extr_values"] = self.__values[self["max"]["extr_indexes"]]
            self["all"]["extr_values"] = self.__values[self["all"]["extr_indexes"]]

    def search_trends(
            self,
            eps_for_min: int,
            eps_for_max: int,
            is_set_data: bool = True
    ):
        self.is_updated_trend = True

        self["trend_eps_min"] = eps_for_min
        self["trend_eps_max"] = eps_for_max

        self._diff_between_sort_indexes_trend(
            eps_for_min=eps_for_min,
            eps_for_max=eps_for_max,
        )

        self._localize_trend()

        self.save_trends_eps()
        self.save_trends_temp()

        if is_set_data:
            self["min"]["trend_indexes"] = self["min"]["trend_indexes"][self["min"]["temp_trend"]]
            self["max"]["trend_indexes"] = self["max"]["trend_indexes"][self["max"]["temp_trend"]]
            self["all"]["trend_indexes"] = np.sort(
                np.hstack([self["min"]["trend_indexes"], self["max"]["trend_indexes"]])
            )

            self["min"]["trend_values"] = self.__values[self["min"]["trend_indexes"]]
            self["max"]["trend_values"] = self.__values[self["max"]["trend_indexes"]]
            self["all"]["trend_values"] = self.__values[self["all"]["trend_indexes"]]

    def _diff_between_sort_indexes(self, eps: int = 1) -> tuple[np.ndarray, np.ndarray]:

        _indexes = argsort(self["all"]["extr_values"])

        n = len(_indexes)
        self._diff_min_extr = np.empty_like(_indexes, dtype=np.uint32)
        self._diff_max_extr = np.empty_like(_indexes, dtype=np.uint32)

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

            self._diff_min_extr[_indexes[i]] = min_diff_for_minimal
            marker_diff_for_minimal[min_diff_for_minimal] = 1

            # max
            for j in range(1, (n - i)):
                diff = abs(_indexes[i] - _indexes[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= eps:
                    break

            self._diff_max_extr[_indexes[i]] = min_diff_for_maximal
            marker_diff_for_maximal[min_diff_for_maximal] = 1

        # endregion Calculating the difference between indexes

        return marker_diff_for_minimal, marker_diff_for_maximal

    def _diff_between_sort_indexes_trend(self, eps_for_min: int, eps_for_max: int) -> None:

        _indexes_min = argsort(self["min"]["extr_values"])
        _indexes_max = argsort(self["max"]["extr_values"])

        n_min = len(_indexes_min)
        n_max = len(_indexes_max)

        self._diff_min_trend = np.empty_like(_indexes_min, dtype=np.uint32)
        self._diff_max_trend = np.empty_like(_indexes_max, dtype=np.uint32)

        for i in range(n_min):
            min_diff_for_minimal = n_min

            # min
            for j in range(1, i + 1):
                diff = abs(_indexes_min[i] - _indexes_min[i - j])
                if diff < min_diff_for_minimal:
                    min_diff_for_minimal = diff
                if min_diff_for_minimal <= eps_for_min:
                    break

            self._diff_min_trend[_indexes_min[i]] = min_diff_for_minimal

        for i in range(n_max):
            min_diff_for_maximal = n_max

            # max
            for j in range(1, (n_max - i)):
                diff = abs(_indexes_max[i] - _indexes_max[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= eps_for_max:
                    break

            self._diff_max_trend[_indexes_max[i]] = min_diff_for_maximal

    def _localize_extremes(self):
        _batch = len(self._diff_min_extr)

        _extr_min_index = np.empty_like(self._diff_min_extr)
        _extr_max_index = np.empty_like(self._diff_max_extr)

        _k_min = 0
        _k_max = 0
        for i in range(_batch):
            is_min_extr = self._diff_min_extr[i] > self["extr_eps_min"] or self._diff_min_extr[i] == _batch
            if is_min_extr:
                _extr_min_index[_k_min] = i
                _k_min += 1

            is_max_extr = self._diff_max_extr[i] > self["extr_eps_max"] or self._diff_max_extr[i] == _batch
            if is_max_extr:
                _extr_max_index[_k_max] = i
                _k_max += 1

        self["min"]["temp_extr"] = _extr_min_index[:_k_min]
        self["max"]["temp_extr"] = _extr_max_index[:_k_max]

    def _localize_trend(self):
        _batch_min = len(self._diff_min_trend)
        _batch_max = len(self._diff_max_trend)

        _trend_min_index = np.empty_like(self._diff_min_trend)
        _trend_max_index = np.empty_like(self._diff_max_trend)

        _k_min = 0
        _k_max = 0
        for i, j in zip_longest(range(_batch_min), range(_batch_max)):
            if i is not None:
                is_min_extr = self._diff_min_trend[i] > self["trend_eps_min"] or self._diff_min_trend[i] == _batch_min
                if is_min_extr:
                    _trend_min_index[_k_min] = i
                    _k_min += 1

            if j is not None:
                is_max_extr = self._diff_max_trend[j] > self["trend_eps_max"] or self._diff_max_trend[j] == _batch_max
                if is_max_extr:
                    _trend_max_index[_k_max] = j
                    _k_max += 1

        self["min"]["temp_trend"] = _trend_min_index[:_k_min]
        self["max"]["temp_trend"] = _trend_max_index[:_k_max]

    @staticmethod
    def _select_eps(marker_diff: np.ndarray, coincident: int, eps: int) -> int:
        count_zero = 0
        last_non_zero_index = eps
        for i in range(eps + 1, len(marker_diff)):
            if count_zero >= coincident - 1:
                _select_eps = last_non_zero_index + coincident - 1
                return _select_eps

            if marker_diff[i] == 0:
                count_zero += 1
            else:
                count_zero = 0
                last_non_zero_index = i

        _select_eps = last_non_zero_index + coincident - 1

        return _select_eps


class ExtremesStorage:
    __values: CustomList | None = None

    def __init__(self, batch: int):
        self._storage: CustomList[ExtremesContainer] = CustomList()
        self.__iter: int = 0
        self.__batch: int = batch

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return iter(self._storage)

    def __next__(self):
        if self.__iter < len(self._storage):
            __data = self._storage[self.__iter]
            self.__iter += 1
            return __data
        else:
            raise StopIteration

    def __getitem__(
            self, item: int | slice | None
    ) -> ExtremesContainer | list[ExtremesContainer]:
        if item is None:
            return self._storage

        if isinstance(item, int | slice):
            return self._storage[item]

        raise IndexError(f"Unsupported index value: {item}")

    def __repr__(self):
        return "\n".join(f"============{data.get_sub_interval():^10}============\n{data!r}" for data in self)

    @classmethod
    def set_values(cls, values):
        cls.__values = CustomList(values)

    @classmethod
    def add_values(cls, values):
        if isinstance(values, Iterable):
            cls.__values.extend(values)
        else:
            cls.__values.append(values)

    def search_extremes(
            self,
            coincident: int,
            eps: int,
            item: int | slice | None = None,
    ) -> None:

        if isinstance(item, int):
            __data = [self[item]]
        elif item is None or isinstance(item, slice):
            __data = self[item]
        else:
            raise IndexError(f"Unsupported index: {item}")

        _begin_all = __data[0].get_extr_begin_combined()
        _begin_min = __data[0].get_extr_begin_min()
        _begin_max = __data[0].get_extr_begin_max()

        for data in __data:
            data.search_extremes(
                coincident=coincident,
                eps=eps,
                is_set_data=False
            )

    def search_trend(
            self,
            eps_for_min: int,
            eps_for_max: int,
            item: int | slice | None = None,
    ) -> None:

        if isinstance(item, int):
            __data = [self[item]]
        elif item is None or isinstance(item, slice):
            __data = self[item]
        else:
            raise IndexError(f"Unsupported index: {item}")

        for data in __data:
            data.search_trends(
                eps_for_min=eps_for_min,
                eps_for_max=eps_for_max,
                is_set_data=False
            )

    def filter_extremes(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> None:

        if isinstance(item, int):
            __data = [self[item]]
        elif item is None or isinstance(item, slice):
            __data = self[item]
        else:
            raise IndexError(f"Unsupported index: {item}")

        for data in __data:
            min_mask = self._filter_min_extremes(sub_interval=data.get_sub_interval(), after_iter=after_iter)
            max_mask = self._filter_max_extremes(sub_interval=data.get_sub_interval(), after_iter=after_iter)

            data["min"]["temp_extr"] = data.get_extr_min_temp(after_iter=after_iter)[min_mask]
            data["max"]["temp_extr"] = data.get_extr_max_temp(after_iter=after_iter)[max_mask]

            data["min"]["extr_indexes"] = (
                data.get_extr_indexes_combined(after_iter=after_iter)
            )[data["min"]["temp_extr"]]

            data["max"]["extr_indexes"] = (
                data.get_extr_indexes_combined(after_iter=after_iter)
            )[data["max"]["temp_extr"]]

            data["all"]["extr_indexes"] = np.sort(
                np.hstack([data.get_extr_indexes_min(), data.get_extr_indexes_max()])
            )

            data["min"]["extr_values"] = data.get_values()[data.get_extr_indexes_min()]
            data["max"]["extr_values"] = data.get_values()[data.get_extr_indexes_max()]
            data["all"]["extr_values"] = data.get_values()[data.get_extr_indexes_combined()]

    def filter_trends(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> None:

        if isinstance(item, int):
            __data = [self[item]]
        elif item is None or isinstance(item, slice):
            __data = self[item]
        else:
            raise IndexError(f"Unsupported index: {item}")

        for data in __data:
            min_mask = self._filter_min_trends(sub_interval=data.get_sub_interval(), after_iter=after_iter)
            max_mask = self._filter_max_trends(sub_interval=data.get_sub_interval(), after_iter=after_iter)

            data["min"]["temp_trend"] = data.get_trend_min_temp(after_iter=after_iter)[min_mask]
            data["max"]["temp_trend"] = data.get_trend_max_temp(after_iter=after_iter)[max_mask]

            data["min"]["trend_indexes"] = (
                data.get_extr_indexes_min(after_iter=after_iter)
            )[data["min"]["temp_trend"]]

            data["max"]["trend_indexes"] = (
                data.get_extr_indexes_max(after_iter=after_iter)
            )[data["max"]["temp_trend"]]

            data["all"]["trend_indexes"] = np.sort(
                np.hstack([data.get_trend_indexes_min(), data.get_trend_indexes_max()])
            )

            data["min"]["trend_values"] = data.get_values()[data.get_trend_indexes_min()]
            data["max"]["trend_values"] = data.get_values()[data.get_trend_indexes_max()]
            data["all"]["trend_values"] = data.get_values()[data.get_trend_indexes_combined()]

    def save_extremes(
            self,
            item: int | slice | None = None,
    ) -> None:

        if isinstance(item, int):
            __data = [self[item]]
        elif item is None or isinstance(item, slice):
            __data = self[item]
        else:
            raise IndexError(f"Unsupported index: {item}")

        _begin_all = __data[0].get_extr_begin_combined()
        _begin_min = __data[0].get_extr_begin_min()
        _begin_max = __data[0].get_extr_begin_max()

        for data in __data:
            data.set_extr_begin_combined(value=_begin_all)
            _end_all = data.get_extr_begin_combined() + len(data.get_extr_indexes_combined())
            data.set_extr_end_combined(value=_end_all)
            _begin_all = _end_all

            data.set_extr_begin_min(value=_begin_min)
            _end_min = data.get_extr_begin_min() + len(data.get_extr_indexes_min())
            data.set_extr_end_min(value=_end_min)
            _begin_min = _end_min

            data.set_extr_begin_max(value=_begin_max)
            _end_max = data.get_extr_begin_max() + len(data.get_extr_indexes_max())
            data.set_extr_end_max(value=_end_max)
            _begin_max = _end_max

            data.save_extremes()

    def save_trends(
            self,
            item: int | slice | None = None,
    ) -> None:

        if isinstance(item, int):
            __data = [self[item]]
        elif item is None or isinstance(item, slice):
            __data = self[item]
        else:
            raise IndexError(f"Unsupported index: {item}")

        for data in __data:
            data.save_trends()

    def _filter_min_extremes(self, sub_interval: int, after_iter: int | None = None):
        _left = True
        _right = True

        data = self._storage[sub_interval]
        if after_iter is None:
            after_iter = data.get_current_iter()

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1
        eps_min = data.get_extr_eps_min(after_iter=after_iter)
        begin = data.get_extr_begin_combined(after_iter=after_iter)
        end = data.get_extr_end_combined(after_iter=after_iter)

        __extr_value = data.get_extr_values_combined(after_iter=after_iter)

        _mask = []
        for _extr_index in data.get_extr_min_temp(after_iter=after_iter):
            max_check_previous = -1 * int(_extr_index) + eps_min
            _extr_value = __extr_value[_extr_index]
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self._storage[previous_sub_interval].get_extr_values_combined(
                        after_iter=after_iter,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value <= _extr_value:
                            _left = False

            max_check_next = (begin + _extr_index + eps_min) - end
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self._storage[next_sub_interval].get_extr_values_combined(
                        after_iter=after_iter,
                    )
                    next_sub_interval += 1
                    for value in next_extr_combined:
                        if num_check > max_check_next:
                            break

                        if value < _extr_value:
                            _right = False

                        num_check += 1

            _mask.append(_left and _right)

        return _mask

    def _filter_min_trends(self, sub_interval: int, after_iter: int | None = None):
        _left = True
        _right = True

        data = self._storage[sub_interval]
        if after_iter is None:
            after_iter = data.get_current_iter()

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1
        eps_min = data.get_trend_eps_min(after_iter=after_iter)
        begin = data.get_extr_begin_min(after_iter=after_iter)
        end = data.get_extr_end_min(after_iter=after_iter)

        __extr_value = data.get_extr_values_min(after_iter=after_iter)

        _mask = []
        for _extr_index in data.get_trend_min_temp(after_iter=after_iter):
            max_check_previous = -1 * int(_extr_index) + eps_min
            _extr_value = __extr_value[_extr_index]
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self._storage[previous_sub_interval].get_extr_values_min(
                        after_iter=after_iter,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value <= _extr_value:
                            _left = False

            max_check_next = (begin + _extr_index + eps_min) - end
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self._storage[next_sub_interval].get_extr_values_min(
                        after_iter=after_iter,
                    )
                    next_sub_interval += 1
                    for value in next_extr_combined:
                        if num_check > max_check_next:
                            break

                        if value < _extr_value:
                            _right = False

                        num_check += 1

            _mask.append(_left and _right)

        return _mask

    def _filter_max_extremes(self, sub_interval: int, after_iter: int | None = None):
        _left = True
        _right = True

        data = self._storage[sub_interval]
        if after_iter is None:
            after_iter = data.get_current_iter()

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1
        eps_max = data.get_extr_eps_max(after_iter=after_iter)
        begin = data.get_extr_begin_combined(after_iter=after_iter)
        end = data.get_extr_end_combined(after_iter=after_iter)

        __extr_value = data.get_extr_values_combined(after_iter=after_iter)

        _mask = []
        for _extr_index in data.get_extr_max_temp(after_iter=after_iter):
            max_check_previous = -1 * int(_extr_index) + eps_max
            _extr_value = __extr_value[_extr_index]
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self._storage[previous_sub_interval].get_extr_values_combined(
                        after_iter=after_iter,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value > _extr_value:
                            _left = False

            max_check_next = (begin + _extr_index + eps_max) - end
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self._storage[next_sub_interval].get_extr_values_combined(
                        after_iter=after_iter,
                    )
                    next_sub_interval += 1
                    for value in next_extr_combined:
                        if num_check > max_check_next:
                            break
                        if value >= _extr_value:
                            _right = False

                        num_check += 1

            _mask.append(_left and _right)

        return _mask

    def _filter_max_trends(self, sub_interval: int, after_iter: int | None = None):
        _left = True
        _right = True

        data = self._storage[sub_interval]
        if after_iter is None:
            after_iter = data.get_current_iter()

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1
        eps_max = data.get_trend_eps_max(after_iter=after_iter)
        begin = data.get_extr_begin_max(after_iter=after_iter)
        end = data.get_extr_end_max(after_iter=after_iter)

        __extr_value = data.get_extr_values_max(after_iter=after_iter)

        _mask = []
        for _extr_index in data.get_trend_max_temp(after_iter=after_iter):
            max_check_previous = -1 * int(_extr_index) + eps_max
            _extr_value = __extr_value[_extr_index]
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self._storage[previous_sub_interval].get_extr_values_max(
                        after_iter=after_iter,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value > _extr_value:
                            _left = False

            max_check_next = (begin + _extr_index + eps_max) - end
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self._storage[next_sub_interval].get_extr_values_max(
                        after_iter=after_iter,
                    )
                    next_sub_interval += 1
                    for value in next_extr_combined:
                        if num_check > max_check_next:
                            break
                        if value >= _extr_value:
                            _right = False

                        num_check += 1

            _mask.append(_left and _right)

        return _mask

    def build(self, values: np.ndarray[float], split: int):

        ExtremesContainer.set_values(values=values)

        for begin in range(0, split, self.__batch):
            end = min(begin + self.__batch, split)
            container = ExtremesContainer(
                values=values[begin: end],
                indexes=np.arange(begin, end, dtype=int),
                begin=begin,
                end=end,
                sub_interval=begin // self.__batch
            )

            self._storage.append(container)

    def add(self, value: float | list | np.ndarray):

        if not isinstance(value, Iterable):
            value = [value]

        ExtremesContainer.add_values(values=value)

        _begin = self[-1].get_extr_begin_combined(after_iter=0)
        _end = self[-1].get_extr_end_combined(after_iter=0)

        for begin in range(_begin, _end + len(value), self.__batch):
            end = min(begin + self.__batch, _end + len(value))

            if end == _end:
                continue

            container = ExtremesContainer(
                values=ExtremesContainer.get_values()[begin: end],
                indexes=np.arange(begin, end, dtype=int),
                begin=begin,
                end=end,
                sub_interval=begin // self.__batch
            )

            if begin == _begin:
                self._storage[-1] = container
            else:
                self._storage.append(container)

    def get_current_iter(
            self,
            item: int | slice | None = None,
    ):
        return self.__parse_getter(
            callable_name="get_current_iter",
            item=item,
        )

    def get_extr_indexes_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_indexes_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_values_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_values_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_indexes_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_indexes_min",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_values_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_values_min",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_indexes_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_indexes_max",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_values_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_values_max",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_indexes_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_trend_indexes_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_values_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_trend_values_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_indexes_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_trend_indexes_min",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_values_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_trend_values_min",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_indexes_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_trend_indexes_max",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_values_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_trend_values_max",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_eps_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_eps_min",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_eps_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__parse_getter(
            callable_name="get_extr_eps_max",
            after_iter=after_iter,
            item=item,
        )

    def __parse_getter(
            self,
            callable_name: str,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ):
        if isinstance(item, int):
            return getattr(self[item], callable_name)(after_iter=after_iter)

        __data = []
        for data in self[item]:
            __values = getattr(data, callable_name)(after_iter=after_iter)
            if __values is not None:
                if not isinstance(__values, Iterable):
                    __values = [__values]
                __data.extend(__values)
        return np.array(__data)


def main():
    values = np.array([13, 48, 36, 26, 11, 14, 24, 11, 13, 36, 41, 26, 21])
    # values = np.array([23, 49, 31, 37, 36, 35, 36, 49, 45, 13, 15, 33, 28])

    # np.random.seed(455)
    # values = np.array([np.random.randint(10, 50) for _ in range(size)])

    size = 13
    indexes = np.arange(size)
    print(indexes)
    print(values)
    print()

    stor = ExtremesStorage(batch=3)
    stor.build(values=values, split=13)

    stor.search_extremes(coincident=1, eps=2)
    stor.filter_extremes()
    stor.save_extremes()

    stor.search_trend(eps_for_min=2, eps_for_max=2)
    stor.filter_trends()
    stor.save_trends()

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print()

    print(stor.get_trend_indexes_combined())
    print(stor.get_trend_values_combined())

    print(stor.get_trend_indexes_min())
    print(stor.get_trend_values_min())

    print(stor.get_trend_indexes_max())
    print(stor.get_trend_values_max())

    print()
    print("----------")
    print()

    stor.search_extremes(coincident=1, eps=2)
    stor.filter_extremes()
    stor.save_extremes()
    stor.search_trend(eps_for_min=2, eps_for_max=2)
    stor.filter_trends()
    stor.save_trends()

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print()

    print(stor.get_trend_indexes_combined())
    print(stor.get_trend_values_combined())

    print(stor.get_trend_indexes_min())
    print(stor.get_trend_values_min())

    print(stor.get_trend_indexes_max())
    print(stor.get_trend_values_max())

    print()
    print("----------")
    print()


def main_dynamic():
    values = np.array([13, 48, 36, 26, 11, 14, 24, 11, 13, 36, 41, 26, 21])
    size = len(values)
    indexes = np.arange(size)
    print(indexes)
    print(values)

    stor = ExtremesStorage(batch=3)
    stor.build(values=values, split=size)

    stor.search_extremes(coincident=1, eps=13)
    stor.filter_extremes()
    stor.save_extremes()

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print("------------------------")

    stor.search_extremes(coincident=1, eps=6)
    stor.filter_extremes()
    stor.save_extremes()

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print("###########################")

    stor.add(49)

    stor.search_extremes(coincident=1, eps=6, item=slice(-1, None, None))
    stor.filter_extremes(after_iter=0, item=slice(-6, None, None))
    stor.save_extremes(item=slice(-6, None, None))

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print("--------")

    stor.search_extremes(coincident=1, eps=6, item=slice(-1, None, None))
    stor.filter_extremes(after_iter=1, item=slice(-6, None, None))
    stor.save_extremes(item=slice(-6, None, None))

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print("--------")

    stor.add(9)

    stor.search_extremes(coincident=1, eps=6, item=slice(-1, None, None))
    stor.filter_extremes(after_iter=0, item=slice(-6, None, None))
    stor.save_extremes(item=slice(-6, None, None))

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print("--------")

    stor.search_extremes(coincident=1, eps=6, item=slice(-1, None, None))
    stor.filter_extremes(after_iter=1, item=slice(-6, None, None))
    stor.save_extremes(item=slice(-6, None, None))

    print(stor.get_extr_indexes_combined())
    print(stor.get_extr_values_combined())

    print(stor.get_extr_indexes_min())
    print(stor.get_extr_values_min())

    print(stor.get_extr_indexes_max())
    print(stor.get_extr_values_max())

    print("--------")


if __name__ == '__main__':
    main_dynamic()
