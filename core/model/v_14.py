from copy import deepcopy
from dataclasses import dataclass, field, asdict
from itertools import zip_longest
from typing import Iterable, Sequence

import numpy as np

from core.sort import argsort


class ExtendedList(list):

    def __getitem__(self, item):
        if isinstance(item, (int, np.int8, np.int16, np.int32, np.int64)):
            return super().__getitem__(item)
        elif isinstance(item, slice):
            return ExtendedList(super().__getitem__(item))
        elif isinstance(item, Sequence):
            if not item:
                return ExtendedList()
            elif isinstance(item[0], bool):
                return ExtendedList(item for item, flag in zip(self, item) if flag)
            elif isinstance(item[0], int):
                return ExtendedList(self[i] for i in item)
        elif item is None:
            return ExtendedList(self)

        raise TypeError(f"Unsupported index type! {item} {type(item)}")

    def __repr__(self):
        return f"{np.array(self)}"


@dataclass
class AllExtremes:
    extr_values: ExtendedList = field(default=None)
    extr_indexes: ExtendedList = field(default=None)
    trend_values: ExtendedList = field(default=None)
    trend_indexes: ExtendedList = field(default=None)

    def __repr__(self):
        field_values = asdict(self)
        return "\n".join(f"\t{key:<14}: {value}" for key, value in field_values.items() if value is not None)


@dataclass
class MinExtremes(AllExtremes):
    temp_extr: ExtendedList = field(default=None)
    temp_trend: ExtendedList = field(default=None)

    def __repr__(self):
        field_values = asdict(self)
        return "\n".join(f"\t{key:<14}: {value}" for key, value in field_values.items() if value is not None)


@dataclass
class MaxExtremes(AllExtremes):
    temp_extr: ExtendedList = field(default=None)
    temp_trend: ExtendedList = field(default=None)

    def __repr__(self):
        field_values = asdict(self)
        return "\n".join(f"\t{key:<14}: {value}" for key, value in field_values.items() if value is not None)


@dataclass
class ExtremesData:
    all: AllExtremes = field(default_factory=AllExtremes)
    min: MinExtremes = field(default_factory=MinExtremes)
    max: MaxExtremes = field(default_factory=MaxExtremes)
    extr_eps_min: int = field(default=None)
    extr_eps_max: int = field(default=None)
    trend_eps_min: int = field(default=None)
    trend_eps_max: int = field(default=None)

    def __repr__(self):
        all_ = f"=== all ==============================\n{self.all}\n"
        min_ = f"=== min ==============================\n{self.min}\n"
        max_ = f"=== max ==============================\n{self.max}\n"
        fill = "--------------------------------------\n"
        extr_eps_min = f"{'extr_eps_min':<18}: {self.extr_eps_min}\n" if self.extr_eps_min is not None else ""
        extr_eps_max = f"{'extr_eps_max':<18}: {self.extr_eps_max}\n" if self.extr_eps_max is not None else ""
        trend_eps_min = f"{'trend_eps_min':<18}: {self.trend_eps_min}\n" if self.trend_eps_min is not None else ""
        trend_eps_max = f"{'trend_eps_max':<18}: {self.trend_eps_max}\n" if self.trend_eps_max is not None else ""
        return f"{all_}{min_}{max_}{fill}{extr_eps_min}{extr_eps_max}{trend_eps_min}{trend_eps_max}"


class ExtremesBase:
    __values: ExtendedList[float] | None = None

    def __init__(self):
        self.__extr_iter: int = 0
        self.__history_data: dict[int, ExtremesData] = {0: ExtremesData()}
        self.__current_data: ExtremesData = ExtremesData()

    def __repr__(self):
        return f"{self.__current_data!r}"

    @classmethod
    def set_values(cls, values):
        cls.__values = ExtendedList(values)

    @classmethod
    def get_values(cls):
        return cls.__values

    @classmethod
    def append(cls, values: float | Iterable):
        if isinstance(values, Iterable):
            cls.__values.extend(values)
        else:
            cls.__values.append(values)

    @property
    def current(self):
        return self.__current_data

    @property
    def history(self):
        return self.__history_data

    def add_extremes_iter(self):
        self.__extr_iter += 1

    def print_current(self):
        return f"current data\n {self.__current_data}"

    def print_history(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        return f"history data, {after_iter=} \n {self.__history_data[after_iter]}"

    def save_extremes_data(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        self._check_history_data_for_save(after_iter=after_iter)

        self.__history_data[after_iter].all.extr_indexes = deepcopy(self.__current_data.all.extr_indexes)
        self.__history_data[after_iter].all.extr_values = deepcopy(self.__current_data.all.extr_values)

        self.__history_data[after_iter].min.extr_indexes = deepcopy(self.__current_data.min.extr_indexes)
        self.__history_data[after_iter].min.extr_values = deepcopy(self.__current_data.min.extr_values)

        self.__history_data[after_iter].max.extr_indexes = deepcopy(self.__current_data.max.extr_indexes)
        self.__history_data[after_iter].max.extr_values = deepcopy(self.__current_data.max.extr_values)

    def save_extremes_eps(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        self._check_history_data_for_save(after_iter=after_iter)

        self.__history_data[after_iter].extr_eps_min = self.__current_data.extr_eps_min
        self.__history_data[after_iter].extr_eps_max = self.__current_data.extr_eps_max

    def save_extremes_temp(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        self._check_history_data_for_save(after_iter=after_iter)

        self.__history_data[after_iter].min.temp_extr = deepcopy(self.__current_data.min.temp_extr)
        self.__history_data[after_iter].max.temp_extr = deepcopy(self.__current_data.max.temp_extr)

    def save_trends_data(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        self._check_history_data_for_save(after_iter=after_iter)

        self.__history_data[after_iter].all.trend_indexes = deepcopy(self.__current_data.all.trend_indexes)
        self.__history_data[after_iter].all.trend_values = deepcopy(self.__current_data.all.trend_values)

        self.__history_data[after_iter].min.trend_indexes = deepcopy(self.__current_data.min.trend_indexes)
        self.__history_data[after_iter].min.trend_values = deepcopy(self.__current_data.min.trend_values)

        self.__history_data[after_iter].max.trend_indexes = deepcopy(self.__current_data.max.trend_indexes)
        self.__history_data[after_iter].max.trend_values = deepcopy(self.__current_data.max.trend_values)

    def save_trends_eps(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        self._check_history_data_for_save(after_iter=after_iter)

        self.__history_data[after_iter].trend_eps_min = self.__current_data.trend_eps_min
        self.__history_data[after_iter].trend_eps_max = self.__current_data.trend_eps_max

    def save_trends_temp(self, after_iter: int | None = None):
        after_iter = self._validate_after_iter(after_iter=after_iter)
        self._check_history_data_for_save(after_iter=after_iter)

        self.__history_data[after_iter].min.temp_trend = deepcopy(self.__current_data.min.temp_trend)
        self.__history_data[after_iter].max.temp_trend = deepcopy(self.__current_data.max.temp_trend)

    def get_current_iter(self, after_iter: int | None = None) -> int:
        return self._validate_after_iter(after_iter=after_iter)

    # region Extremum getter

    def get_extr_indexes_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.all.extr_indexes

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].all.extr_indexes

    def get_extr_indexes_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.extr_indexes

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.extr_indexes

    def get_extr_indexes_min_temp(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.temp_extr

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.temp_extr

    def get_extr_indexes_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.extr_indexes

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.extr_indexes

    def get_extr_indexes_max_temp(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.temp_extr

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.temp_extr

    def get_extr_values_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.all.extr_values

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].all.extr_values

    def get_extr_values_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.extr_values

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.extr_values

    def get_extr_values_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.extr_values

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.extr_values

    def get_extr_eps_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.extr_eps_min

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].extr_eps_min

    def get_extr_eps_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.extr_eps_max

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].extr_eps_max

    # endregion Extremum getter

    # region Trend getter

    def get_trend_indexes_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.all.trend_indexes

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].all.trend_indexes

    def get_trend_indexes_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.trend_indexes

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.trend_indexes

    def get_trend_indexes_min_temp(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.temp_trend

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.temp_trend

    def get_trend_indexes_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.trend_indexes

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.trend_indexes

    def get_trend_indexes_max_temp(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.temp_trend

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.temp_trend

    def get_trend_values_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.all.trend_values

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].all.trend_values

    def get_trend_values_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.trend_values

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.trend_values

    def get_trend_values_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.trend_values

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.trend_values

    def get_trend_eps_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.trend_eps_min

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].trend_eps_min

    def get_trend_eps_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.trend_eps_max

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].trend_eps_max

    # endregion Trend getter

    def _validate_after_iter(self, after_iter: int | None) -> int:
        if after_iter is None:
            return self.__extr_iter

        if 0 <= after_iter <= self.__extr_iter:
            return after_iter

        raise ValueError(
            f"0 <= after_iter value <= self.__extr_iter! {after_iter=}, {self.__extr_iter=}"
        )

    def _check_history_data_for_save(self, after_iter: int) -> None:
        if self.__history_data.get(after_iter) is None:
            self.__history_data[after_iter] = ExtremesData()


class Extremes(ExtremesBase):

    def __init__(
            self,
            values: list[float],
            indexes: list[int],
            sub_interval: int | None = None,
    ):
        super().__init__()

        self.current.all.extr_values = ExtendedList(values)
        self.current.all.extr_indexes = ExtendedList(indexes)

        self.current.min.extr_values = ExtendedList(values)
        self.current.min.extr_indexes = ExtendedList(indexes)

        self.current.max.extr_values = ExtendedList(values)
        self.current.max.extr_indexes = ExtendedList(indexes)

        self._sub_interval: int | None = sub_interval

        self.save_extremes_data()

    def get_sub_interval(self):
        return self._sub_interval

    def search_extremes(
            self,
            eps_min: int,
            eps_max: int,
            coincident: int,
    ) -> None:
        self.add_extremes_iter()

        _marker_min, _marker_max, _diff_min_extr, _diff_max_extr = (
            self._diff_between_indexes_for_extremes(
                eps_min=eps_min,
                eps_max=eps_max,
            )
        )

        self._select_eps_min(
            marker_diff_min=_marker_min,
            coincident=coincident,
            eps=eps_min,
        )
        self._select_eps_max(
            marker_diff_max=_marker_max,
            coincident=coincident,
            eps=eps_max,
        )
        self._localize_extremes(
            diff_min_extr=_diff_min_extr,
            diff_max_extr=_diff_max_extr,
        )

        self.__set_extremes_data()
        self.__save_extremes_data()

    def search_trends(
            self,
            eps_for_min: int,
            eps_for_max: int,
            after_iter: int
    ):
        after_iter = self._validate_after_iter(after_iter=after_iter)

        self.current.trend_eps_min = eps_for_min
        self.current.trend_eps_max = eps_for_max

        _diff_min_extr, _diff_max_extr = (
            self._diff_between_indexes_for_trends(
                eps_for_min=eps_for_min,
                eps_for_max=eps_for_max,
                after_iter=after_iter,
            )
        )

        self._localize_trends(
            diff_min_extr=_diff_min_extr,
            diff_max_extr=_diff_max_extr,
        )

        self.__set_trends_data(after_iter=after_iter)
        self.__save_trends_data(after_iter=after_iter)

    def _diff_between_indexes_for_extremes(self, eps_min: int, eps_max: int):

        _indexes = argsort(self.current.all.extr_values)
        n = len(_indexes)

        diff_min_extr = [0] * n
        diff_max_extr = [0] * n

        # Stores labels about the value of the index difference for local minima
        # That is abs(_index[i] - _index[i - j]) = _diff -> marker_diff_for_minimal[_diff] = 1
        marker_diff_for_minimal = [0] * (n + 1)

        # Stores labels about the value of the index difference for local maxima
        # That is abs(_index[i] - _index[i + j]) = _diff -> marker_diff_for_maximal[_diff] = 1
        marker_diff_for_maximal = [0] * (n + 1)

        # region Calculating the difference between indexes

        for i in range(n):
            min_diff_for_minimal, min_diff_for_maximal = n, n

            # min
            for j in range(1, i + 1):
                diff = abs(_indexes[i] - _indexes[i - j])
                if diff < min_diff_for_minimal:
                    min_diff_for_minimal = diff
                if min_diff_for_minimal <= eps_min:
                    break

            diff_min_extr[_indexes[i]] = min_diff_for_minimal
            marker_diff_for_minimal[min_diff_for_minimal] = 1

            # max
            for j in range(1, (n - i)):
                diff = abs(_indexes[i] - _indexes[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= eps_max:
                    break

            diff_max_extr[_indexes[i]] = min_diff_for_maximal
            marker_diff_for_maximal[min_diff_for_maximal] = 1

        # endregion Calculating the difference between indexes

        return marker_diff_for_minimal, marker_diff_for_maximal, diff_min_extr, diff_max_extr

    def _diff_between_indexes_for_trends(self, eps_for_min: int, eps_for_max: int, after_iter: int):
        # sourcery skip: use-assigned-variable

        _indexes_min = argsort(self.get_extr_values_min(after_iter=after_iter))
        _indexes_max = argsort(self.get_extr_values_max(after_iter=after_iter))

        n_min = len(_indexes_min)
        n_max = len(_indexes_max)

        diff_min_trend = [0] * n_min
        diff_max_trend = [0] * n_max

        for i in range(n_min):
            min_diff_for_minimal = n_min

            # min
            for j in range(1, i + 1):
                diff = abs(_indexes_min[i] - _indexes_min[i - j])
                if diff < min_diff_for_minimal:
                    min_diff_for_minimal = diff
                if min_diff_for_minimal <= eps_for_min:
                    break

            diff_min_trend[_indexes_min[i]] = min_diff_for_minimal

        for i in range(n_max):
            min_diff_for_maximal = n_max

            # max
            for j in range(1, (n_max - i)):
                diff = abs(_indexes_max[i] - _indexes_max[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= eps_for_max:
                    break

            diff_max_trend[_indexes_max[i]] = min_diff_for_maximal

        return diff_min_trend, diff_max_trend

    def _localize_extremes(self, diff_min_extr: list[int], diff_max_extr: list[int]):
        _global_min = len(diff_min_extr)
        _global_max = len(diff_max_extr)

        _extr_min_index = [0] * len(diff_min_extr)
        _extr_max_index = [0] * len(diff_max_extr)

        _k_min = 0
        _k_max = 0
        for i, (diff_min, diff_max) in enumerate(zip(diff_min_extr, diff_max_extr)):
            is_min_extr = diff_min > self.current.extr_eps_min or diff_min == _global_min
            if is_min_extr:
                _extr_min_index[_k_min] = i
                _k_min += 1

            is_max_extr = diff_max > self.current.extr_eps_max or diff_max == _global_max
            if is_max_extr:
                _extr_max_index[_k_max] = i
                _k_max += 1

        self.current.min.temp_extr = ExtendedList(_extr_min_index[:_k_min])
        self.current.max.temp_extr = ExtendedList(_extr_max_index[:_k_max])

    def _localize_trends(self, diff_min_extr: list[int], diff_max_extr: list[int]):
        _global_min = len(diff_min_extr)
        _global_max = len(diff_max_extr)

        _extr_min_index = [0] * len(diff_min_extr)
        _extr_max_index = [0] * len(diff_max_extr)

        _k_min = 0
        _k_max = 0
        for i, (diff_min, diff_max) in enumerate(zip_longest(diff_min_extr, diff_max_extr)):
            if diff_min is not None:
                is_min_extr = diff_min > self.current.trend_eps_min or diff_min == _global_min
                if is_min_extr:
                    _extr_min_index[_k_min] = i
                    _k_min += 1

            if diff_max is not None:
                is_max_extr = diff_max > self.current.trend_eps_max or diff_max == _global_max
                if is_max_extr:
                    _extr_max_index[_k_max] = i
                    _k_max += 1

        self.current.min.temp_trend = ExtendedList(_extr_min_index[:_k_min])
        self.current.max.temp_trend = ExtendedList(_extr_max_index[:_k_max])

    def _select_eps_min(self, marker_diff_min: list[int], coincident: int, eps: int):
        count_zero = 0
        last_non_zero_index = eps
        for i in range(last_non_zero_index + 1, len(marker_diff_min)):
            if count_zero >= coincident - 1:
                self.current.extr_eps_min = last_non_zero_index + coincident - 1
                return
            if marker_diff_min[i] == 0:
                count_zero += 1
            else:
                count_zero = 0
                last_non_zero_index = i

        self.current.extr_eps_min = last_non_zero_index + coincident - 1

    def _select_eps_max(self, marker_diff_max: list[int], coincident: int, eps: int):
        count_zero = 0
        last_non_zero_index = eps
        for i in range(last_non_zero_index + 1, len(marker_diff_max)):
            if count_zero >= coincident - 1:
                self.current.extr_eps_max = last_non_zero_index + coincident - 1
                return
            if marker_diff_max[i] == 0:
                count_zero += 1
            else:
                count_zero = 0
                last_non_zero_index = i

        self.current.extr_eps_max = last_non_zero_index + coincident - 1

    def __set_extremes_data(self):
        self.current.min.extr_indexes = self.current.all.extr_indexes[self.current.min.temp_extr]
        self.current.max.extr_indexes = self.current.all.extr_indexes[self.current.max.temp_extr]
        self.current.all.extr_indexes = ExtendedList(
            sorted(self.current.min.extr_indexes + self.current.max.extr_indexes)
        )

        self.current.min.extr_values = self.get_values()[self.current.min.extr_indexes]
        self.current.max.extr_values = self.get_values()[self.current.max.extr_indexes]
        self.current.all.extr_values = self.get_values()[self.current.all.extr_indexes]

    def __save_extremes_data(self):
        self.save_extremes_eps()
        self.save_extremes_temp()
        self.save_extremes_data()

    def __set_trends_data(self, after_iter: int):
        self.current.min.trend_indexes = self.get_extr_indexes_min(after_iter=after_iter)[self.current.min.temp_trend]
        self.current.max.trend_indexes = self.get_extr_indexes_max(after_iter=after_iter)[self.current.max.temp_trend]
        self.current.all.trend_indexes = ExtendedList(
            sorted(self.current.min.trend_indexes + self.current.max.trend_indexes)
        )

        self.current.min.trend_values = self.get_values()[self.current.min.trend_indexes]
        self.current.max.trend_values = self.get_values()[self.current.max.trend_indexes]
        self.current.all.trend_values = self.get_values()[self.current.all.trend_indexes]

    def __save_trends_data(self, after_iter: int):
        self.save_trends_eps(after_iter=after_iter)
        self.save_trends_temp(after_iter=after_iter)
        self.save_trends_data(after_iter=after_iter)


class ExtremesStorageBase:

    def __init__(self, batch: int):
        self.__storage: ExtendedList[Extremes] = ExtendedList()
        self.__batch: int = batch
        self.__iter: int = 0

    def __len__(self):
        return len(self.__storage)

    def __iter__(self):
        return iter(self.__storage)

    def __next__(self):
        if self.__iter >= len(self.__storage):
            raise StopIteration
        __data = self.__storage[self.__iter]
        self.__iter += 1
        return __data

    def __getitem__(
            self, item: int | slice | None
    ) -> Extremes | list[Extremes]:
        return self.__storage[item]

    def __setitem__(
            self, item: int, value: Extremes
    ) -> None:
        self.__storage[item] = value

    def __repr__(self):
        return "\n".join(f"=============={data.get_sub_interval():^10}==============\n{data!r}" for data in self)

    def build(self, values: np.ndarray[float], split: int):

        Extremes.set_values(values=values)
        for begin in range(0, split, self.__batch):
            end = min(begin + self.__batch, split)
            container = Extremes(
                values=values[begin: end],
                indexes=list(range(begin, end)),
                sub_interval=begin // self.__batch
            )

            self.__storage.append(container)

    def append(self, values: float | Iterable):
        if not isinstance(values, Iterable):
            values = [values]

        for value in values:
            Extremes.append(values=value)

            begin = self.__batch * (len(self) - 1)
            end = len(Extremes.get_values())
            for _begin in range(begin, end, self.__batch):
                _end = min(_begin + self.__batch, end)

                if end != _end:
                    continue

                _values = Extremes.get_values()[_begin: _end]
                container = Extremes(
                    values=_values,
                    indexes=list(range(_begin, _end)),
                    sub_interval=_begin // self.__batch,
                )

                if len(_values) == 1:
                    self.__storage.append(container)
                else:
                    self[-1] = container

                # print(
                #     value,
                #     Extremes.get_values()[_begin: _begin + self.__batch],
                #     len(Extremes.get_values()[_begin: _begin + self.__batch]) == 1,
                # )

    def get_current_iter(
            self,
            item: int | slice | None = None,
    ):
        return self.__prepare_getter_data(
            callable_name="get_current_iter",
            item=item,
        )

    def get_extr_indexes_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_indexes_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_values_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_values_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_indexes_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_indexes_min",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_values_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_values_min",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_indexes_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_indexes_max",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_values_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_values_max",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_indexes_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_trend_indexes_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_values_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_trend_values_combined",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_indexes_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_trend_indexes_min",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_values_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_trend_values_min",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_indexes_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_trend_indexes_max",
            after_iter=after_iter,
            item=item,
        )

    def get_trend_values_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_trend_values_max",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_eps_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_eps_min",
            after_iter=after_iter,
            item=item,
        )

    def get_extr_eps_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        return self.__prepare_getter_data(
            callable_name="get_extr_eps_max",
            after_iter=after_iter,
            item=item,
        )

    def __prepare_getter_data(
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


class ExtremesStorage(ExtremesStorageBase):

    def __init__(self, batch: int):
        super().__init__(batch=batch)

    def search_extremes(
            self,
            coincident: int,
            eps_min: int,
            eps_max: int,
            item: int | slice | None = None,
    ) -> None:
        __data = self.__prepare_storage_data(item=item)
        for data in __data:
            data.search_extremes(
                coincident=coincident,
                eps_min=eps_min,
                eps_max=eps_max,
            )

    def search_trends(
            self,
            eps_for_min: int,
            eps_for_max: int,
            after_iter: int,
            item: int | slice | None = None,
    ) -> None:
        __data = self.__prepare_storage_data(item=item)

        for data in __data:
            data.search_trends(
                eps_for_min=eps_for_min,
                eps_for_max=eps_for_max,
                after_iter=after_iter,
            )

    def filter_extremes(
            self,
            after_iter: int,
            item: int | slice | None = None,
    ) -> None:
        __data = self.__prepare_storage_data(item=item)

        for data in __data:
            mask_min = self._filter_min_extremes(
                sub_interval=data.get_sub_interval(),
                after_iter=after_iter,
            )

            mask_max = self._filter_max_extremes(
                sub_interval=data.get_sub_interval(),
                after_iter=after_iter,
            )

            data.current.min.extr_indexes = data.get_extr_indexes_min(after_iter=after_iter)[mask_min]
            data.current.max.extr_indexes = data.get_extr_indexes_max(after_iter=after_iter)[mask_max]

            data.current.min.extr_values = data.get_extr_values_min(after_iter=after_iter)[mask_min]
            data.current.max.extr_values = data.get_extr_values_max(after_iter=after_iter)[mask_max]

            data.current.min.temp_extr = data.get_extr_indexes_min_temp(after_iter=after_iter)[mask_min]
            data.current.max.temp_extr = data.get_extr_indexes_max_temp(after_iter=after_iter)[mask_max]

            data.current.all.extr_indexes = ExtendedList(
                sorted(data.current.min.extr_indexes + data.current.max.extr_indexes)
            )
            data.current.all.extr_values = data.get_values()[data.current.all.extr_indexes]

            data.save_extremes_temp(after_iter=after_iter)
            data.save_extremes_data(after_iter=after_iter)

    def filter_trends(
            self,
            after_iter: int,
            item: int | slice | None = None,
    ) -> None:
        __data = self.__prepare_storage_data(item=item)

        for data in __data:
            mask_min = self._filter_min_trends(
                sub_interval=data.get_sub_interval(),
                after_iter=after_iter,
            )
            mask_max = self._filter_max_trends(
                sub_interval=data.get_sub_interval(),
                after_iter=after_iter,
            )

            data.current.min.trend_indexes = data.get_trend_indexes_min(after_iter=after_iter)[mask_min]
            data.current.max.trend_indexes = data.get_trend_indexes_max(after_iter=after_iter)[mask_max]

            data.current.min.trend_values = data.get_trend_values_min(after_iter=after_iter)[mask_min]
            data.current.max.trend_values = data.get_trend_values_max(after_iter=after_iter)[mask_max]

            data.current.min.temp_trend = data.get_trend_indexes_min_temp(after_iter=after_iter)[mask_min]
            data.current.max.temp_trend = data.get_trend_indexes_max_temp(after_iter=after_iter)[mask_max]

            data.current.all.trend_indexes = ExtendedList(
                sorted(data.current.min.trend_indexes + data.current.max.trend_indexes)
            )
            data.current.all.trend_values = data.get_values()[data.current.all.trend_indexes]

            data.save_trends_temp(after_iter=after_iter)
            data.save_trends_data(after_iter=after_iter)

    def _filter_min_extremes(self, sub_interval: int, after_iter: int):
        _left = True
        _right = True

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1

        eps_min = self[sub_interval].get_extr_eps_min(after_iter=after_iter)
        num_combined = len(self[sub_interval].get_extr_values_combined(after_iter=after_iter - 1))

        _mask = []
        _temp_i = self[sub_interval].get_extr_indexes_min_temp(after_iter=after_iter)
        _min_v = self[sub_interval].get_extr_values_min(after_iter=after_iter)

        assert len(_temp_i) == len(_min_v), print(f"{sub_interval=}, {after_iter=} {_temp_i=} {_min_v=}")

        for extr_index, extr_value in zip(_temp_i, _min_v):
            max_check_previous = eps_min - extr_index
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self[previous_sub_interval].get_extr_values_combined(
                        after_iter=after_iter - 1,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value <= extr_value:
                            _left = False

            max_check_next = (extr_index + eps_min) - num_combined
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self[next_sub_interval].get_extr_values_combined(
                        after_iter=after_iter - 1,
                    )
                    next_sub_interval += 1
                    for value in next_extr_combined:
                        if num_check > max_check_next:
                            break

                        if value < extr_value:
                            _right = False

                        num_check += 1

            _mask.append(_left and _right)

        return _mask

    def _filter_max_extremes(self, sub_interval: int, after_iter: int):
        _left = True
        _right = True

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1

        eps_max = self[sub_interval].get_extr_eps_max(after_iter=after_iter)
        num_combined = len(self[sub_interval].get_extr_values_combined(after_iter=after_iter - 1))

        _mask = []
        _temp_i = self[sub_interval].get_extr_indexes_max_temp(after_iter=after_iter)
        _max_v = self[sub_interval].get_extr_values_max(after_iter=after_iter)

        assert len(_temp_i) == len(_max_v), print(f"{sub_interval=}, {after_iter=} {_temp_i=} {_max_v=}")

        for extr_index, extr_value in zip(_temp_i, _max_v):
            max_check_previous = eps_max - extr_index
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self[previous_sub_interval].get_extr_values_combined(
                        after_iter=after_iter - 1,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value > extr_value:
                            _left = False

            max_check_next = (extr_index + eps_max) - num_combined
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self[next_sub_interval].get_extr_values_combined(
                        after_iter=after_iter - 1,
                    )
                    next_sub_interval += 1
                    for value in next_extr_combined:
                        if num_check > max_check_next:
                            break
                        if value >= extr_value:
                            _right = False

                        num_check += 1

            _mask.append(_left and _right)

        return _mask

    def _filter_min_trends(self, sub_interval: int, after_iter: int):
        _left = True
        _right = True

        data = self[sub_interval]

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1

        eps_min = data.get_trend_eps_min(after_iter=after_iter)
        num_min = len(data.get_extr_values_min(after_iter=after_iter))

        _mask = []
        _temp_i = data.get_trend_indexes_min_temp(after_iter=after_iter)
        _min_v = data.get_trend_values_min(after_iter=after_iter)

        assert len(_temp_i) == len(_min_v), print(f"{sub_interval=}, {after_iter=} {_temp_i=} {_min_v=}")

        for extr_index, _extr_value in zip(_temp_i, _min_v):
            max_check_previous = eps_min - extr_index
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self[previous_sub_interval].get_extr_values_min(
                        after_iter=after_iter,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value <= _extr_value:
                            _left = False

            max_check_next = (extr_index + eps_min) - num_min
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self[next_sub_interval].get_extr_values_min(
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

    def _filter_max_trends(self, sub_interval: int, after_iter: int):
        _left = True
        _right = True

        data = self[sub_interval]

        next_sub_interval = sub_interval + 1
        previous_sub_interval = sub_interval - 1

        eps_max = data.get_trend_eps_max(after_iter=after_iter)
        num_max = len(data.get_extr_values_max(after_iter=after_iter))

        _mask = []
        _temp_i = data.get_trend_indexes_max_temp(after_iter=after_iter)
        _max_v = data.get_trend_values_max(after_iter=after_iter)

        assert len(_temp_i) == len(_max_v), print(f"{sub_interval=}, {after_iter=} {_temp_i=} {_max_v=}")

        for extr_index, _extr_value in zip(_temp_i, _max_v):
            max_check_previous = eps_max - extr_index
            if max_check_previous > 0:
                num_check = 0
                while previous_sub_interval >= 0 and num_check <= max_check_previous:
                    previous_extr_combined = self[previous_sub_interval].get_extr_values_max(
                        after_iter=after_iter,
                    )
                    previous_sub_interval -= 1
                    for value in previous_extr_combined[::-1]:
                        num_check += 1
                        if num_check > max_check_previous:
                            break

                        if value > _extr_value:
                            _left = False

            max_check_next = (extr_index + eps_max) - num_max
            if max_check_next >= 0:
                num_check = 0
                while next_sub_interval < len(self) and num_check <= max_check_next:
                    next_extr_combined = self[next_sub_interval].get_extr_values_max(
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

    def __prepare_storage_data(self, item: int | slice | None):
        return [self[item]] if isinstance(item, int) else self[item]


def main_extremes():
    # sourcery skip: extract-duplicate-method

    values = [13, 48, 36, 26, 11, 14, 24, 11, 13, 36, 41, 26, 21]
    indexes = list(range(len(values)))

    Extremes.set_values(values)
    a = Extremes(
        values=values,
        indexes=indexes,
    )
    a.search_extremes(eps_min=2, eps_max=2, coincident=1)
    a.search_trends(eps_for_min=2, eps_for_max=2, after_iter=1)

    print(a.get_extr_indexes_combined(after_iter=1))
    print(a.get_extr_values_combined(after_iter=1))

    print(a.get_extr_indexes_min(after_iter=1))
    print(a.get_extr_values_min(after_iter=1))

    print(a.get_extr_indexes_max(after_iter=1))
    print(a.get_extr_values_max(after_iter=1))
    print()
    print(a.get_trend_indexes_combined(after_iter=1))
    print(a.get_trend_values_combined(after_iter=1))

    print(a.get_trend_indexes_min(after_iter=1))
    print(a.get_trend_values_min(after_iter=1))

    print(a.get_trend_indexes_max(after_iter=1))
    print(a.get_trend_values_max(after_iter=1))


def main_extremes_storage():
    values = [13, 48, 36, 26, 11, 14, 24, 11, 13, 36, 41, 26, 21]
    print(np.arange(len(values)))
    print(np.array(values))
    print()
    a = ExtremesStorage(batch=3)
    a.build(values=values, split=len(values))
    _extracted_from_main_extremes_storage_8(a, 1)
    _extracted_from_main_extremes_storage_8(a, 2)
    _extracted_from_main_trends_storage_8(a, 1)
    _extracted_from_main_trends_storage_8(a, 2)


def _extracted_from_main_extremes_storage_8(a, after_iter):
    a.search_extremes(coincident=1, eps=2)
    a.filter_extremes(after_iter=after_iter)
    print(a.get_extr_indexes_combined(after_iter=after_iter))
    print(a.get_extr_values_combined(after_iter=after_iter))
    print(a.get_extr_indexes_min(after_iter=after_iter))
    print(a.get_extr_values_min(after_iter=after_iter))
    print(a.get_extr_indexes_max(after_iter=after_iter))
    print(a.get_extr_values_max(after_iter=after_iter))
    print()


def _extracted_from_main_trends_storage_8(a, after_iter):
    a.search_trends(eps_for_min=2, eps_for_max=2, after_iter=after_iter)
    a.filter_trends(after_iter=after_iter)
    print(a.get_trend_indexes_combined(after_iter=after_iter))
    print(a.get_trend_values_combined(after_iter=after_iter))
    print(a.get_trend_indexes_min(after_iter=after_iter))
    print(a.get_trend_values_min(after_iter=after_iter))
    print(a.get_trend_indexes_max(after_iter=after_iter))
    print(a.get_trend_values_max(after_iter=after_iter))
    print()


if __name__ == '__main__':
    main_extremes()
