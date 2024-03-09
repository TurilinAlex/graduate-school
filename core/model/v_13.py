from copy import deepcopy
from dataclasses import dataclass, field, asdict
from itertools import zip_longest
from typing import Sequence

import numpy as np

from core.sort import argsort


class ExtendedList(list):

    def __getitem__(self, item: int | Sequence | slice | None):
        if isinstance(item, int | np.int32):
            return super().__getitem__(item)
        elif isinstance(item, slice):
            return ExtendedList(super().__getitem__(item))
        elif isinstance(item, Sequence) and isinstance(item[0], bool):
            return ExtendedList(item for item, flag in zip(self, item) if flag)
        elif isinstance(item, Sequence) and isinstance(item[0], int):
            return ExtendedList(self[i] for i in item)
        elif item is None:
            return ExtendedList(self)
        else:
            raise TypeError(f"Unsupported index type! {item} {type(item)}")


@dataclass
class AllExtremes:
    extr_values: ExtendedList = field(default=None)
    extr_indexes: ExtendedList = field(default=None)
    trend_values: ExtendedList = field(default=None)
    trend_indexes: ExtendedList = field(default=None)
    begin: int = field(default=None)
    end: int = field(default=None)

    def __repr__(self):
        field_values = asdict(self)
        return "\n".join(f"\t{key:<14}: {value}" for key, value in field_values.items())


@dataclass
class MinExtremes(AllExtremes):
    temp_extr: ExtendedList = field(default=None)
    temp_trend: ExtendedList = field(default=None)

    def __repr__(self):
        field_values = asdict(self)
        return "\n".join(f"\t{key:<14}: {value}" for key, value in field_values.items())


@dataclass
class MaxExtremes(AllExtremes):
    temp_extr: ExtendedList = field(default=None)
    temp_trend: ExtendedList = field(default=None)

    def __repr__(self):
        field_values = asdict(self)
        return "\n".join(f"\t{key:<14}: {value}" for key, value in field_values.items())


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
        return (
            f"=== all ==============================\n{self.all}\n"
            f"=== min ==============================\n{self.min}\n"
            f"=== max ==============================\n{self.max}\n"
            f"--------------------------------------\n"
            f"{'extr_eps_min':<18}: {self.extr_eps_min}\n"
            f"{'extr_eps_max':<18}: {self.extr_eps_max}\n"
            f"{'trend_eps_min':<18}: {self.trend_eps_min}\n"
            f"{'trend_eps_max':<18}: {self.trend_eps_max}\n"
        )


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
        self.__history_data[after_iter].all.begin = self.__current_data.all.begin
        self.__history_data[after_iter].all.end = self.__current_data.all.end

        self.__history_data[after_iter].min.extr_indexes = deepcopy(self.__current_data.min.extr_indexes)
        self.__history_data[after_iter].min.extr_values = deepcopy(self.__current_data.min.extr_values)
        self.__history_data[after_iter].min.begin = self.__current_data.min.begin
        self.__history_data[after_iter].min.end = self.__current_data.min.end

        self.__history_data[after_iter].max.extr_indexes = deepcopy(self.__current_data.max.extr_indexes)
        self.__history_data[after_iter].max.extr_values = deepcopy(self.__current_data.max.extr_values)
        self.__history_data[after_iter].max.begin = self.__current_data.max.begin
        self.__history_data[after_iter].max.end = self.__current_data.max.end

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

    # region Interval getter

    def get_extr_begin_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.all.begin

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].all.begin

    def get_extr_begin_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.begin

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.begin

    def get_extr_begin_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.begin

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.begin

    def get_extr_end_combined(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.all.end

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].all.end

    def get_extr_end_min(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.min.end

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].min.end

    def get_extr_end_max(self, after_iter: int | None = None):
        if after_iter is None:
            return self.__current_data.max.end

        after_iter = self._validate_after_iter(after_iter=after_iter)
        return self.__history_data[after_iter].max.end

    # endregion Interval getter

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
            begin: int | None = None,
            end: int | None = None,
    ):
        super().__init__()

        self.current.all.extr_values = ExtendedList(values)
        self.current.all.extr_indexes = ExtendedList(indexes)
        self.current.all.begin = begin
        self.current.all.end = end

        self.current.min.extr_values = ExtendedList(values)
        self.current.min.extr_indexes = ExtendedList(indexes)
        self.current.min.begin = begin
        self.current.min.end = end

        self.current.max.extr_values = ExtendedList(values)
        self.current.max.extr_indexes = ExtendedList(indexes)
        self.current.max.begin = begin
        self.current.max.end = end

        self._sub_interval: int | None = sub_interval

        self.save_extremes_data()

    def get_sub_interval(self):
        return self._sub_interval

    def search_extremes(
            self,
            eps: int,
            coincident: int,
            start_all_index: int = None,
            start_min_index: int = None,
            start_max_index: int = None,
    ) -> None:
        self.add_extremes_iter()

        _marker_min, _marker_max, _diff_min_extr, _diff_max_extr = (
            self._diff_between_indexes_for_extremes(
                eps=eps,
            )
        )

        self._select_eps_min(
            marker_diff_min=_marker_min,
            coincident=coincident,
            eps=eps,
        )
        self._select_eps_max(
            marker_diff_max=_marker_max,
            coincident=coincident,
            eps=eps,
        )
        self._localize_extremes(
            diff_min_extr=_diff_min_extr,
            diff_max_extr=_diff_max_extr,
        )

        self.__set_extremes_data()
        self.__update_interval_values(
            start_all_index=start_all_index,
            start_min_index=start_min_index,
            start_max_index=start_max_index,
        )
        self.__save_extremes_data()

    def __update_interval_values(
            self,
            start_all_index: int = None,
            start_min_index: int = None,
            start_max_index: int = None,
    ):
        if start_all_index is not None:
            self.current.all.begin = start_all_index
        elif self.current.all.begin is None:
            self.current.all.begin = 0
        self.current.all.end = self.current.all.begin + len(self.current.all.extr_indexes)

        if start_min_index is not None:
            self.current.min.begin = start_min_index
        elif self.current.min.begin is None:
            self.current.min.begin = 0
        self.current.min.end = self.current.min.begin + len(self.current.min.extr_indexes)

        if start_max_index is not None:
            self.current.max.begin = start_max_index
        elif self.current.max.begin is None:
            self.current.max.begin = 0
        self.current.max.end = self.current.max.begin + len(self.current.max.extr_indexes)

    def search_trends(
            self,
            eps_for_min: int,
            eps_for_max: int,
    ):

        self.current.trend_eps_min = eps_for_min
        self.current.trend_eps_max = eps_for_max

        _diff_min_extr, _diff_max_extr = (
            self._diff_between_indexes_for_trends(
                eps_for_min=eps_for_min,
                eps_for_max=eps_for_max,
            )
        )

        self._localize_trends(
            diff_min_extr=_diff_min_extr,
            diff_max_extr=_diff_max_extr,
        )

        self.__set_trends_data()
        self.__save_trends_data()

    def _diff_between_indexes_for_extremes(self, eps: int = 1):

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
                if min_diff_for_minimal <= eps:
                    break

            diff_min_extr[_indexes[i]] = min_diff_for_minimal
            marker_diff_for_minimal[min_diff_for_minimal] = 1

            # max
            for j in range(1, (n - i)):
                diff = abs(_indexes[i] - _indexes[i + j])
                if diff < min_diff_for_maximal:
                    min_diff_for_maximal = diff
                if min_diff_for_maximal <= eps:
                    break

            diff_max_extr[_indexes[i]] = min_diff_for_maximal
            marker_diff_for_maximal[min_diff_for_maximal] = 1

        # endregion Calculating the difference between indexes

        return marker_diff_for_minimal, marker_diff_for_maximal, diff_min_extr, diff_max_extr

    def _diff_between_indexes_for_trends(self, eps_for_min: int, eps_for_max: int):
        # sourcery skip: use-assigned-variable

        _indexes_min = argsort(self.current.min.extr_values)
        _indexes_max = argsort(self.current.max.extr_values)

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

        self.current.min.temp_extr = _extr_min_index[:_k_min]
        self.current.max.temp_extr = _extr_max_index[:_k_max]

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

        self.current.min.temp_trend = _extr_min_index[:_k_min]
        self.current.max.temp_trend = _extr_max_index[:_k_max]

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

    def __set_trends_data(self):
        self.current.min.trend_indexes = self.current.min.extr_indexes[self.current.min.temp_trend]
        self.current.max.trend_indexes = self.current.max.extr_indexes[self.current.max.temp_trend]
        self.current.all.trend_indexes = ExtendedList(
            sorted(self.current.min.trend_indexes + self.current.max.trend_indexes)
        )

        self.current.min.trend_values = self.get_values()[self.current.min.trend_indexes]
        self.current.max.trend_values = self.get_values()[self.current.max.trend_indexes]
        self.current.all.trend_values = self.get_values()[self.current.all.trend_indexes]

    def __save_extremes_data(self):
        self.save_extremes_eps()
        self.save_extremes_temp()
        self.save_extremes_data()

    def __save_trends_data(self):
        self.save_trends_eps()
        self.save_trends_temp()
        self.save_trends_data()


class ExtremesStorageBase:

    def __init__(self, batch: int):
        self.__storage: ExtendedList[Extremes] = ExtendedList()
        self.__iter: int = 0
        self.__batch: int = batch

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

    def __repr__(self):
        return "\n".join(f"=============={data.get_sub_interval():^10}==============\n{data!r}" for data in self)

    def build(self, values: np.ndarray[float], split: int):

        Extremes.set_values(values=values)
        for begin in range(0, split, self.__batch):
            end = min(begin + self.__batch, split)
            container = Extremes(
                values=values[begin: end],
                indexes=list(range(begin, end)),
                begin=begin,
                end=end,
                sub_interval=begin // self.__batch
            )

            self.__storage.append(container)

    def search_extremes(
            self,
            coincident: int,
            eps: int,
            item: int | slice | None = None,
    ) -> None:

        __data = self.__prepare_storage_data(item=item)

        _start_all_index = __data[0].get_extr_begin_combined()
        _start_min_index = __data[0].get_extr_begin_min()
        _start_max_index = __data[0].get_extr_begin_max()

        for data in __data:
            data.search_extremes(
                coincident=coincident,
                eps=eps,
                start_all_index=_start_all_index,
                start_min_index=_start_min_index,
                start_max_index=_start_max_index,
            )
            _start_all_index = data.get_extr_end_combined()
            _start_min_index = data.get_extr_end_min()
            _start_max_index = data.get_extr_end_max()

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
    a.search_extremes(eps=2, coincident=1)
    a.search_trends(eps_for_min=2, eps_for_max=2)
    print(a.print_current())
    print(a.print_history(after_iter=0))
    print(a.print_history(after_iter=1))

    print()

    a.search_extremes(eps=2, coincident=1)
    a.search_trends(eps_for_min=2, eps_for_max=2)
    print(a.print_current())
    print(a.print_history(after_iter=0))
    print(a.print_history(after_iter=1))
    print(a.print_history(after_iter=2))

    print()

    a.search_extremes(eps=2, coincident=1)
    a.search_trends(eps_for_min=2, eps_for_max=2)
    print(a.print_current())
    print(a.print_history(after_iter=0))
    print(a.print_history(after_iter=1))
    print(a.print_history(after_iter=2))
    print(a.print_history(after_iter=3))

    print(a.get_extr_indexes_combined(after_iter=0))
    print(a.get_extr_values_combined(after_iter=0))
    print(a.get_extr_indexes_combined(after_iter=1))
    print(a.get_extr_values_combined(after_iter=1))
    print(a.get_extr_indexes_combined(after_iter=2))
    print(a.get_extr_values_combined(after_iter=2))
    print(a.get_extr_indexes_combined(after_iter=3))
    print(a.get_extr_values_combined(after_iter=3))


def main_extremes_storage():
    values = [13, 48, 36, 26, 11, 14, 24, 11, 13, 36, 41, 26, 21]

    a = ExtremesStorageBase(batch=3)
    a.build(values=values, split=len(values))
    a.search_extremes(coincident=1, eps=2)
    print(a)


if __name__ == '__main__':
    main_extremes_storage()
