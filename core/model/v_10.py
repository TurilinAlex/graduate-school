from dataclasses import dataclass
from operator import le, lt, gt, ge
from typing import Callable

import numpy as np

from core.matches_extremum import MatchesOnInputArray
from core.sort import argsort
from core.trend.combined_trend_point import CombinedTrendDetection


@dataclass
class TrendData:
    extr_values: np.ndarray[np.float32]
    extr_indexes: np.ndarray[np.int32]

    trend_values: np.ndarray[np.float32]
    trend_indexes: np.ndarray[np.int32]

    begin: int
    end: int


class ExtremesContainer:

    def __init__(
            self,
            values: np.ndarray[np.float32],
            indexes: np.ndarray[np.int32],
            begin: int,
            end: int,
            sub_interval: int,
    ):
        self.is_updated_extr: bool = False
        self.is_updated_trend: bool = False

        self.all: TrendData = TrendData(
            extr_values=values,
            extr_indexes=indexes,
            trend_values=values,
            trend_indexes=indexes,
            begin=begin,
            end=end,
        )

        self.min: TrendData = TrendData(
            extr_values=values,
            extr_indexes=indexes,
            trend_values=values,
            trend_indexes=indexes,
            begin=begin,
            end=end,
        )

        self.max: TrendData = TrendData(
            extr_values=values,
            extr_indexes=indexes,
            trend_values=values,
            trend_indexes=indexes,
            begin=begin,
            end=end,
        )

        self._current_iter = 0
        self._sub_interval = sub_interval

        self._diff_min_extr: np.ndarray[int] | None = None
        self._diff_max_extr: np.ndarray[int] | None = None

        self._diff_min_trend: np.ndarray[int] | None = None
        self._diff_max_trend: np.ndarray[int] | None = None

        self._extr_all_index: np.ndarray[int] | None = None
        self._extr_min_index: np.ndarray[int] | None = None
        self._extr_max_index: np.ndarray[int] | None = None

        self._trend_all_index: np.ndarray[int] | None = None
        self._trend_min_index: np.ndarray[int] | None = None
        self._trend_max_index: np.ndarray[int] | None = None

        self._eps_min_extr: int | None = None
        self._eps_max_extr: int | None = None

        self._eps_min_trend: int | None = None
        self._eps_max_trend: int | None = None

    def __repr__(self):
        return (f"Sub: {self._sub_interval}..........\n"
                f"All: {self.all}\n"
                f"Min: {self.min}\n"
                f"Max: {self.max}\n")

    def search_extremes(
            self,
            eps: int,
            coincident: int,
            border_check: Callable = None,
            is_update_extr_data: bool = True,
    ) -> None:

        self.is_updated_extr = True

        _marker_min, _marker_max = self._diff_between_sort_indexes(eps=eps)
        _eps_min = self._select_eps(
            marker_diff=_marker_min,
            coincident=coincident,
            eps=eps,
        )
        _eps_max = self._select_eps(
            marker_diff=_marker_max,
            coincident=coincident,
            eps=eps,
        )

        self._eps_min_extr = _eps_min
        self._eps_max_extr = _eps_max

        self._extract_all_extremes_from_diff(border_check=border_check)

        _extr_all_index = self.all.extr_indexes[self._extr_all_index]
        _extr_min_index = self.all.extr_indexes[self._extr_min_index]
        _extr_max_index = self.all.extr_indexes[self._extr_max_index]

        self.all.extr_indexes = _extr_all_index
        self.min.extr_indexes = _extr_min_index
        self.max.extr_indexes = _extr_max_index

        if is_update_extr_data:
            self.update_extr_data()

        self._current_iter += 1

    def search_trends(
            self,
            eps_for_min: int,
            eps_for_max: int,
            border_check_min: Callable = None,
            border_check_max: Callable = None,
            is_update_trend_data: bool = True,
    ):
        self.is_updated_trend = True

        self._eps_min_trend = eps_for_min
        self._eps_max_trend = eps_for_max

        self._diff_between_sort_indexes_trend(eps_for_min=eps_for_min, eps_for_max=eps_for_max)
        self._filter_extremes_trend(border_check_min=border_check_min, border_check_max=border_check_max)

        _extr_min_index_trend = self.min.extr_indexes[self._trend_min_index]
        _extr_max_index_trend = self.max.extr_indexes[self._trend_max_index]
        _extr_all_index_trend = np.sort(np.hstack([_extr_min_index_trend, _extr_max_index_trend]))
        self._trend_all_index = np.argsort(_extr_all_index_trend)

        self.all.trend_indexes = _extr_all_index_trend
        self.min.trend_indexes = _extr_min_index_trend
        self.max.trend_indexes = _extr_max_index_trend

        if is_update_trend_data:
            self.update_trend_data()

    def update_extr_data(self):
        if self.is_updated_extr:
            self.min.extr_values = self.all.extr_values[self._extr_min_index]
            self.max.extr_values = self.all.extr_values[self._extr_max_index]
            self.all.extr_values = self.all.extr_values[self._extr_all_index]
        self.is_updated_extr = False

    def update_trend_data(self):
        if self.is_updated_trend:
            self.min.trend_values = self.min.extr_values[self._trend_min_index]
            self.max.trend_values = self.max.extr_values[self._trend_max_index]
            self.all.trend_values = self.all.extr_values[self._trend_all_index]
        self.is_updated_trend = False

    def _diff_between_sort_indexes(self, eps: int = 1) -> tuple[np.ndarray, np.ndarray]:

        _indexes = argsort(self.all.extr_values)

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

        _indexes_min = argsort(self.min.extr_values)
        _indexes_max = argsort(self.max.extr_values)

        n_min = len(_indexes_min)
        n_max = len(_indexes_max)

        self._diff_min_trend = np.empty_like(_indexes_min, dtype=np.uint32)
        self._diff_max_trend = np.empty_like(_indexes_max, dtype=np.uint32)

        # region Calculating the difference between indexes

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

        # endregion Calculating the difference between indexes

    def _extract_all_extremes_from_diff(self, border_check: Callable):
        print(f"--------{self._sub_interval}--------")
        _offset = self.all.begin
        _batch = len(self._diff_min_extr)
        self._extr_all_index = np.empty_like(self._diff_min_extr)
        self._extr_min_index = np.empty_like(self._diff_min_extr)
        self._extr_max_index = np.empty_like(self._diff_max_extr)

        _k_all = 0
        _k_min = 0
        _k_max = 0
        for i in range(_batch):
            is_min_extr = self._diff_min_extr[i] > self._eps_min_extr or self._diff_min_extr[i] == _batch
            if is_min_extr:
                is_add_index = border_check(
                    left=le,
                    right=lt,
                    after_iter=self._current_iter,
                    extremes_index=i,
                    offset=_offset,
                    batch=_batch,
                    eps=self._eps_min_extr,
                ) if callable(border_check) else True

                if is_add_index:
                    print(f"Is min: {i}, is True min")
                    self._extr_all_index[_k_all] = i
                    self._extr_min_index[_k_min] = i
                    _k_all += 1
                    _k_min += 1
                    continue
                print(f"Is min: {i}, is False min")

            is_max_extr = self._diff_max_extr[i] > self._eps_max_extr or self._diff_max_extr[i] == _batch
            if is_max_extr:
                is_add_index = border_check(
                    left=gt,
                    right=ge,
                    after_iter=self._current_iter,
                    extremes_index=i,
                    offset=_offset,
                    batch=_batch,
                    eps=self._eps_max_extr,
                ) if callable(border_check) else True

                if is_add_index:
                    print(f"Is max: {i}, is True max")
                    self._extr_all_index[_k_all] = i
                    self._extr_max_index[_k_max] = i
                    _k_all += 1
                    _k_max += 1
                    continue
                print(f"Is max: {i}, is False max")
                continue

            print(f"Is not extr: {i}")

        self._extr_all_index = self._extr_all_index[:_k_all]
        self._extr_min_index = self._extr_min_index[:_k_min]
        self._extr_max_index = self._extr_max_index[:_k_max]

    # def _remove_disputed_extremes_border(self, border_check: Callable):
    #     _offset = self.all.begin
    #
    #     _batch = len(self._extr_min_index)
    #     for extr in range(_batch):
    #         extr = self._extr_min_index[extr]
    #         is_extr = border_check(
    #             left=gt,
    #             right=ge,
    #             after_iter=self._current_iter,
    #             extremes_index=extr,
    #             offset=_offset,
    #             batch=_batch,
    #             eps=self._eps_max_extr,
    #         )
    #
    #         if not is_extr:
    #             index_all, = np.where(self._extr_all_index == extr)
    #             self._extr_all_index = np.delete(self._extr_all_index, index_all)
    #
    #             index_min, = np.where(self._extr_min_index == extr)
    #             self._extr_min_index = np.delete(self._extr_min_index, index_min)
    #
    #     _batch = len(self._extr_max_index)
    #     for extr in range(_batch):
    #         extr = self._extr_max_index[extr]
    #         is_extr = border_check(
    #             left=gt,
    #             right=ge,
    #             after_iter=self._current_iter,
    #             extremes_index=extr,
    #             offset=_offset,
    #             batch=_batch,
    #             eps=self._eps_min_extr,
    #         )
    #
    #         if not is_extr:
    #             index_all, = np.where(self._extr_all_index == extr)
    #             self._extr_all_index = np.delete(self._extr_all_index, index_all)
    #
    #             index_max, = np.where(self._extr_max_index == extr)
    #             self._extr_max_index = np.delete(self._extr_max_index, index_max)

    def _filter_extremes_trend(self, border_check_min: Callable, border_check_max: Callable):
        _offset_min = self.min.begin
        _offset_max = self.max.begin
        _batch_min = len(self._diff_min_trend)
        _batch_max = len(self._diff_max_trend)

        self._trend_min_index = np.empty_like(self._diff_min_trend)
        self._trend_max_index = np.empty_like(self._diff_max_trend)

        _k_min = 0
        _k_max = 0
        for i in range(_batch_min):
            is_min_extr = self._diff_min_trend[i] > self._eps_min_trend or self._diff_min_trend[i] == _batch_min
            if is_min_extr:
                is_add_index = border_check_min(
                    left=le,
                    right=lt,
                    after_iter=self._current_iter,
                    extremes_index=i,
                    offset=_offset_min,
                    batch=_batch_min,
                    eps=self._eps_min_trend,
                ) if callable(border_check_min) else True

                if is_add_index:
                    self._trend_min_index[_k_min] = i
                    _k_min += 1

        for i in range(_batch_max):
            is_max_extr = self._diff_max_trend[i] > self._eps_max_trend or self._diff_max_trend[i] == _batch_max
            if is_max_extr:
                is_add_index = border_check_max(
                    left=gt,
                    right=ge,
                    after_iter=self._current_iter,
                    extremes_index=i,
                    offset=_offset_max,
                    batch=_batch_max,
                    eps=self._eps_max_trend,
                ) if callable(border_check_max) else True

                if is_add_index:
                    self._trend_max_index[_k_max] = i
                    _k_max += 1

        self._trend_min_index = self._trend_min_index[:_k_min]
        self._trend_max_index = self._trend_max_index[:_k_max]

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

    # region Extremum getter
    def get_extr_indexes_min(self) -> np.ndarray[np.uint32]:
        return self.min.extr_indexes

    def get_extr_indexes_max(self) -> np.ndarray[np.uint32]:
        return self.max.extr_indexes

    def get_extr_values_min(self) -> np.ndarray[np.float32]:
        return self.min.extr_values

    def get_extr_values_max(self) -> np.ndarray[np.float32]:
        return self.max.extr_values

    def get_extr_eps_min(self) -> int:
        return self._eps_min_extr

    def get_extr_eps_max(self) -> int:
        return self._eps_max_extr

    def get_extr_indexes_combined(self) -> np.ndarray[np.uint32]:
        return self.all.extr_indexes

    def get_extr_values_combined(self) -> np.ndarray[np.uint32]:
        return self.all.extr_values

    # endregion Extremum getter

    # region Trend getter

    def get_trend_indexes_min(self) -> np.ndarray[np.uint32]:
        return self.min.trend_indexes

    def get_trend_indexes_max(self) -> np.ndarray[np.uint32]:
        return self.max.trend_indexes

    def get_trend_values_min(self) -> np.ndarray[np.float32]:
        return self.min.trend_values

    def get_trend_values_max(self) -> np.ndarray[np.float32]:
        return self.max.trend_values

    def get_trend_eps_min(self) -> int:
        return self._eps_min_trend

    def get_trend_eps_max(self) -> int:
        return self._eps_max_trend

    def get_trend_indexes_combined(self) -> np.ndarray[np.uint32]:
        return self.all.trend_indexes

    def get_trend_values_combined(self) -> np.ndarray[np.float32]:
        return self.all.trend_values

    # endregion Trend getter

    def get_sub_interval(self) -> int:
        return self._sub_interval

    def get_current_iter(self) -> int:
        return self._current_iter


class ExtremeStorage:

    def __init__(self):
        self._storage: list[ExtremesContainer] = []
        self._save_storage: {int, {str, list[TrendData | int]}} = {}
        self._save_storage_current_num: int = 0
        self._current_iter = 0

        self._split_index: int | None = None

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return iter(self._storage)

    def __next__(self):
        if self._current_iter < len(self._storage):
            __data = self._storage[self._current_iter]
            self._current_iter += 1
            return __data
        else:
            raise StopIteration

    def __getitem__(
            self, item: int | slice
    ) -> ExtremesContainer | list[ExtremesContainer]:
        return self._storage[item]

    # region CoreLogic

    def search_extremes(
            self,
            coincident: int,
            eps: int,
            item: int | slice | None = None,
            is_border_check: bool = True,
    ) -> None:

        border_check_all = self.border_check_all if is_border_check else None

        if item is None:
            for data in self._storage:
                data.search_extremes(
                    coincident=coincident,
                    eps=eps,
                    border_check=border_check_all,
                    is_update_extr_data=False,
                )

        if isinstance(item, int):
            self._storage[item].search_extremes(
                coincident=coincident,
                eps=eps,
                border_check=border_check_all,
                is_update_extr_data=False,
            )

        if isinstance(item, slice):
            for data in self._storage[item]:
                data.search_extremes(
                    coincident=coincident,
                    eps=eps,
                    border_check=border_check_all,
                    is_update_extr_data=False,
                )

        _begin_all = self._storage[0].all.begin
        _begin_min = self._storage[0].min.begin
        _begin_max = self._storage[0].max.begin

        for data in self._storage:
            data.update_extr_data()

            data.all.begin = _begin_all
            _end_all = data.all.begin + len(data.all.extr_indexes)
            data.all.end = _end_all
            _begin_all = _end_all

            data.min.begin = _begin_min
            _end_min = data.min.begin + len(data.min.extr_indexes)
            data.min.end = _end_min
            _begin_min = _end_min

            data.max.begin = _begin_max
            _end_max = data.max.begin + len(data.max.extr_indexes)
            data.max.end = _end_max
            _begin_max = _end_max

        self._save_storage_current_num += 1
        self.__add_save_default_storage()
        self.__save_storage_extr()

    def search_trends(
            self,
            eps_for_min: int,
            eps_for_max: int,
            item: int | slice | None = None,
            is_check_border: bool = True,
    ) -> None:

        border_check_min = self.border_check_min if is_check_border else None
        border_check_max = self.border_check_max if is_check_border else None

        if item is None:
            for data in self._storage:
                data.search_trends(
                    eps_for_min=eps_for_min,
                    eps_for_max=eps_for_max,
                    border_check_min=border_check_min,
                    border_check_max=border_check_max,
                    is_update_trend_data=False,
                )

        if isinstance(item, int):
            self._storage[item].search_trends(
                eps_for_min=eps_for_min,
                eps_for_max=eps_for_max,
                border_check_min=border_check_min,
                border_check_max=border_check_max,
                is_update_trend_data=False,
            )

        if isinstance(item, slice):
            for data in self._storage[item]:
                data.search_trends(
                    eps_for_min=eps_for_min,
                    eps_for_max=eps_for_max,
                    border_check_min=border_check_min,
                    border_check_max=border_check_max,
                    is_update_trend_data=False,
                )
        for data in self._storage:
            data.update_trend_data()

        self.__save_storage_trend()

    def border_check_all(
            self,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            after_iter: int,
            extremes_index: int,
            offset: int,
            batch: int,
            eps: int,
    ) -> bool:

        _left = True
        _right = True

        _i_extr, _j_extr = self.unravel_index_all(index=(extremes_index + offset), after_iter=after_iter)
        _extr_value = self._save_storage[after_iter][_i_extr]["all"]["extr_values"][_j_extr]

        if extremes_index - eps < 0:
            for i in range(offset - 1, extremes_index + offset - eps - 1, -1):
                if i < 0:
                    break
                _i, _j = self.unravel_index_all(index=i, after_iter=after_iter)
                if left(
                        self._save_storage[after_iter][_i]["all"]["extr_values"][_j],
                        _extr_value
                ):
                    _left = False
                    break

        if extremes_index + eps >= batch:
            for i in range(offset + batch, extremes_index + offset + eps + 1):
                if i >= self._save_storage[after_iter][-1]["all"]["end"]:
                    break

                _i, _j = self.unravel_index_all(index=i, after_iter=after_iter)
                if right(
                        self._save_storage[after_iter][_i]["all"]["extr_values"][_j],
                        _extr_value
                ):
                    _right = False
                    break

        return _left and _right

    def border_check_min(
            self,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            after_iter: int,
            extremes_index: int,
            offset: int,
            batch: int,
            eps: int,
    ) -> bool:

        _left = True
        _right = True

        _i_extr, _j_extr = self.unravel_index_min(index=(extremes_index + offset), after_iter=after_iter)
        _extr_value = self._save_storage[after_iter][_i_extr]["min"]["extr_values"][_j_extr]

        if extremes_index - eps < 0:
            for i in range(offset - 1, extremes_index + offset - eps - 1, -1):
                if i < 0:
                    break
                _i, _j = self.unravel_index_min(index=i, after_iter=after_iter)
                if left(
                        self._save_storage[after_iter][_i]["min"]["extr_values"][_j],
                        _extr_value
                ):
                    _left = False
                    break

        if extremes_index + eps >= batch:
            for i in range(offset + batch, extremes_index + offset + eps + 1):
                if i >= self._save_storage[after_iter][-1]["min"]["end"]:
                    break

                _i, _j = self.unravel_index_min(index=i, after_iter=after_iter)
                if right(
                        self._save_storage[after_iter][_i]["min"]["extr_values"][_j],
                        _extr_value
                ):
                    _right = False
                    break

        return _left and _right

    def border_check_max(
            self,
            left: Callable[[float, float], bool],
            right: Callable[[float, float], bool],
            after_iter: int,
            extremes_index: int,
            offset: int,
            batch: int,
            eps: int,
    ) -> bool:

        _left = True
        _right = True

        _i_extr, _j_extr = self.unravel_index_max(index=(extremes_index + offset), after_iter=after_iter)
        _extr_value = self._save_storage[after_iter][_i_extr]["max"]["extr_values"][_j_extr]

        if extremes_index - eps < 0:
            for i in range(offset - 1, extremes_index + offset - eps - 1, -1):
                if i < 0:
                    break
                _i, _j = self.unravel_index_max(index=i, after_iter=after_iter)
                if left(
                        self._save_storage[after_iter][_i]["max"]["extr_values"][_j],
                        _extr_value
                ):
                    _left = False
                    break

        if extremes_index + eps >= batch:
            for i in range(offset + batch, extremes_index + offset + eps + 1):
                if i >= self._save_storage[after_iter][-1]["max"]["end"]:
                    break

                _i, _j = self.unravel_index_max(index=i, after_iter=after_iter)
                if right(
                        self._save_storage[after_iter][_i]["max"]["extr_values"][_j],
                        _extr_value
                ):
                    _right = False
                    break

        return _left and _right

    def unravel_index_all(self, index: int, after_iter: int):
        i = 0
        size = 0
        while size < len(self) - 1:
            index -= len(self._save_storage[after_iter][i]["all"]["extr_values"])
            if index < 0:
                break
            i += 1
            size += 1

        j = 0
        if len(self._save_storage[after_iter][i]["all"]["extr_values"]):
            j = index % len(self._save_storage[after_iter][i]["all"]["extr_values"])

        return i, j

    def unravel_index_min(self, index: int, after_iter: int):
        i = 0
        size = 0
        while size < len(self) - 1:
            index -= len(self._save_storage[after_iter][i]["min"]["extr_values"])
            if index < 0:
                break
            i += 1
            size += 1

        j = 0
        if len(self._save_storage[after_iter][i]["min"]["extr_values"]):
            j = index % len(self._save_storage[after_iter][i]["min"]["extr_values"])

        return i, j

    def unravel_index_max(self, index: int, after_iter: int):
        i = 0
        size = 0
        while size < len(self) - 1:
            index -= len(self._save_storage[after_iter][i]["max"]["extr_values"])
            if index < 0:
                break
            i += 1
            size += 1

        j = 0
        if len(self._save_storage[after_iter][i]["max"]["extr_values"]):
            j = index % len(self._save_storage[after_iter][i]["max"]["extr_values"])

        return i, j

    def build(self, values: np.ndarray[float], split: int, batch: int):

        self._split_index = split

        for begin in range(0, split, batch):
            end = min(begin + batch, split)
            container = ExtremesContainer(
                values=values[begin: end],
                indexes=np.arange(begin, end, dtype=int),
                begin=begin,
                end=end,
                sub_interval=begin // batch
            )

            self.__add(container=container)

        self.__add_save_default_storage()
        self.__save_storage_extr()
        self.__save_storage_trend()

    def __add(self, container: ExtremesContainer):
        self._storage.append(container)

    def __add_save_default_storage(self):
        if not self._save_storage_current_num:
            self._save_storage[self._save_storage_current_num] = [
                {
                    "all": {
                        "extr_indexes": None,
                        "extr_values": None,
                        "trend_indexes": None,
                        "trend_values": None,
                        "begin": None,
                        "end": None,
                    },
                    "min": {
                        "extr_indexes": None,
                        "extr_values": None,
                        "trend_indexes": None,
                        "trend_values": None,
                        "begin": None,
                        "end": None,
                    },
                    "max": {
                        "extr_indexes": None,
                        "extr_values": None,
                        "trend_indexes": None,
                        "trend_values": None,
                        "begin": None,
                        "end": None,
                    },
                    "extr_eps_min": None,
                    "extr_eps_max": None,
                    "trend_eps_min": None,
                    "trend_eps_max": None,
                } for _ in range(len(self))
            ]

            return

        self._save_storage[self._save_storage_current_num] = [None for _ in range(len(self))]
        for data in self._storage:
            __data = {
                "all": {
                    "extr_indexes":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["all"]["extr_indexes"],
                    "extr_values":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["all"]["extr_values"],
                    "trend_indexes":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["all"]["trend_indexes"],
                    "trend_values":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["all"]["trend_values"],
                    "begin":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["all"]["begin"],
                    "end":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["all"]["end"],
                },
                "min": {
                    "extr_indexes":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["min"]["extr_indexes"],
                    "extr_values":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["min"]["extr_values"],
                    "trend_indexes":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["min"]["trend_indexes"],
                    "trend_values":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["min"]["trend_values"],
                    "begin":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["min"]["begin"],
                    "end":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["min"]["end"],
                },
                "max": {
                    "extr_indexes":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["max"]["extr_indexes"],
                    "extr_values":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["max"]["extr_values"],
                    "trend_indexes":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["max"]["trend_indexes"],
                    "trend_values":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["max"]["trend_values"],
                    "begin":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["max"]["begin"],
                    "end":
                        self._save_storage[self._save_storage_current_num - 1]
                        [data.get_sub_interval()]["max"]["end"],
                },
                "extr_eps_min":
                    self._save_storage[self._save_storage_current_num - 1]
                    [data.get_sub_interval()]["extr_eps_min"],
                "extr_eps_max":
                    self._save_storage[self._save_storage_current_num - 1]
                    [data.get_sub_interval()]["extr_eps_max"],
                "trend_eps_min":
                    self._save_storage[self._save_storage_current_num - 1]
                    [data.get_sub_interval()]["trend_eps_min"],
                "trend_eps_max":
                    self._save_storage[self._save_storage_current_num - 1]
                    [data.get_sub_interval()]["trend_eps_max"],
            }
            self._save_storage[self._save_storage_current_num][data.get_sub_interval()] = __data

    def __save_storage_extr(self):

        for data in self._storage:
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["all"]["extr_indexes"] \
                = data.get_extr_indexes_combined()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["all"]["extr_values"] \
                = data.get_extr_values_combined()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["all"]["begin"] \
                = data.all.begin
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["all"]["end"] \
                = data.all.end

            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["min"]["extr_indexes"] \
                = data.get_extr_indexes_min()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["min"]["extr_values"] \
                = data.get_extr_values_min()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["min"]["begin"] \
                = data.min.begin
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["min"]["end"] \
                = data.min.end

            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["max"]["extr_indexes"] \
                = data.get_extr_indexes_max()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["max"]["extr_values"] \
                = data.get_extr_values_max()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["max"]["begin"] \
                = data.max.begin
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["max"]["end"] \
                = data.max.end

            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["extr_eps_min"] \
                = data.get_extr_eps_min()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["extr_eps_max"] \
                = data.get_extr_eps_max()

    def __save_storage_trend(self):

        for data in self._storage:
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["all"]["trend_indexes"] \
                = data.get_trend_indexes_combined()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["all"]["trend_values"] \
                = data.get_trend_values_combined()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["min"]["trend_indexes"] \
                = data.get_trend_indexes_min()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["min"]["trend_values"] \
                = data.get_trend_values_min()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["max"]["trend_indexes"] \
                = data.get_trend_indexes_max()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["max"]["trend_values"] \
                = data.get_trend_values_max()

            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["trend_eps_min"] \
                = data.get_trend_eps_min()
            self._save_storage[data.get_current_iter()][data.get_sub_interval()]["trend_eps_max"] \
                = data.get_trend_eps_max()

    # endregion CoreLogic

    # region Extremum getter

    def get_extr_indexes_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["all"]["extr_indexes"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["all"]["extr_indexes"] is not None:
                    __data.extend(data["all"]["extr_indexes"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["all"]["extr_indexes"] is not None:
                    __data.extend(data["all"]["extr_indexes"])
            return np.array(__data)

    def get_extr_values_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["all"]["extr_values"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["all"]["extr_values"] is not None:
                    __data.extend(data["all"]["extr_values"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["all"]["extr_values"] is not None:
                    __data.extend(data["all"]["extr_values"])
            return np.array(__data)

    def get_extr_indexes_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["min"]["extr_indexes"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["min"]["extr_indexes"] is not None:
                    __data.extend(data["min"]["extr_indexes"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["min"]["extr_indexes"] is not None:
                    __data.extend(data["min"]["extr_indexes"])
            return np.array(__data)

    def get_extr_values_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.float32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["min"]["extr_values"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["min"]["extr_values"] is not None:
                    __data.extend(data["min"]["extr_values"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["min"]["extr_values"] is not None:
                    __data.extend(data["min"]["extr_values"])
            return np.array(__data)

    def get_extr_indexes_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["max"]["extr_indexes"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["max"]["extr_indexes"] is not None:
                    __data.extend(data["max"]["extr_indexes"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["max"]["extr_indexes"] is not None:
                    __data.extend(data["max"]["extr_indexes"])
            return np.array(__data)

    def get_extr_values_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.float32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["max"]["extr_values"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["max"]["extr_values"] is not None:
                    __data.extend(data["max"]["extr_values"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["max"]["extr_values"] is not None:
                    __data.extend(data["max"]["extr_values"])
            return np.array(__data)

    def get_extr_eps_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["extr_eps_min"]
            if __data is not None:
                return np.array([__data])
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["extr_eps_min"] is not None:
                    __data.append(data["extr_eps_min"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["extr_eps_min"] is not None:
                    __data.append(data["extr_eps_min"])
            return np.array(__data)

    def get_extr_max_eps(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["extr_eps_max"]
            if __data is not None:
                return np.array([__data])
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["extr_eps_max"] is not None:
                    __data.append(data["extr_eps_max"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["extr_eps_max"] is not None:
                    __data.append(data["extr_eps_max"])
            return np.array(__data)

    # endregion Extremum getter

    # region Trend getter

    def get_trend_indexes_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["all"]["trend_indexes"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["all"]["trend_indexes"] is not None:
                    __data.extend(data["all"]["trend_indexes"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["all"]["trend_indexes"] is not None:
                    __data.extend(data["all"]["trend_indexes"])
            return np.array(__data)

    def get_trend_values_combined(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["all"]["trend_values"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["all"]["trend_values"] is not None:
                    __data.extend(data["all"]["trend_values"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["all"]["trend_values"] is not None:
                    __data.extend(data["all"]["trend_values"])
            return np.array(__data)

    def get_trend_indexes_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["min"]["trend_indexes"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["min"]["trend_indexes"] is not None:
                    __data.extend(data["min"]["trend_indexes"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["min"]["trend_indexes"] is not None:
                    __data.extend(data["min"]["trend_indexes"])
            return np.array(__data)

    def get_trend_values_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.float32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["min"]["trend_values"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["min"]["trend_values"] is not None:
                    __data.extend(data["min"]["trend_values"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["min"]["trend_values"] is not None:
                    __data.extend(data["min"]["trend_values"])
            return np.array(__data)

    def get_trend_indexes_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["max"]["trend_indexes"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["max"]["trend_indexes"] is not None:
                    __data.extend(data["max"]["trend_indexes"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["max"]["trend_indexes"] is not None:
                    __data.extend(data["max"]["trend_indexes"])
            return np.array(__data)

    def get_trend_values_max(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.float32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["max"]["trend_values"]
            if __data is not None:
                return __data
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["max"]["trend_values"] is not None:
                    __data.extend(data["max"]["trend_values"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["max"]["trend_values"] is not None:
                    __data.extend(data["max"]["trend_values"])
            return np.array(__data)

    def get_trend_eps_min(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["trend_eps_min"]
            if __data is not None:
                return np.array([__data])
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["trend_eps_min"] is not None:
                    __data.append(data["trend_eps_min"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["trend_eps_min"] is not None:
                    __data.append(data["trend_eps_min"])
            return np.array(__data)

    def get_trend_max_eps(
            self,
            item: int | slice | None = None,
            after_iter: int | None = None,
    ) -> np.ndarray[np.uint32]:

        if after_iter is None:
            after_iter = self._save_storage_current_num
        elif isinstance(after_iter, int):
            if 0 >= after_iter > self._save_storage_current_num:
                raise IndexError(
                    f"After iteration {after_iter}. "
                    f"This is incorrect num. Current num: {self._save_storage_current_num}"
                )

        if isinstance(item, int):
            __data = self._save_storage[after_iter][item]["trend_eps_max"]
            if __data is not None:
                return np.array([__data])
            raise IndexError

        __data = []
        if item is None:
            for data in self._save_storage[after_iter]:
                if data["trend_eps_max"] is not None:
                    __data.append(data["trend_eps_max"])
            return np.array(__data)

        if isinstance(item, slice):
            for data in self._save_storage[after_iter][item]:
                if data["trend_eps_max"] is not None:
                    __data.append(data["trend_eps_max"])
            return np.array(__data)

    # endregion Trend getter

    def get_split_index(self) -> int:
        return self._split_index

    def get_current_iter(self) -> int:
        return self._save_storage_current_num


def main():
    np.random.seed(12222)

    size = 13
    batch = 3
    values = np.array([np.random.randint(10, 50) for _ in range(size)])
    indexes = np.arange(size)
    print(indexes)
    print(values)
    print()

    match_array = MatchesOnInputArray()
    trend_combined = CombinedTrendDetection(values=values, test_size=size, coincident=match_array)

    extr_storage = ExtremeStorage()
    extr_storage.build(values=values, split=size, batch=batch)

    for i in range(1):
        eps_extr = 2
        eps_trend = 2

        trend_combined.search_extremum(num_coincident=1, start_eps=eps_extr)
        trend_combined.search_change_trend_point(eps=eps_trend)

        extr_storage.search_extremes(coincident=1, eps=eps_extr)
        extr_storage.search_trends(eps_for_min=eps_trend, eps_for_max=eps_trend)

        print(f"{i:3d} Assert len.........")
        assert len(trend_combined.get_combined_indexes()) == len(extr_storage.get_extr_indexes_combined()), print(
            f"Old extr combined index: {len(trend_combined.get_combined_indexes())}",
            f"New extr combined index: {len(extr_storage.get_extr_indexes_combined())}"
        )
        assert len(trend_combined.get_min_indexes()) == len(extr_storage.get_extr_indexes_min()), print(
            f"Old extr min index: {len(trend_combined.get_min_indexes())}",
            f"New extr min index: {len(extr_storage.get_extr_indexes_min())}"
        )
        assert len(trend_combined.get_max_indexes()) == len(extr_storage.get_extr_indexes_max()), print(
            f"Old extr max index: {len(trend_combined.get_max_indexes())}",
            f"New extr max index: {len(extr_storage.get_extr_indexes_max())}"
        )
        assert len(trend_combined.get_combined_values()) == len(extr_storage.get_extr_values_combined()), print(
            f"Old extr combined index: {len(trend_combined.get_combined_values())}",
            f"New extr combined index: {len(extr_storage.get_extr_values_combined())}"
        )
        assert len(trend_combined.get_min_values()) == len(extr_storage.get_extr_values_min()), print(
            f"Old extr min index: {len(trend_combined.get_min_values())}",
            f"New extr min index: {len(extr_storage.get_extr_values_min())}"
        )
        assert len(trend_combined.get_max_values()) == len(extr_storage.get_extr_values_max()), print(
            f"Old extr max index: {len(trend_combined.get_max_values())}",
            f"New extr max index: {len(extr_storage.get_extr_values_max())}"
        )

        assert len(trend_combined.get_trend_indexes_by_min()) == len(extr_storage.get_trend_indexes_min()), print(
            f"Old extr min index: {len(trend_combined.get_trend_indexes_by_min())}",
            f"New extr min index: {len(extr_storage.get_trend_indexes_min())}"
        )
        assert len(trend_combined.get_trend_indexes_by_max()) == len(extr_storage.get_trend_indexes_max()), print(
            f"Old extr max index: {len(trend_combined.get_trend_indexes_by_max())}",
            f"New extr max index: {len(extr_storage.get_trend_indexes_max())}"
        )

        assert len(trend_combined.get_trend_values_by_min()) == len(extr_storage.get_trend_values_min()), print(
            f"Old extr min index: {len(trend_combined.get_trend_values_by_min())}",
            f"New extr min index: {len(extr_storage.get_trend_values_min())}"
        )
        assert len(trend_combined.get_trend_values_by_max()) == len(extr_storage.get_trend_values_max()), print(
            f"Old extr max index: {len(trend_combined.get_trend_values_by_max())}",
            f"New extr max index: {len(extr_storage.get_trend_values_max())}"
        )


if __name__ == '__main__':
    main()
