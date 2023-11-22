from abc import abstractmethod
from enum import Enum
from typing import Protocol

import numpy as np
from TradingMath.extremal import extremal_min, extremal_max

from ..matches_extremum import BaseMatchesOnArray
from ..sort import argsort

__all__ = [
    "BaseTrendDetection",
]


class BaseTrendDetection(Protocol):
    """
    Базовый класс реализующий функционал по поиску точек смены тренда
    """

    class ExtrName(Enum):
        min_indexes = "min_indexes_iter_{after_iter}_extr"
        max_indexes = "max_indexes_iter_{after_iter}_extr"
        max_values = "max_values_iter_{after_iter}_extr"
        min_values = "min_values_iter_{after_iter}_extr"
        max_eps = "max_eps_iter_{after_iter}_extr"
        min_eps = "min_eps_iter_{after_iter}_extr"
        combined_indexes = "combined_indexes_iter_{after_iter}_extr"
        combined_values = "combined_values_iter_{after_iter}_extr"

    class TrendName(Enum):
        min_indexes = "min_indexes_iter_{after_iter}_trend"
        max_indexes = "max_indexes_iter_{after_iter}_trend"
        max_values = "max_values_iter_{after_iter}_trend"
        min_values = "min_values_iter_{after_iter}_trend"
        max_eps = "max_eps_iter_{after_iter}_trend"
        min_eps = "min_eps_iter_{after_iter}_trend"
        combined_indexes = "combined_indexes_iter_{after_iter}_trend"
        combined_values = "combined_values_iter_{after_iter}_trend"

    # region Other getter
    def __init__(
        self,
        values: np.ndarray[np.float32],
        test_size: int,
        matches: BaseMatchesOnArray,
    ):
        self._matches = matches
        self._all_values = values
        self._values = values[:test_size]
        self._current_iter: int = 0
        self._split_index: int = test_size
        self._last_values = None
        self._last_indexes = None

    def get_split_index(self) -> int:
        return self._split_index

    def get_all_values(self) -> np.ndarray[np.float32]:
        return self._all_values

    def get_current_iter(self) -> int:
        return self._current_iter

    # endregion Other getter

    # region Extremum getter
    def get_min_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.ExtrName.min_indexes, after_iter)

    def get_max_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.ExtrName.max_indexes, after_iter)

    def get_min_values(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        return self._getter(self.ExtrName.min_values, after_iter)

    def get_max_values(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        return self._getter(self.ExtrName.max_values, after_iter)

    def get_min_eps(self, after_iter: int | None = None) -> int:
        return self._getter(self.ExtrName.min_eps, after_iter)

    def get_max_eps(self, after_iter: int | None = None) -> int:
        return self._getter(self.ExtrName.max_eps, after_iter)

    def get_combined_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.ExtrName.combined_indexes, after_iter)

    def get_combined_values(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.ExtrName.combined_values, after_iter)

    # endregion Extremum getter

    # region Trend getter

    def get_trend_indexes_by_min(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.TrendName.min_indexes, after_iter)

    def get_trend_indexes_by_max(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.TrendName.max_indexes, after_iter)

    def get_trend_values_by_min(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        return self._getter(self.TrendName.min_values, after_iter)

    def get_trend_values_by_max(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        return self._getter(self.TrendName.max_values, after_iter)

    def get_trend_eps_by_min(self, after_iter: int | None = None) -> int:
        return self._getter(self.TrendName.min_eps, after_iter)

    def get_trend_eps_by_max(self, after_iter: int | None = None) -> int:
        return self._getter(self.TrendName.max_eps, after_iter)

    def get_trend_combined_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.TrendName.combined_indexes, after_iter)

    def get_trend_combined_values(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        return self._getter(self.TrendName.combined_values, after_iter)

    # endregion Trend getter

    # region Calculation Function

    def search_extremum(self, num_coincident: int, start_eps: int):
        self._current_iter += 1
        if self._last_values is None and self._last_indexes is None:
            self._start_iteration(num_coincident, start_eps)
        else:
            self._continuation_iterations(num_coincident, start_eps)

        return self

    def search_change_trend_point(self, eps: int, after_iter: int | None = None):
        self.search_trend_point_by_min(after_iter=after_iter, eps=eps)
        self.search_trend_point_by_max(after_iter=after_iter, eps=eps)

    def search_trend_point_by_min(self, eps: int, after_iter: int | None = None):
        if after_iter is None:
            after_iter = self.get_current_iter()

        values = self.get_min_values(after_iter)
        indexes = self.get_min_indexes(after_iter)
        if values is not None and indexes is not None:
            self._search_trend_point_by_min(
                after_iter=after_iter,
                indexes=indexes,
                values=values,
                eps=eps,
            )

    def search_trend_point_by_max(self, eps: int, after_iter: int | None = None):
        if after_iter is None:
            after_iter = self.get_current_iter()

        values = self.get_max_values(after_iter)
        indexes = self.get_max_indexes(after_iter)
        if values is not None and indexes is not None:
            self._search_trend_point_by_max(
                after_iter=after_iter,
                indexes=indexes,
                values=values,
                eps=eps,
            )

    def _search_trend_point_by_min(
        self,
        values: np.ndarray[np.float32],
        indexes: np.ndarray[np.uint32],
        eps: int,
        after_iter: int | None = None,
    ):
        _index = argsort(values)
        __min_index = np.sort(extremal_min(index=_index, eps=eps))
        _min_values = values[__min_index]
        _min_indexes = indexes[__min_index]

        setattr(
            self,
            self.TrendName.min_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.min_values.value.format(after_iter=after_iter),
            _min_values,
        )
        setattr(
            self,
            self.TrendName.min_indexes.value.format(after_iter=after_iter),
            _min_indexes,
        )

    def _search_trend_point_by_max(
        self,
        values: np.ndarray[np.float32],
        indexes: np.ndarray[np.uint32],
        eps: int,
        after_iter: int | None = None,
    ):
        _index = argsort(values)
        __max_index = np.sort(extremal_max(index=_index, eps=eps))
        _max_values = values[__max_index]
        _max_indexes = indexes[__max_index]

        setattr(
            self,
            self.TrendName.max_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.max_values.value.format(after_iter=after_iter),
            _max_values,
        )
        setattr(
            self,
            self.TrendName.max_indexes.value.format(after_iter=after_iter),
            _max_indexes,
        )

    def _getter(self, name: ExtrName | TrendName, after_iter: int | None = None):
        result = None

        if after_iter is None:
            after_iter = self.get_current_iter()
        if after_iter > self._current_iter:
            return result
        try:
            result = getattr(self, name.value.format(after_iter=after_iter))
        except AttributeError as error:
            print(error)

        return result

    def _setter(
        self,
        min_values: np.ndarray[np.float32],
        min_indexes: np.ndarray[np.uint32],
        max_values: np.ndarray[np.float32],
        max_indexes: np.ndarray[np.uint32],
        combined_values: np.ndarray[np.float32],
        combined_indexes: np.ndarray[np.uint32],
        min_eps: int,
        max_eps: int,
    ):
        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            min_indexes,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            max_indexes,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            combined_values,
        )
        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            combined_indexes,
        )

    # endregion Calculation Function

    # region Abstract implement

    @abstractmethod
    def _start_iteration(self, num_coincident: int, start_eps: int):
        pass

    @abstractmethod
    def _continuation_iterations(self, num_coincident: int, start_eps: int):
        pass

    # endregion Abstract implement
