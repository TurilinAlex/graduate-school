import numpy as np
from TradingMath.extremal import extremal_max, extremal_min

from .base_trend_point import BaseTrendDetection
from ..matches_extremum import BaseMatchesOnArray
from ..sort import argsort

__all__ = [
    "SplitTrendDetection",
]


class SplitTrendDetection(BaseTrendDetection):
    def __init__(
        self,
        values: np.ndarray[np.float32],
        test_size: int,
        coincident: BaseMatchesOnArray,
    ):
        super().__init__(values, test_size, coincident)

        self._last_values_min: np.ndarray[np.float32] | None = None
        self._last_values_max: np.ndarray[np.float32] | None = None
        self._last_indexes_min: np.ndarray[np.uint32] | None = None
        self._last_indexes_max: np.ndarray[np.uint32] | None = None

    # region Abstract implement

    def _start_iteration(self, num_coincident: int, start_eps: int):
        _index = argsort(self._values)
        _max_indexes, _max_eps = self._matches(
            extremal_max, index=_index, max_coincident=num_coincident, eps=start_eps
        )
        _min_indexes, _min_eps = self._matches(
            extremal_min, index=_index, max_coincident=num_coincident, eps=start_eps
        )

        _max_values = self._values[_max_indexes]
        _min_values = self._values[_min_indexes]

        _combined_indexes = np.sort(np.hstack((_max_indexes, _min_indexes)))
        _combined_values = self._values[_combined_indexes]

        self._last_min_values = _min_values
        self._last_max_values = _max_values

        self._last_min_indexes = _min_indexes
        self._last_max_indexes = _max_indexes

        self._last_values = _combined_values
        self._last_indexes = _combined_indexes

        self._setter(
            min_values=_min_values,
            min_indexes=_min_indexes,
            max_values=_max_values,
            max_indexes=_max_indexes,
            combined_values=_combined_values,
            combined_indexes=_combined_indexes,
            min_eps=_min_eps,
            max_eps=_max_eps,
        )

    def _continuation_iterations(self, num_coincident: int, start_eps: int):
        _index_min = argsort(self._last_min_values)
        _index_max = argsort(self._last_max_values)
        __max_index, _max_eps = self._matches(
            extremal_max, index=_index_max, max_coincident=num_coincident, eps=start_eps
        )
        __min_index, _min_eps = self._matches(
            extremal_min, index=_index_min, max_coincident=num_coincident, eps=start_eps
        )

        _max_values = self._last_max_values[__max_index]
        _min_values = self._last_min_values[__min_index]

        _min_indexes = self._last_min_indexes[__min_index]
        _max_indexes = self._last_max_indexes[__max_index]

        _combined_indexes = np.sort(np.hstack((_max_indexes, _min_indexes)))
        _combined_values = self._values[_combined_indexes]

        self._last_min_indexes = _min_indexes
        self._last_max_indexes = _max_indexes

        self._last_min_values = _min_values
        self._last_max_values = _max_values

        self._last_indexes = _combined_indexes
        self._last_values = _combined_values

        self._setter(
            min_values=_min_values,
            min_indexes=_min_indexes,
            max_values=_max_values,
            max_indexes=_max_indexes,
            combined_values=_combined_values,
            combined_indexes=_combined_indexes,
            min_eps=_min_eps,
            max_eps=_max_eps,
        )

    # endregion Abstract implement
