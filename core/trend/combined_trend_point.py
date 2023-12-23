import numpy as np
from TradingMath.extremum import localize_minimals, localize_maximals

from .base_trend_point import BaseTrendDetection
from ..matches_extremum import BaseMatchesOnArray
from ..sort import argsort

__all__ = [
    "CombinedTrendDetection",
]


class CombinedTrendDetection(BaseTrendDetection):
    def __init__(
        self,
        values: np.ndarray[np.float32],
        test_size: int,
        coincident: BaseMatchesOnArray,
    ):
        super().__init__(values, test_size, coincident)

    # region Abstract implement

    def _start_iteration(self, num_coincident: int, start_eps: int):
        _index = argsort(self._values)
        _max_indexes, _max_eps = self._matches(
            localize_maximals, index=_index, max_coincident=num_coincident, eps=start_eps
        )
        _min_indexes, _min_eps = self._matches(
            localize_minimals, index=_index, max_coincident=num_coincident, eps=start_eps
        )
        _max_values = self._values[_max_indexes]
        _min_values = self._values[_min_indexes]

        _combined_indexes = np.sort(np.hstack((_max_indexes, _min_indexes)))
        _combined_values = self._values[_combined_indexes]

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
        _index = argsort(self._last_values)
        __max_index, _max_eps = self._matches(
            localize_maximals, index=_index, max_coincident=num_coincident, eps=start_eps
        )
        __min_index, _min_eps = self._matches(
            localize_minimals, index=_index, max_coincident=num_coincident, eps=start_eps
        )
        _max_values = self._last_values[__max_index]
        _min_values = self._last_values[__min_index]

        _index_combined = np.sort(np.hstack((__max_index, __min_index)))

        _min_indexes = self._last_indexes[__min_index]
        _max_indexes = self._last_indexes[__max_index]
        _combined_values = self._last_values[_index_combined]
        _combined_indexes = self._last_indexes[_index_combined]

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
