# distutils: language=c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, infer_types=True

import numpy as np
cimport numpy as np

from libcpp cimport bool
from cython.parallel cimport prange

from ..sort cimport argsort
from ..typing cimport DLONGINTTYPE_t

__all__ = [
    "localize_extremes",
    "localize_minimals",
    "localize_maximals",
]


cdef void __diff_between_minimals_indexes(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] min_extr_diff_view,
    const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
) noexcept nogil:

    cdef:
        DLONGINTTYPE_t i, j, min_v, diff

    for i in range(n):
        # min
        min_v = n
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            if diff < min_v:
                min_v = diff
            if min_v <= eps:
                break

        min_extr_diff_view[i] = min_v

cdef void __diff_between_minimals_indexes_parallel(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] min_extr_diff_view,
    const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
) noexcept nogil:

    cdef:
        DLONGINTTYPE_t i, j, min_v, diff

    for i in prange(n, nogil=True):
        # min
        min_v = n
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            if diff < min_v:
                min_v = diff
            if min_v <= eps:
                break

        min_extr_diff_view[i] = min_v

cdef void __diff_between_maximals_indexes(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] max_extr_diff_view,
    const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
) noexcept nogil:

    cdef:
        DLONGINTTYPE_t i, j, max_v, diff

    for i in range(n):
        # max
        max_v = n
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v <= eps:
                break

        max_extr_diff_view[i] = max_v

cdef void __diff_between_maximals_indexes_parallel(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] max_extr_diff_view,
    const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
) noexcept nogil:

    cdef:
        DLONGINTTYPE_t i, j, max_v, diff

    for i in prange(n, nogil=True):
        # max
        max_v = n
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v <= eps:
                break

        max_extr_diff_view[i] = max_v

cdef void __diff_between_all_indexes(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] min_extr_diff_view,
    DLONGINTTYPE_t[::1] max_extr_diff_view,
    DLONGINTTYPE_t[::1] min_count,
    DLONGINTTYPE_t[::1] max_count,
    const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
) noexcept nogil:

    cdef:
        DLONGINTTYPE_t i, j, min_v, max_v, diff

    for i in range(n):
        # min
        min_v = n
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            if diff < min_v:
                min_v = diff
            if min_v <= eps:
                break

        min_extr_diff_view[i] = min_v
        min_count[min_v] += 1

        # max
        max_v = n
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v <= eps:
                break

        max_extr_diff_view[i] = max_v
        max_count[max_v] += 1

cdef void __diff_between_all_indexes_parallel(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] min_extr_diff_view,
    DLONGINTTYPE_t[::1] max_extr_diff_view,
    DLONGINTTYPE_t[::1] min_count,
    DLONGINTTYPE_t[::1] max_count,
    const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
) noexcept nogil:

    cdef:
        DLONGINTTYPE_t i, j, min_v, max_v, diff

    for i in prange(n, nogil=True):
        # min
        min_v = n
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            if diff < min_v:
                min_v = diff
            if min_v <= eps:
                break

        min_extr_diff_view[i] = min_v
        min_count[min_v] += 1

        # max
        max_v = n
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v <= eps:
                break

        max_extr_diff_view[i] = max_v
        max_count[max_v] += 1

cpdef localize_minimals(
    const DLONGINTTYPE_t[::1] index,
    const DLONGINTTYPE_t coincident,
    const DLONGINTTYPE_t eps, bool parallel = False
):
    cdef:
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t i, min_eps = eps
        DLONGINTTYPE_t min_split_index = 1
        DLONGINTTYPE_t[::1] min_extr_diff = np.empty_like(index)
        DLONGINTTYPE_t[::1] min_extr_diff_index = np.empty_like(index)
        np.ndarray[DLONGINTTYPE_t, ndim=1] min_main_index

    # region Вычисление разницы между индексами для минимумов

    if parallel:
        __diff_between_minimals_indexes_parallel(index, min_extr_diff, n, eps)
    else:
        __diff_between_minimals_indexes(index, min_extr_diff, n, eps)

    # endregion Вычисление разницы между индексами для минимумов

    # region Поиск главных локальных минимумов по заданному числу совпадений и eps-окрестности

    argsort(min_extr_diff, min_extr_diff_index, reverse=True, parallel=parallel)

    for i in range(n - 1, 0, -1):
        if min_extr_diff[min_extr_diff_index[i - 1]] <= eps:
            continue
        if (
                min_extr_diff[min_extr_diff_index[i - 1]] -
                min_extr_diff[min_extr_diff_index[i]]
                >= coincident
        ):
            min_split_index = i
            min_eps = min_extr_diff[min_extr_diff_index[i]] + coincident - 1
            break

    min_main_index = np.empty((min_split_index,), np.int32)
    for i in range(min_split_index):
        min_main_index[i] = index[min_extr_diff_index[i]]

    # endregion Поиск главных локальных минимумов по заданному числу совпадений и eps-окрестности

    return min_main_index, min_eps

cpdef localize_maximals(
    const DLONGINTTYPE_t[::1] index,
    const DLONGINTTYPE_t coincident,
    const DLONGINTTYPE_t eps, bool parallel = False
):
    cdef:
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t i, max_eps = eps
        DLONGINTTYPE_t max_split_index = n - 1
        DLONGINTTYPE_t[::1] max_extr_diff = np.empty_like(index)
        DLONGINTTYPE_t[::1] max_extr_diff_index = np.empty_like(index)
        np.ndarray[DLONGINTTYPE_t, ndim=1] max_main_index

    # region Вычисление разницы между индексами

    if parallel:
        __diff_between_maximals_indexes_parallel(index, max_extr_diff, n, eps)
    else:
        __diff_between_maximals_indexes(index, max_extr_diff, n, eps)

    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных максимумов по заданному числу совпадений и eps-окрестности

    argsort(max_extr_diff, max_extr_diff_index, reverse=False, parallel=parallel)

    for i in range(n - 1):
        if max_extr_diff[max_extr_diff_index[i + 1]] <= eps:
            continue
        if (
                max_extr_diff[max_extr_diff_index[i + 1]] -
                max_extr_diff[max_extr_diff_index[i]]
                >= coincident
        ):
            max_split_index = i + 1
            max_eps = max_extr_diff[max_extr_diff_index[i]] + coincident - 1
            break

    max_main_index = np.empty((n - max_split_index,), np.int32)
    for i in range(max_split_index, n):
        max_main_index[i - max_split_index] = index[max_extr_diff_index[i]]

    # endregion Поиск главных локальных максимумов по заданному числу совпадений и eps-окрестности

    return max_main_index, max_eps

cpdef localize_extremes(
    const DLONGINTTYPE_t[::1] index,
    const DLONGINTTYPE_t coincident,
    const DLONGINTTYPE_t eps, bool parallel = False
):
    cdef:
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t i, j, min_v, max_v, diff, min_eps, max_eps, last_index, diff_for_coincident, sum_min, sum_max, ii
        DLONGINTTYPE_t[::1] min_extr_diff = np.zeros((n,), dtype=np.int32)
        DLONGINTTYPE_t[::1] max_extr_diff = np.zeros((n,), dtype=np.int32)
        np.ndarray[DLONGINTTYPE_t, ndim=1] min_extr_diff_index
        np.ndarray[DLONGINTTYPE_t, ndim=1] max_extr_diff_index

        DLONGINTTYPE_t[::1] min_count = np.zeros((n + 1,), dtype=np.int32)
        DLONGINTTYPE_t[::1] max_count = np.zeros((n + 1,), dtype=np.int32)

        np.ndarray[DLONGINTTYPE_t, ndim=1] min_main_index
        np.ndarray[DLONGINTTYPE_t, ndim=1] max_main_index

    # region Вычисление разницы между индексами

    if parallel:
        __diff_between_all_indexes_parallel(index, min_extr_diff, max_extr_diff, min_count, max_count, n, eps)
    else:
        __diff_between_all_indexes(index, min_extr_diff, max_extr_diff, min_count, max_count, n, eps)

    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных минимумов по заданному числу совпадений

    last_index = eps
    while min_count[last_index] <= 0:
        last_index += 1

    sum_min = 1

    min_eps = eps + coincident - 1
    for i in range(eps, n + 1):
        if min_count[i] >= 1 or coincident == 1:
            diff_for_coincident = i - last_index
            min_eps = last_index + coincident - 1
            last_index = i
            if diff_for_coincident >= coincident:
                for j in range(last_index, n):
                    sum_min += min_count[j]
                break

    if sum_min == 1:
        min_main_index = np.array([index[0]])
    else:
        ii = 0
        min_main_index = np.empty((sum_min,), np.int32)
        for i in range(n):
            if min_extr_diff[i] > min_eps:
                min_main_index[ii] = index[i]
                ii += 1

    # endregion Поиск главных локальных минимумов по заданному числу совпадений

    # region Поиск главных локальных максимумов по заданному числу совпадений

    last_index = eps
    while max_count[last_index] <= 0:
        last_index += 1

    sum_max = 1
    max_eps = eps + coincident - 1
    for i in range(eps, n + 1):
        if max_count[i] >= 1 or coincident == 1:
            diff_for_coincident = i - last_index
            max_eps = last_index + coincident - 1
            last_index = i
            if diff_for_coincident >= coincident:
                for j in range(last_index, n):
                    sum_max += max_count[j]
                break

    if sum_max == 1:
        max_main_index = np.array([index[n - 1]])
    else:
        ii = 0
        max_main_index = np.empty((sum_max,), np.int32)
        for i in range(n):
            if max_extr_diff[i] > max_eps:
                max_main_index[ii] = index[i]
                ii += 1

    # endregion Поиск главных локальных максимумов по заданному числу совпадений

    return min_main_index, max_main_index, min_eps, max_eps
