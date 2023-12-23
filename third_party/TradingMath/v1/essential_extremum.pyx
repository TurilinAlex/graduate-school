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

        # max
        max_v = n
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v <= eps:
                break

        max_extr_diff_view[i] = max_v

cdef void __diff_between_all_indexes_parallel(
    const DLONGINTTYPE_t[::1] index,
    DLONGINTTYPE_t[::1] min_extr_diff_view,
    DLONGINTTYPE_t[::1] max_extr_diff_view,
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

        # max
        max_v = n
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < max_v:
                max_v = diff
            if max_v <= eps:
                break

        max_extr_diff_view[i] = max_v


cpdef localize_minimals(
    const DLONGINTTYPE_t[::1] index,
    const DLONGINTTYPE_t coincident,
    const DLONGINTTYPE_t eps, bool parallel = False
):
    cdef:
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t i, min_eps = eps + coincident - 1
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
        DLONGINTTYPE_t i, max_eps = eps + coincident - 1
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
        DLONGINTTYPE_t i, j, min_v, max_v, diff, min_eps, max_eps
        DLONGINTTYPE_t min_split_index = 1
        DLONGINTTYPE_t max_split_index = n - 1
        DLONGINTTYPE_t[::1] min_extr_diff = np.empty_like(index)
        DLONGINTTYPE_t[::1] max_extr_diff = np.empty_like(index)
        DLONGINTTYPE_t[::1] min_extr_diff_index = np.empty_like(index)
        DLONGINTTYPE_t[::1] max_extr_diff_index = np.empty_like(index)

    min_eps = eps + coincident - 1
    max_eps = eps + coincident - 1

    # region Вычисление разницы между индексами

    if parallel:
        __diff_between_all_indexes_parallel(index, min_extr_diff, max_extr_diff, n, eps)
    else:
        __diff_between_all_indexes(index, min_extr_diff, max_extr_diff, n, eps)

    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных минимумов по заданному числу совпадений

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

    cdef np.ndarray[DLONGINTTYPE_t, ndim=1] min_main_index = np.empty((min_split_index,), np.int32)
    for i in range(min_split_index):
        min_main_index[i] = index[min_extr_diff_index[i]]

    # endregion Поиск главных локальных минимумов по заданному числу совпадений

    # region Поиск главных локальных максимумов по заданному числу совпадений

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

    cdef np.ndarray[DLONGINTTYPE_t, ndim=1] max_main_index = np.empty((n - max_split_index,), np.int32)
    for i in range(max_split_index, n):
        max_main_index[i - max_split_index] = index[max_extr_diff_index[i]]

    # endregion Поиск главных локальных максимумов по заданному числу совпадений

    return min_main_index, max_main_index, min_eps, max_eps
