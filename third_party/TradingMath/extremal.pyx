# distutils: language=c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, infer_types=True

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool
from cython.parallel cimport prange

from .extremal cimport DALLTYPE_t, DLONGINTTYPE_t

cpdef void argsort(DALLTYPE_t[:] value, DLONGINTTYPE_t[:] index_view, bool reverse = False):
    cdef DLONGINTTYPE_t n = value.shape[0]
    if reverse:
        __argsort_reverse(value, index_view, n)
    else:
        __argsort(value, index_view, n)

cpdef search_main_extremum(
    const DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t coincident, const DLONGINTTYPE_t eps, bool parallel = False
):
    cdef:
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t i, j, min_v, max_v, diff, min_eps, max_eps
        DLONGINTTYPE_t min_split_index = 1
        DLONGINTTYPE_t max_split_index = n - 1
        DLONGINTTYPE_t[:] min_extr_diff = np.empty_like(index)
        DLONGINTTYPE_t[:] max_extr_diff = np.empty_like(index)
        DLONGINTTYPE_t[:] min_extr_diff_index = np.empty_like(index)
        DLONGINTTYPE_t[:] max_extr_diff_index = np.empty_like(index)

    min_eps = eps
    max_eps = eps

    # region Вычисление разницы между индексами

    if parallel:
        __diff_between_indexes_parallel(index, min_extr_diff, max_extr_diff, n, eps)
    else:
        __diff_between_indexes(index, min_extr_diff, max_extr_diff, n, eps)

    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных минимумов по заданному числу совпадений

    __argsort_reverse(min_extr_diff, min_extr_diff_index, n)

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

    cdef np.ndarray[DLONGINTTYPE_t, ndim=1] min_main_index = np.empty((min_split_index,), np.int64)
    for i in range(min_split_index):
        min_main_index[i] = index[min_extr_diff_index[i]]

    # endregion Поиск главных локальных минимумов по заданному числу совпадений

    # region Поиск главных локальных максимумов по заданному числу совпадений

    __argsort(max_extr_diff, max_extr_diff_index, n)

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

    cdef np.ndarray[DLONGINTTYPE_t, ndim=1] max_main_index = np.empty((n - max_split_index,), np.int64)
    for i in range(max_split_index, n):
        max_main_index[i - max_split_index] = index[max_extr_diff_index[i]]

    # endregion Поиск главных локальных максимумов по заданному числу совпадений

    return min_main_index, max_main_index, min_eps, max_eps

cpdef void __argsort(DALLTYPE_t[:] value, DLONGINTTYPE_t[:] index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, h, size, ls, le, rs, re, hh
    cdef int status = 0

    try:
        h = 0
        size = 8
        while h < n:
            if h + size > n:
                hh = n
            else:
                hh = h + size
            for rs in range(h, hh):
                k = h
                for ls in range(h, rs):
                    if value[rs] >= value[ls]:
                        k += 1
                for ls in range(rs + 1, hh):
                    if value[rs] > value[ls]:
                        k += 1
                index_view[k] = rs
            h += size

        h = size
        while h <= n:
            hh = 0
            while hh < n - h:
                ls = hh
                le = hh + h - 1
                rs = hh + h
                if hh + 2 * h - 1 < n - 1:
                    re = hh + 2 * h - 1
                else:
                    re = n - 1

                memcpy(index_temp_view + ls, &index_view[0] + ls, (re + 1 - ls) * sizeof(DLONGINTTYPE_t))

                for k in range(ls, re + 1):
                    if ls <= le and rs <= re:
                        if value[index_temp_view[rs]] == value[index_temp_view[ls]]:
                            status = 0
                        if value[index_temp_view[rs]] < value[index_temp_view[ls]]:
                            status = -1
                        if value[index_temp_view[rs]] > value[index_temp_view[ls]]:
                            status = 1
                    if ls > le: status = -1
                    if rs > re: status = 1

                    if status >= 0:
                        index_view[k] = index_temp_view[ls]
                        ls += 1
                    else:
                        index_view[k] = index_temp_view[rs]
                        rs += 1
                hh += 2 * h
            h *= 2
    finally:
        free(index_temp_view)

cpdef void __argsort_reverse(DALLTYPE_t[:] value, DLONGINTTYPE_t[:] index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, h, size, ls, le, rs, re, hh
    cdef int status = 0

    try:
        h = 0
        size = 8
        while h < n:
            if h + size > n:
                hh = n
            else:
                hh = h + size
            for rs in range(h, hh):
                k = hh - 1
                for ls in range(h, rs):
                    if value[rs] >= value[ls]:
                        k -= 1
                for ls in range(rs + 1, hh):
                    if value[rs] > value[ls]:
                        k -= 1
                index_view[k] = rs
            h += size

        h = size
        while h <= n:
            hh = 0
            while hh < n - h:
                ls = hh
                le = hh + h - 1
                rs = hh + h
                if hh + 2 * h - 1 < n - 1:
                    re = hh + 2 * h - 1
                else:
                    re = n - 1

                memcpy(index_temp_view + ls, &index_view[0] + ls, (re + 1 - ls) * sizeof(DLONGINTTYPE_t))

                for k in range(ls, re + 1):
                    if ls <= le and rs <= re:
                        if value[index_temp_view[rs]] == value[index_temp_view[ls]]:
                            status = 0
                        if value[index_temp_view[rs]] < value[index_temp_view[ls]]:
                            status = 1
                        if value[index_temp_view[rs]] > value[index_temp_view[ls]]:
                            status = -1
                    if ls > le: status = -1
                    if rs > re: status = 1

                    if status >= 1:
                        index_view[k] = index_temp_view[ls]
                        ls += 1
                    else:
                        index_view[k] = index_temp_view[rs]
                        rs += 1
                hh += 2 * h
            h *= 2
    finally:
        free(index_temp_view)

cdef void __argsort_ptr(DLONGINTTYPE_t *value, DLONGINTTYPE_t *index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, h, size, ls, le, rs, re, hh
    cdef int status = 0

    try:
        h = 0
        size = 8
        while h < n:
            if h + size > n:
                hh = n
            else:
                hh = h + size
            for rs in range(h, hh):
                k = h
                for ls in range(h, rs):
                    if value[rs] >= value[ls]:
                        k += 1
                for ls in range(rs + 1, hh):
                    if value[rs] > value[ls]:
                        k += 1
                index_view[k] = rs
            h += size

        h = size
        while h <= n:
            hh = 0
            while hh < n - h:
                ls = hh
                le = hh + h - 1
                rs = hh + h
                if hh + 2 * h - 1 < n - 1:
                    re = hh + 2 * h - 1
                else:
                    re = n - 1

                memcpy(index_temp_view + ls, &index_view[0] + ls, (re + 1 - ls) * sizeof(DLONGINTTYPE_t))

                for k in range(ls, re + 1):
                    if ls <= le and rs <= re:
                        if value[index_temp_view[rs]] == value[index_temp_view[ls]]:
                            status = 0
                        if value[index_temp_view[rs]] < value[index_temp_view[ls]]:
                            status = -1
                        if value[index_temp_view[rs]] > value[index_temp_view[ls]]:
                            status = 1
                    if ls > le: status = -1
                    if rs > re: status = 1

                    if status >= 0:
                        index_view[k] = index_temp_view[ls]
                        ls += 1
                    else:
                        index_view[k] = index_temp_view[rs]
                        rs += 1
                hh += 2 * h
            h *= 2
    finally:
        free(index_temp_view)

cdef void __argsort_ptr_reverse(DLONGINTTYPE_t *value, DLONGINTTYPE_t *index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, h, size, ls, le, rs, re, hh
    cdef int status = 0

    try:
        h = 0
        size = 8
        while h < n:
            if h + size > n:
                hh = n
            else:
                hh = h + size
            for rs in range(h, hh):
                k = hh - 1
                for ls in range(h, rs):
                    if value[rs] >= value[ls]:
                        k -= 1
                for ls in range(rs + 1, hh):
                    if value[rs] > value[ls]:
                        k -= 1
                index_view[k] = rs
            h += size

        h = size
        while h <= n:
            hh = 0
            while hh < n - h:
                ls = hh
                le = hh + h - 1
                rs = hh + h
                if hh + 2 * h - 1 < n - 1:
                    re = hh + 2 * h - 1
                else:
                    re = n - 1

                memcpy(index_temp_view + ls, &index_view[0] + ls, (re + 1 - ls) * sizeof(DLONGINTTYPE_t))

                for k in range(ls, re + 1):
                    if ls <= le and rs <= re:
                        if value[index_temp_view[rs]] == value[index_temp_view[ls]]:
                            status = 0
                        if value[index_temp_view[rs]] < value[index_temp_view[ls]]:
                            status = 1
                        if value[index_temp_view[rs]] > value[index_temp_view[ls]]:
                            status = -1
                    if ls > le: status = -1
                    if rs > re: status = 1

                    if status >= 1:
                        index_view[k] = index_temp_view[ls]
                        ls += 1
                    else:
                        index_view[k] = index_temp_view[rs]
                        rs += 1
                hh += 2 * h
            h *= 2
    finally:
        free(index_temp_view)

cdef void __extremal_max(
    DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
) noexcept nogil:
    cdef:
        DLONGINTTYPE_t k, l

    for k in range(n):
        extremum_view[k] = -1
        for l in range(1, (n - k)):
            if abs(index[k] - index[k + l]) <= eps:
                break
        else:
            extremum_view[k] = index[k]

cdef void __extremal_max_parallel(
    DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
) noexcept nogil:
    cdef:
        DLONGINTTYPE_t k, l

    for k in prange(n, nogil=True):
        extremum_view[k] = -1
        for l in range(1, (n - k)):
            if abs(index[k] - index[k + l]) <= eps:
                break
        else:
            extremum_view[k] = index[k]

cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] extremal_max(DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t eps, bool parallel = False):
    cdef:
        np.ndarray[DLONGINTTYPE_t, ndim=1] return_extremum, positive
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t[:] extremum_view

    extremum_view = np.empty_like(index, dtype=np.int64)

    if parallel:
        __extremal_max_parallel(index, extremum_view, n, eps)
    else:
        __extremal_max(index, extremum_view, n, eps)

    return_extremum = np.asarray(extremum_view)
    positive = np.extract(return_extremum >= 0, return_extremum)
    return positive

cdef void __extremal_min(
    DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
) noexcept nogil:
    cdef:
        DLONGINTTYPE_t k, l

    for k in range(n):
        extremum_view[k] = -1
        for l in range(1, k + 1):
            if abs(index[k] - index[k - l]) <= eps:
                break
        else:
            extremum_view[k] = index[k]

cdef void __extremal_min_parallel(
    DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
) noexcept nogil:
    cdef:
        DLONGINTTYPE_t k, l

    for k in prange(n, nogil=True):
        extremum_view[k] = -1
        for l in range(1, k + 1):
            if abs(index[k] - index[k - l]) <= eps:
                break
        else:
            extremum_view[k] = index[k]

cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] extremal_min(DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t eps, bool parallel = False):
    cdef:
        np.ndarray[DLONGINTTYPE_t, ndim=1] return_extremum, positive
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t[:] extremum_view

    extremum_view = np.empty_like(index, dtype=np.int64)

    if parallel:
        __extremal_min_parallel(index, extremum_view, n, eps)
    else:
        __extremal_min(index, extremum_view, n, eps)

    return_extremum = np.asarray(extremum_view)
    positive = np.extract(return_extremum >= 0, return_extremum)
    return positive

cdef void __diff_between_indexes(
    const DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] min_extr_diff_view, DLONGINTTYPE_t[:] max_extr_diff_view,
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

cdef void __diff_between_indexes_parallel(
    const DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] min_extr_diff_view, DLONGINTTYPE_t[:] max_extr_diff_view,
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
