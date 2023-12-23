# distutils: language=c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, infer_types=True

from math import log2, ceil

from libcpp cimport bool
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

from cython.parallel cimport prange

from .typing cimport DALLTYPE_t, DLONGINTTYPE_t

__all__ = [
    "argsort",
]

cdef void __inner_merge(DALLTYPE_t[::1] value, DLONGINTTYPE_t* index_temp_view, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t hh, const DLONGINTTYPE_t h) noexcept nogil:

    cdef DLONGINTTYPE_t k, ls, le, rs, re

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
            elif value[index_temp_view[rs]] < value[index_temp_view[ls]]:
                status = -1
            else:
                status = 1
        if ls > le: status = -1
        if rs > re: status = 1

        if status >= 0:
            index_view[k] = index_temp_view[ls]
            ls += 1
        else:
            index_view[k] = index_temp_view[rs]
            rs += 1

cdef void __inner_merge_reverse(DALLTYPE_t[::1] value, DLONGINTTYPE_t* index_temp_view, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t hh, const DLONGINTTYPE_t h) noexcept nogil:

    cdef DLONGINTTYPE_t k, ls, le, rs, re

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
            elif value[index_temp_view[rs]] < value[index_temp_view[ls]]:
                status = 1
            else:
                status = -1
        if ls > le: status = -1
        if rs > re: status = 1

        if status >= 1:
            index_view[k] = index_temp_view[ls]
            ls += 1
        else:
            index_view[k] = index_temp_view[rs]
            rs += 1


cpdef void _argsort(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, ls, le, rs, re, hh, step
    cdef int h = 0
    cdef int status = 0

    cdef int degree = int(ceil(log2(n)))
    cdef int start_degree = 3
    cdef int size = int(2 ** start_degree)

    try:
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

        for i in range(start_degree, degree):
            h = int(2 ** i)
            step = 2 * h
            for hh in range(0, n - h, step):
                __inner_merge(value, index_temp_view, index_view, n, hh, h)
    finally:
        free(index_temp_view)

cpdef void _argsort_parallel(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, ls, le, rs, re, hh, step
    cdef int h = 0
    cdef int status = 0

    cdef int degree = int(ceil(log2(n)))
    cdef int start_degree = 3
    cdef int size = int(2 ** start_degree)

    try:
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

        for i in range(start_degree, degree):
            h = int(2 ** i)
            step = 2 * h
            for hh in prange(0, n - h, step, nogil=True):
                __inner_merge(value, index_temp_view, index_view, n, hh, h)
    finally:
        free(index_temp_view)

cpdef void _argsort_reverse(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, ls, le, rs, re, hh, step
    cdef int h = 0
    cdef int status = 0

    cdef int degree = int(ceil(log2(n)))
    cdef int start_degree = 3
    cdef int size = int(2 ** start_degree)

    try:
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

        for i in range(start_degree, degree):
            h = int(2 ** i)
            step = 2 * h
            for hh in range(0, n - h, step):
                __inner_merge_reverse(value, index_temp_view, index_view, n, hh, h)
    finally:
        free(index_temp_view)

cpdef void _argsort_reverse_parallel(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n):

    cdef DLONGINTTYPE_t *index_temp_view = <DLONGINTTYPE_t *> malloc(n * sizeof(DLONGINTTYPE_t))

    if not index_temp_view:
        raise MemoryError()

    cdef DLONGINTTYPE_t k, ls, le, rs, re, hh, step
    cdef int h = 0
    cdef int status = 0

    cdef int degree = int(ceil(log2(n)))
    cdef int start_degree = 3
    cdef int size = int(2 ** start_degree)

    try:
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

        for i in range(start_degree, degree):
            h = int(2 ** i)
            step = 2 * h
            for hh in prange(0, n - h, step, nogil=True):
                __inner_merge_reverse(value, index_temp_view, index_view, n, hh, h)
    finally:
        free(index_temp_view)

cpdef void argsort(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, bool reverse = False, bool parallel = False):
    cdef DLONGINTTYPE_t n = value.shape[0]
    if not reverse and not parallel:
        _argsort(value, index_view, n)
    elif reverse and not parallel:
        _argsort_reverse(value, index_view, n)
    elif not reverse and parallel:
        _argsort_parallel(value, index_view, n)
    else:
        _argsort_reverse_parallel(value, index_view, n)

# cdef void __count_inner(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t h, const DLONGINTTYPE_t size) noexcept nogil:
#
#     cdef:
#         int rs, ls, k, hh
#
#     hh = h + size
#     for rs in range(h, hh):
#         k = h
#         for ls in range(h, rs):
#             if value[rs] >= value[ls]:
#                 k += 1
#         for ls in range(rs + 1, hh):
#             if value[rs] > value[ls]:
#                 k += 1
#         index_view[k] = rs
#
#
# cpdef void count_sort(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, bool parallel = False):
#
#     cdef:
#         int h, hh, size, n, k, ls, rs, stop
#
#     size = 1024
#     n = value.shape[0]
#     stop = n - size
#     if parallel:
#         for h in prange(0, stop, size, nogil=True):
#             __count_inner(value, index_view, h, size)
#     else:
#         for h in range(0, stop, size):
#             __count_inner(value, index_view, h, size)
#
#     h += size
#     for rs in range(h, n):
#         k = h
#         for ls in range(h, rs):
#             if value[rs] >= value[ls]:
#                 k += 1
#         for ls in range(rs + 1, n):
#             if value[rs] > value[ls]:
#                 k += 1
#         index_view[k] = rs
