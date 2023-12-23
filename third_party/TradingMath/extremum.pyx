# distutils: language=c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, infer_types=True

import numpy as np
cimport numpy as np

from libcpp cimport bool
from cython.parallel cimport prange

from .typing cimport DLONGINTTYPE_t

__all__ = [
    "localize_minimals",
    "localize_maximals",
]


cdef void __extremal_min(
    DLONGINTTYPE_t[:] index,
    DLONGINTTYPE_t[:] extremum_view,
    const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
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
    DLONGINTTYPE_t[:] index,
    DLONGINTTYPE_t[:] extremum_view,
    const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
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

cdef void __extremal_max(
    DLONGINTTYPE_t[:] index,
    DLONGINTTYPE_t[:] extremum_view,
    const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
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
    DLONGINTTYPE_t[:] index,
    DLONGINTTYPE_t[:] extremum_view,
    const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
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

cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] localize_maximals(
    DLONGINTTYPE_t[:] index,
    const DLONGINTTYPE_t eps, bool parallel = False
):

    cdef:
        np.ndarray[DLONGINTTYPE_t, ndim=1] return_extremum, positive
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t[:] extremum_view

    extremum_view = np.empty_like(index, dtype=np.int32)

    if parallel:
        __extremal_max_parallel(index, extremum_view, n, eps)
    else:
        __extremal_max(index, extremum_view, n, eps)

    return_extremum = np.asarray(extremum_view)
    positive = np.extract(return_extremum >= 0, return_extremum)
    return positive


cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] localize_minimals(
    DLONGINTTYPE_t[:] index,
    const DLONGINTTYPE_t eps, bool parallel = False
):

    cdef:
        np.ndarray[DLONGINTTYPE_t, ndim=1] return_extremum, positive
        DLONGINTTYPE_t n = index.shape[0]
        DLONGINTTYPE_t[:] extremum_view

    extremum_view = np.empty_like(index, dtype=np.int32)

    if parallel:
        __extremal_min_parallel(index, extremum_view, n, eps)
    else:
        __extremal_min(index, extremum_view, n, eps)

    return_extremum = np.asarray(extremum_view)
    positive = np.extract(return_extremum >= 0, return_extremum)
    return positive
