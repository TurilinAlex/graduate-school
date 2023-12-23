import numpy as np
cimport numpy as np
from libcpp cimport bool

from .typing cimport DLONGINTTYPE_t

__all__ = [
    "localize_minimals",
    "localize_maximals",
]

cdef void __extremal_min(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil

cdef void __extremal_min_parallel(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil

cdef void __extremal_max(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil

cdef void __extremal_max_parallel(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil

cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] localize_maximals(DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t eps, bool parallel = *)

cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] localize_minimals(DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t eps, bool parallel = *)
