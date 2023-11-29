import numpy as np
cimport numpy as np
from libcpp cimport bool

ctypedef fused DALLTYPE_t:
    long double
    long long int
    unsigned long long int

ctypedef long long int DLONGINTTYPE_t


cpdef void argsort(DALLTYPE_t[:] value, DLONGINTTYPE_t[:] index_view, bool reverse = *)
cpdef void __argsort(DALLTYPE_t[:] value, DLONGINTTYPE_t[:] index_view, const DLONGINTTYPE_t n)
cpdef void __argsort_reverse(DALLTYPE_t[:] value, DLONGINTTYPE_t[:] index_view, const DLONGINTTYPE_t n)
cdef void __argsort_ptr(DLONGINTTYPE_t *value, DLONGINTTYPE_t *index_view, const DLONGINTTYPE_t n)
cdef void __argsort_ptr_reverse(DLONGINTTYPE_t *value, DLONGINTTYPE_t *index_view, const DLONGINTTYPE_t n)
cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] extremal_max(DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t eps, bool parallel = *)
cdef void __extremal_max(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil
cdef void __extremal_max_parallel(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil
cpdef np.ndarray[DLONGINTTYPE_t, ndim=1] extremal_min(DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t eps, bool parallel = *)
cdef void __extremal_min(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil
cdef void __extremal_min_parallel(DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] extremum_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil
cpdef search_main_extremum(
    const DLONGINTTYPE_t[:] index, const DLONGINTTYPE_t coincident, const DLONGINTTYPE_t eps, bool parallel = *
)
cdef void __diff_between_indexes(
    const DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] min_extr_diff_view, DLONGINTTYPE_t[:] max_extr_diff_view,
    const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps
) noexcept nogil

cdef void __diff_between_indexes_parallel(
    const DLONGINTTYPE_t[:] index, DLONGINTTYPE_t[:] min_extr_diff_view, DLONGINTTYPE_t[:] max_extr_diff_view,
    const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps
) noexcept nogil
