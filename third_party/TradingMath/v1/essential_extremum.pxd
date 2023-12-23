from libcpp cimport bool

from ..typing cimport DLONGINTTYPE_t


cdef void __diff_between_minimals_indexes(const DLONGINTTYPE_t[::1] index, DLONGINTTYPE_t[::1] min_extr_diff_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil

cdef void __diff_between_minimals_indexes_parallel(const DLONGINTTYPE_t[::1] index, DLONGINTTYPE_t[::1] min_extr_diff_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil

cdef void __diff_between_maximals_indexes(const DLONGINTTYPE_t[::1] index, DLONGINTTYPE_t[::1] max_extr_diff_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil

cdef void __diff_between_maximals_indexes_parallel(const DLONGINTTYPE_t[::1] index, DLONGINTTYPE_t[::1] max_extr_diff_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil

cdef void __diff_between_all_indexes(const DLONGINTTYPE_t[::1] index, DLONGINTTYPE_t[::1] min_extr_diff_view, DLONGINTTYPE_t[::1] max_extr_diff_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t eps) noexcept nogil

cdef void __diff_between_all_indexes_parallel(const DLONGINTTYPE_t[::1] index, DLONGINTTYPE_t[::1] min_extr_diff_view, DLONGINTTYPE_t[::1] max_extr_diff_view, const DLONGINTTYPE_t &n, const DLONGINTTYPE_t &eps) noexcept nogil

cpdef localize_minimals(const DLONGINTTYPE_t[::1] index, const DLONGINTTYPE_t coincident, const DLONGINTTYPE_t eps, bool parallel = *)

cpdef localize_maximals(const DLONGINTTYPE_t[::1] index, const DLONGINTTYPE_t coincident, const DLONGINTTYPE_t eps, bool parallel = *)

cpdef localize_extremes(const DLONGINTTYPE_t[::1] index, const DLONGINTTYPE_t coincident, const DLONGINTTYPE_t eps, bool parallel = *)
