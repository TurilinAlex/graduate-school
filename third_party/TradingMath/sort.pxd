from libcpp cimport bool

from .typing cimport DALLTYPE_t, DLONGINTTYPE_t

cpdef void argsort(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, bool reverse = *, bool parallel = *)

cpdef void _argsort_reverse_parallel(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n)

cpdef void _argsort_reverse(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n)

cpdef void _argsort_parallel(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n)

cpdef void _argsort(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n)

cdef void __inner_merge_reverse(DALLTYPE_t[::1] value, DLONGINTTYPE_t* index_temp_view, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t hh, const DLONGINTTYPE_t h) noexcept nogil

cdef void __inner_merge(DALLTYPE_t[::1] value, DLONGINTTYPE_t* index_temp_view, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t n, const DLONGINTTYPE_t hh, const DLONGINTTYPE_t h) noexcept nogil

# cdef void __count_inner(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, const DLONGINTTYPE_t h, const DLONGINTTYPE_t size) noexcept nogil

# cpdef void count_sort(DALLTYPE_t[::1] value, DLONGINTTYPE_t[::1] index_view, bool parallel = *)

