# distutils: language=c++
#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport numpy as np

from libcpp.vector cimport vector

cpdef vector[int] merge_arg_sort(vector[int] value)
cpdef vector[int] extremal_max(vector[int] index, int eps)
cpdef vector[int] extremal_min(vector[int] index, int eps)
