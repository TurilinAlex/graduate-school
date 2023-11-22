# distutils: language=c++
#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport numpy as np

from libcpp.vector cimport vector

cpdef vector[int] merge_arg_sort(vector[int] value):
    cdef int n = value.size()
    cdef vector[int] index_temp, index
    index_temp.resize(n, 0)
    index.resize(n, 0)

    cdef int h = 1
    cdef int status = 0

    cdef int i, j, jj, k, ls, le, rs, re
    for i in range(n):
        index[i] = i

    while h <= n:
        jj = 0
        while jj < n - h:
            ls = jj
            le = jj + h - 1
            rs = jj + h
            re = jj + 2 * h - 1 if jj + 2 * h - 1 < n - 1 else n - 1
            for k in range(ls, re + 1):
                index_temp[k] = index[k]
            i = ls
            j = rs
            for k in range(ls, re + 1):
                if i <= le and j <= re:
                    if value[index_temp[j]] == value[index_temp[i]]:
                        status = 0
                    if value[index_temp[j]] < value[index_temp[i]]:
                        status = -1
                    if value[index_temp[j]] > value[index_temp[i]]:
                        status = 1
                if i > le: status = -1
                if j > re: status = 1

                if status >= 0:
                    index[k] = index_temp[i]
                    i += 1
                else:
                    index[k] = index_temp[j]
                    j += 1
            jj += 2 * h
        h *= 2
    return index

cpdef vector[int] extremal_max(vector[int] index, int eps):
    cdef vector[int] extreme_max
    cdef int n = index.size()
    cdef int k, l
    for k in range(n):
        for l in range(1, (n - k)):
            if abs(index[k] - index[k + l]) <= eps:
                break
        else:
            extreme_max.push_back(index[k])
    return extreme_max

cpdef vector[int] extremal_min(vector[int] index, int eps):
    cdef vector[int] extreme_max
    cdef int n = index.size()
    cdef int k, l
    for k in range(n):
        for l in range(1, k + 1):
            if abs(index[k] - index[k - l]) <= eps:
                break
        else:
            extreme_max.push_back(index[k])
    return extreme_max
