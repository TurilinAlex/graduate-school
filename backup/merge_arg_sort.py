import numpy as np

__all__ = [
    "merge_arg_sort",
]


def merge_arg_sort(values: np.ndarray[np.float32] | list[float]) -> np.ndarray[np.int32]:
    # sourcery skip: low-code-quality, use-assigned-variable
    n = len(values)
    index_temp = np.zeros((n,), np.int32)
    index = np.arange(n)

    h = 1
    status = 0
    while h <= n:
        for j in range(0, n - h, 2 * h):
            ls = j
            le = j + h - 1
            rs = j + h
            re = min(j + 2 * h - 1, n - 1)
            index_temp[ls:re + 1] = index[ls:re + 1]
            i1 = ls
            j1 = rs
            for k in range(ls, re + 1):

                if i1 <= le and j1 <= re:
                    if values[index_temp[j1]] == values[index_temp[i1]]:
                        status = 0
                    if values[index_temp[j1]] < values[index_temp[i1]]:
                        status = -1
                    if values[index_temp[j1]] > values[index_temp[i1]]:
                        status = 1

                if i1 > le:
                    status = -1
                if j1 > re:
                    status = 1

                if status >= 0:
                    index[k] = index_temp[i1]
                    i1 += 1
                else:
                    index[k] = index_temp[j1]
                    j1 += 1
        h *= 2
    return index
