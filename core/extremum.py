import numpy as np

__all__ = [
    "min_extremum",
    "max_extremum",
    "merge_extremum",
]


def min_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
    """
    Оператор идентификации экстремумов (локальных минимумов) по массиву индексов с заданным радиусом локализации

    :param index: Входной массив индексов
    :param eps: Радиус локализации ``eps > 0``
    :return: Возвращает массив индексов экстремумов (локальных минимумов) с данным радиусом локализации ``eps``
    """

    n, extreme_min = len(index), []
    for i in range(n):
        for j in range(1, i + 1):
            if abs(index[i] - index[i - j]) <= eps:
                break
        else:
            extreme_min.append(index[i])
    return np.array(extreme_min)


def max_extremum(index: np.ndarray[np.uint32], eps: int) -> np.ndarray[np.uint32]:
    """
    Оператор идентификации экстремумов (локальных максимумов) по массиву индексов с заданным радиусом локализации

    :param index: Входной массив индексов
    :param eps: Радиус локализации ``eps > 0``
    :return: Возвращает массив индексов экстремумов (локальных минимумов) с данным радиусом локализации ``eps``
    """

    n, extreme_max = len(index), []
    for i in range(n):
        for j in range(1, (n - i)):
            if abs(index[i] - index[i + j]) <= eps:
                break
        else:
            extreme_max.append(index[i])
    return np.array(extreme_max)


def merge_extremum(
    extr_min_index: np.ndarray[np.uint32],
    extr_max_index: np.ndarray[np.uint32],
    values: np.ndarray[np.float32],
):
    """
    Слияние двух массивов индексов локальных минимумов и локальных максимумов в один общий с дополнительной фильтрацией
    Если происходит слияние в общий массив подряд двух и более элементов из массива с локальными
    минимумами (максимумами) то соответственно будет добавлен индекс наиболее существенного экстремума из этого
    интервала (в смысле значения) наименьший (наибольший)

    :param extr_min_index: Массив индексов локальных минимумов в исходном порядке следования
    :param extr_max_index: Массив индексов локальных максимумов в исходном порядке следования
    :param values: Исходный массив значений
    :return: Кортеж из трех массивов: итоговый слитый массив индексов экстремумов;
    отфильтрованный массив индексов локальных минимумов; отфильтрованный массив индексов локальных максимумов
    """

    extr, extr_min_new, extr_max_new = [], [], []
    i = j = 0
    status = 0
    i_min = j_max = None
    min_over, max_over = max(values) + 1, min(values) - 1
    value_min, value_max = min_over, max_over

    while i + j < len(extr_min_index) + len(extr_max_index):
        if i < len(extr_min_index) and j < len(extr_max_index):
            if extr_max_index[j] < extr_min_index[i]:
                status = -1
            if extr_max_index[j] > extr_min_index[i]:
                status = 1
            if extr_max_index[j] == extr_min_index[i]:
                status = 0

        if i >= len(extr_min_index):
            status = -1
        if j >= len(extr_max_index):
            status = 1

        if status >= 0:
            if values[extr_min_index[i]] < value_min:
                value_min = values[extr_min_index[i]]
                i_min = extr_min_index[i]
            if j_max is not None:
                extr_max_new.append(j_max)
                extr.append(j_max)
                j_max = None
            value_max = max_over
            i += 1
        else:
            if values[extr_max_index[j]] >= value_max:
                value_max = values[extr_max_index[j]]
                j_max = extr_max_index[j]
            if i_min is not None:
                extr_min_new.append(i_min)
                extr.append(i_min)
                i_min = None
            value_min = min_over
            j += 1

    if status < 0:
        extr.append(j_max)
        extr_max_new.append(j_max)
    else:
        extr.append(i_min)
        extr_min_new.append(i_min)

    return np.array(extr), np.array(extr_min_new), np.array(extr_max_new)
