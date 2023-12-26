import numpy as np

from core.extremum import min_extremum, max_extremum
from core.matches_extremum import MatchesOnInputArray
from core.sort import argsort
from core.trend import CombinedTrendDetection

np.random.seed(12222)


# np.random.seed(123)


def search_split_main_extr(index: np.ndarray[np.uint32], coincident: int = 1, eps: int = 1):
    n = len(index)
    diff_for_minimal = np.empty_like(index, dtype=np.uint32)
    diff_for_maximal = np.empty_like(index, dtype=np.uint32)
    marker_diff_for_minimal = np.zeros((n + 1,), dtype=np.int32)
    marker_diff_for_maximal = np.zeros((n + 1,), dtype=np.int32)

    # region Вычисление разницы между индексами

    for i in range(n):
        min_diff_for_minimal, min_diff_for_maximal = n, n

        # min
        for j in range(1, i + 1):
            diff = abs(index[i] - index[i - j])
            if diff < min_diff_for_minimal:
                min_diff_for_minimal = diff
            if min_diff_for_minimal <= eps:
                break

        diff_for_minimal[i] = min_diff_for_minimal
        marker_diff_for_minimal[min_diff_for_minimal] += 1

        # max
        for j in range(1, (n - i)):
            diff = abs(index[i] - index[i + j])
            if diff < min_diff_for_maximal:
                min_diff_for_maximal = diff
            if min_diff_for_maximal <= eps:
                break
        diff_for_maximal[i] = min_diff_for_maximal
        marker_diff_for_maximal[min_diff_for_maximal] += 1

    # endregion Вычисление разницы между индексами

    # region Поиск главных локальных минимумов по заданному числу совпадений

    count_zero = 0
    last_non_zero_index = eps
    for i in range(eps + 1, n):
        if count_zero >= coincident - 1:
            eps_min = last_non_zero_index + coincident - 1
            break
        if marker_diff_for_minimal[i] == 0:
            count_zero += 1
        else:
            count_zero = 0
            last_non_zero_index = i
    else:
        eps_min = last_non_zero_index + coincident - 1

    if eps_min >= diff_for_minimal[0]:
        extr_min = np.array([index[0]])
    else:
        k = 0
        extr_min = np.empty_like(index)
        for i in range(n):
            if diff_for_minimal[i] > eps_min:
                extr_min[k] = index[i]
                k += 1
        extr_min = extr_min[:k]

    # endregion Поиск главных локальных минимумов по заданному числу совпадений

    # region Поиск главных локальных максимумов по заданному числу совпадений

    count_zero = 0
    last_non_zero_index = eps
    for i in range(eps + 1, n):
        if count_zero >= coincident - 1:
            eps_max = last_non_zero_index + coincident - 1
            break
        if marker_diff_for_maximal[i] == 0:
            count_zero += 1
        else:
            count_zero = 0
            last_non_zero_index = i
    else:
        eps_max = last_non_zero_index + coincident - 1

    if eps_max >= diff_for_maximal[n - 1]:
        extr_max = np.array([index[n - 1]])
    else:
        k = 0
        extr_max = np.empty_like(index)
        for i in range(n):
            if diff_for_maximal[i] > eps_max:
                extr_max[k] = index[i]
                k += 1
        extr_max = extr_max[:k]

    # endregion Поиск главных локальных максимумов по заданному числу совпадений

    return np.sort(extr_min), np.sort(extr_max), eps_min, eps_max


def merge(
        extr_min_index: np.ndarray[np.uint32],
        extr_max_index: np.ndarray[np.uint32],
):
    """
    Слияние двух массивов индексов локальных минимумов и локальных максимумов в один общий с дополнительной фильтрацией
    Если происходит слияние в общий массив подряд двух и более элементов из массива с локальными
    минимумами (максимумами) то соответственно будет добавлен индекс наиболее существенного экстремума из этого
    интервала (в смысле значения) наименьший (наибольший)

    :param extr_min_index: Массив индексов локальных минимумов в исходном порядке следования
    :param extr_max_index: Массив индексов локальных максимумов в исходном порядке следования
    :return: Кортеж из трех массивов: итоговый слитый массив индексов экстремумов;
    отфильтрованный массив индексов локальных минимумов; отфильтрованный массив индексов локальных максимумов
    """

    extr = []
    i = j = status = 0

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
            extr.append(extr_min_index[i])
            i += 1
        else:
            extr.append(extr_max_index[j])
            j += 1

    return np.array(extr)


def main():
    size = 13
    # closes = np.random.uniform(10, 20, size)
    closes = np.array([np.random.randint(10, 50) for _ in range(size)])
    print(closes)
    matches = MatchesOnInputArray()

    eps = 3
    step = 3
    coincident = 1
    extr_min_index = []
    extr_max_index = []
    print()
    for i in range(0, len(closes), step):
        close = closes[i: i + step]
        index = argsort(close)
        print(f"{i}: {close=} {index=}")
        _extr_min_index, _min_eps = matches(
            extremum=min_extremum, index=index, max_coincident=coincident, eps=eps
        )

        # region Gluing Minimals Sub-intervals

        if _extr_min_index[0] - _min_eps < 0:
            for j in range(i - (i - abs(_extr_min_index[0] + i - _min_eps)), i + 1):
                if closes[j] < closes[_extr_min_index[0] + i]:
                    _extr_min_index = _extr_min_index[1:]
                    break

        if len(_extr_min_index) and _extr_min_index[-1] + _min_eps > len(close) - 1:
            for j in range(
                    i + len(close) - 1,
                    len(closes) - abs(len(closes) - 1 - (i + _extr_min_index[-1] + _min_eps)),
            ):
                if closes[j] < closes[_extr_min_index[-1] + i]:
                    _extr_min_index = _extr_min_index[:-1]
                    break

        # endregion Gluing Minimals Sub-intervals

        extr_min_index.extend(_extr_min_index + i)

        _extr_max_index, _max_eps = matches(
            extremum=max_extremum, index=index, max_coincident=coincident, eps=eps
        )

        # region Gluing Maximals Sub-intervals

        if _extr_max_index[0] - _max_eps < 0:
            for j in range(i - (i - abs(_extr_max_index[0] + i - _max_eps)), i + 1):
                if closes[j] > closes[_extr_max_index[0] + i]:
                    _extr_max_index = _extr_max_index[1:]
                    break

        if len(_extr_max_index) and _extr_max_index[-1] + _max_eps > len(close) - 1:
            for j in range(
                    i + len(close) - 1,
                    len(closes) - abs(len(closes) - (i + _extr_max_index[-1] + _max_eps)),
            ):
                if closes[j] > closes[_extr_max_index[-1] + i]:
                    _extr_max_index = _extr_max_index[:-1]
                    break

        # endregion Gluing Maximals Sub-intervals

        extr_max_index.extend(_extr_max_index + i)

    combined_indexes = np.sort(np.hstack([extr_min_index, extr_max_index]))
    combined_values = closes[combined_indexes]

    trend = CombinedTrendDetection(closes, size, matches)
    trend.search_extremum(num_coincident=coincident, start_eps=eps)

    assert len(trend.get_combined_indexes()) == len(combined_indexes) and np.all(
        trend.get_combined_indexes() == combined_indexes
    ), f"""
        {len(trend.get_combined_indexes())}\n
        {len(combined_indexes)}\n
        {trend.get_combined_indexes()=}\n
        {combined_indexes=}
        """

    assert len(trend.get_combined_values()) == len(combined_values) and np.all(
        trend.get_combined_values() == combined_values
    ), f"""
        {len(trend.get_combined_values())}\n
        {len(combined_values)}\n
        {trend.get_combined_values()=}\n
        {combined_values=}
        """

    # plt.plot(closes)
    # plt.plot(combined_indexes, combined_values)
    # plt.show()


def main_new():
    size = 130
    # closes = np.random.uniform(10, 20, size)
    closes = np.array([np.random.randint(10, 50) for _ in range(size)])
    matches = MatchesOnInputArray()

    eps = 37
    step = 10
    coincident = 1
    extr_min_index = []
    extr_max_index = []
    print()
    for i in range(0, len(closes), step):
        close = closes[i: i + step]
        index = argsort(close)
        _extr_min_index, _min_eps = matches(
            extremum=min_extremum, index=index, max_coincident=coincident, eps=eps
        )

        # region Gluing Minimals Sub-intervals

        if _extr_min_index[0] - _min_eps < 0:
            for j in range(abs(_extr_min_index[0] + i - _min_eps), i):
                if closes[j] <= closes[_extr_min_index[0] + i]:
                    _extr_min_index = _extr_min_index[1:]
                    break

        if len(_extr_min_index) and _extr_min_index[-1] + _min_eps >= step:
            for j in range(
                    i + step, len(closes) - abs(len(closes) - 1 - (_extr_min_index[-1] + i + _min_eps))
            ):
                if closes[j] < closes[_extr_min_index[-1] + i]:
                    _extr_min_index = _extr_min_index[:-1]
                    break

        # endregion Gluing Minimals Sub-intervals

        extr_min_index.extend(_extr_min_index + i)

        _extr_max_index, _max_eps = matches(
            extremum=max_extremum, index=index, max_coincident=coincident, eps=eps
        )

        # region Gluing Maximals Sub-intervals

        if _extr_max_index[0] - _max_eps < 0:
            for j in range(abs(_extr_max_index[0] + i - _max_eps), i):
                if closes[j] >= closes[_extr_max_index[0] + i]:
                    _extr_max_index = _extr_max_index[1:]
                    break

        if len(_extr_max_index) and _extr_max_index[-1] + _max_eps >= step:
            for j in range(
                    i + step, len(closes) - abs(len(closes) - 1 - (_extr_max_index[-1] + i + _max_eps))
            ):
                if closes[j] > closes[_extr_max_index[-1] + i]:
                    _extr_max_index = _extr_max_index[:-1]
                    break

        # endregion Gluing Maximals Sub-intervals

        extr_max_index.extend(_extr_max_index + i)

    combined_indexes = np.sort(np.hstack([extr_min_index, extr_max_index]))
    combined_values = closes[combined_indexes]

    trend = CombinedTrendDetection(closes, size, matches)
    trend.search_extremum(num_coincident=coincident, start_eps=eps)

    assert len(trend.get_combined_indexes()) == len(combined_indexes) and np.all(
        trend.get_combined_indexes() == combined_indexes
    ), f"""
        {len(trend.get_combined_indexes())}\n
        {len(combined_indexes)}\n
        {trend.get_combined_indexes()=}\n
        {combined_indexes=}
        """

    assert len(trend.get_combined_values()) == len(combined_values) and np.all(
        trend.get_combined_values() == combined_values
    ), f"""
        {len(trend.get_combined_values())}\n
        {len(combined_values)}\n
        {trend.get_combined_values()=}\n
        {combined_values=}
        """

    # plt.plot(closes)
    # plt.plot(combined_indexes, combined_values)
    # plt.show()


def main_new_new():
    size = 13
    # closes = np.random.uniform(10, 20, size)
    closes = np.array([np.random.randint(10, 50) for _ in range(size)])
    matches = MatchesOnInputArray()

    eps = 1
    step = 3
    coincident = 1
    # extr_min_index = []
    # extr_max_index = []
    # for i in range(0, len(closes), step):
    #     close = closes[i: i + step]
    #     index = argsort(close)
    #     _extr_min_index, _min_eps = matches(
    #         extremum=min_extremum, index=index, max_coincident=coincident, eps=eps
    #     )
    #
    #     # region Gluing Minimals Sub-intervals
    #
    #     if _extr_min_index[0] - _min_eps < 0:
    #         for j in range(i - 1, _extr_min_index[0] + i - _min_eps - 1, -1):
    #             if j < 0:
    #                 break
    #
    #             if closes[j] <= closes[_extr_min_index[0] + i]:
    #                 _extr_min_index = _extr_min_index[1:]
    #                 break
    #
    #     if len(_extr_min_index) and _extr_min_index[-1] + _min_eps >= step:
    #         for j in range(i + step, _extr_min_index[-1] + i + _min_eps + 1):
    #             if j >= len(closes):
    #                 break
    #
    #             if closes[j] < closes[_extr_min_index[-1] + i]:
    #                 _extr_min_index = _extr_min_index[:-1]
    #                 break
    #
    #     # endregion Gluing Minimals Sub-intervals
    #
    #     extr_min_index.extend(_extr_min_index + i)
    #     _extr_max_index, _max_eps = matches(
    #         extremum=max_extremum, index=index, max_coincident=coincident, eps=eps
    #     )
    #
    #     # region Gluing Maximals Sub-intervals
    #
    #     if _extr_max_index[0] - _max_eps < 0:
    #         for j in range(i - 1, _extr_max_index[0] + i - _max_eps - 1, -1):
    #             if j < 0:
    #                 break
    #
    #             if closes[j] > closes[_extr_max_index[0] + i]:
    #                 _extr_max_index = _extr_max_index[1:]
    #                 break
    #
    #     if len(_extr_max_index) and _extr_max_index[-1] + _max_eps >= step:
    #         for j in range(i + step, _extr_max_index[-1] + i + _max_eps + 1):
    #             if j >= len(closes):
    #                 break
    #
    #             if closes[j] >= closes[_extr_max_index[-1] + i]:
    #                 _extr_max_index = _extr_max_index[:-1]
    #                 break
    #
    #     # endregion Gluing Maximals Sub-intervals
    #
    #     extr_max_index.extend(_extr_max_index + i)
    #
    # combined_indexes = np.sort(np.hstack([extr_min_index, extr_max_index]))
    # combined_values = closes[combined_indexes]
    # print(np.sort(extr_min_index))
    # print(np.sort(extr_max_index))
    # print(combined_indexes)

    trend = CombinedTrendDetection(closes, size, matches)
    trend.search_extremum(num_coincident=coincident, start_eps=eps)
    print(trend.get_combined_indexes())
    print(trend.get_min_indexes())
    print(trend.get_max_indexes())
    # print()
    # trend.search_extremum(num_coincident=coincident, start_eps=2)
    # print(trend.get_combined_indexes())
    # print(trend.get_min_indexes())
    # print(trend.get_max_indexes())
    # print()
    # print()
    # trend.search_extremum(num_coincident=coincident, start_eps=eps)
    # print(trend.get_combined_indexes())
    # print(trend.get_min_indexes())
    # print(trend.get_max_indexes())
    # print()
    # trend.search_extremum(num_coincident=coincident, start_eps=eps)
    # print(trend.get_combined_indexes())
    # print(trend.get_min_indexes())
    # print(trend.get_max_indexes())
    # print()
    # assert len(trend.get_combined_indexes()) == len(combined_indexes) and np.all(
    #     trend.get_combined_indexes() == combined_indexes
    # ), f"""
    #     {len(trend.get_combined_indexes())}\n
    #     {len(combined_indexes)}\n
    #     {trend.get_combined_indexes()=}\n
    #     {combined_indexes=}
    #     """
    #
    # assert len(trend.get_combined_values()) == len(combined_values) and np.all(
    #     trend.get_combined_values() == combined_values
    # ), f"""
    #     {len(trend.get_combined_values())}\n
    #     {len(combined_values)}\n
    #     {trend.get_combined_values()=}\n
    #     {combined_values=}
    #     """


if __name__ == "__main__":
    main_new_new()
