from abc import abstractmethod
from typing import Protocol, Callable

import numpy as np

__all__ = [
    "BaseMatchesOnArray",
    "MatchesOnInputArray",
    "MatchesOnRecalculatedArray",
]


class BaseMatchesOnArray(Protocol):
    """
    Этот класс предназначен для выделения массива индексов существенно информативных экстремумов
    по входному массиву индексов, которыми отсортирован массив значений. Под "существенным"
    информативными индексами подразумевается массив который при последовательном увеличении
    значения радиуса локализации требуемое количество раз остался неизменным

    Конкретная реализация будет отличаться и заключаться в классах наследниках в методе ``__call__(...)``
    """

    @staticmethod
    @abstractmethod
    def __call__(
        extremum: Callable[[np.ndarray[np.uint32], int], np.ndarray[np.uint32]],
        index: np.ndarray[np.uint32],
        max_coincident: int = 1,
        eps: int = 1,
    ) -> tuple[np.ndarray[np.uint32], int]:
        """
        Конкретное описание следует смотреть в классе-наследнике

        :param extremum: Оператор выделения экстремумов из последовательности индексов
        :param index: Массив индексов
        :param max_coincident: Требуемое количество последовательно равных массивов экстремумов,
        что бы их считать существенными
        :param eps: Начальный радиус локализации ``eps > 0``
        :return: Возвращает массив индексов существенных экстремумов и радиус локализации при котором он был выделен
        """
        pass


class MatchesOnInputArray(BaseMatchesOnArray):
    @staticmethod
    def __call__(
        extremum: Callable[[np.ndarray[np.uint32], int], np.ndarray[np.uint32]],
        index: np.ndarray[np.uint32],
        max_coincident=1,
        eps: int = 1,
    ) -> tuple[np.ndarray[np.uint32], int]:
        """
        Выделяет экстремумы при увеличении радиусе локализации `всегда` во входном массиве

        :param extremum: Оператор выделения экстремумов из последовательности индексов
        :param index: Массив индексов
        :param max_coincident: Требуемое количество последовательно равных массивов экстремумов,
        что бы их считать существенными
        :param eps: Начальный радиус локализации ``eps > 0``
        :return: Возвращает массив индексов существенных экстремумов и радиус локализации при котором он был выделен
        """

        coincident_num = 1
        extreme = extremum(index, eps)
        while coincident_num < max_coincident:
            eps += 1
            recalculated_extreme = extremum(index, eps)
            if len(extreme) == len(recalculated_extreme):
                coincident_num += 1
            else:
                extreme = recalculated_extreme
                coincident_num = 1
        return np.sort(extreme), eps


class MatchesOnRecalculatedArray(BaseMatchesOnArray):
    @staticmethod
    def __call__(
        extremum: Callable[[np.ndarray[np.uint32], int], np.ndarray[np.uint32]],
        index: np.ndarray[np.uint32],
        max_coincident=1,
        eps: int = 1,
    ):
        """
        Выделяет экстремумы при увеличении радиусе локализации `всегда` из последнего выделенного массива экстремумов

        :param extremum: Оператор выделения экстремумов из последовательности индексов
        :param index: Массив индексов
        :param max_coincident: Требуемое количество последовательно равных массивов экстремумов,
        что бы их считать существенными
        :param eps: Начальный радиус локализации ``eps > 0``
        :return: Возвращает массив индексов существенных экстремумов и радиус локализации при котором он был выделен
        """

        coincident_num = 1
        extreme = extremum(index, eps)
        while coincident_num < max_coincident:
            eps += 1
            recalculated_extreme = extremum(extreme, eps)
            if len(extreme) == len(recalculated_extreme):
                coincident_num += 1
            else:
                extreme = recalculated_extreme
                coincident_num = 1
        return np.sort(extreme), eps
