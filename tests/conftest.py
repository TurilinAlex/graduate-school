import numpy as np
import pytest


@pytest.fixture
def merge_sort_first_set_data() -> np.ndarray[np.float32]:
    return np.array([0.94, -1.04, -42.24, 12.35, 23.24, 23.24, 0.94, 0.025, 0.0])


@pytest.fixture
def merge_sort_first_set_index() -> np.ndarray[np.uint32]:
    return np.array([2, 1, 8, 7, 0, 6, 3, 4, 5])


@pytest.fixture
def merge_sort_second_set_data() -> np.ndarray[np.float32]:
    return np.array([0.0, -1.0, 4.0, -1.0, 0.0, 0.0, -6.0, 2.0, 6.0])


@pytest.fixture
def merge_sort_second_set_index() -> np.ndarray[np.uint32]:
    return np.array([6, 1, 3, 0, 4, 5, 7, 2, 8])
