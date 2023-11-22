import numpy as np

from core.sort import argsort


def test_first_set(merge_sort_first_set_index, merge_sort_first_set_data):
    sort_index = argsort(merge_sort_first_set_data)
    assert np.all(merge_sort_first_set_index == sort_index)


def test_second_set(merge_sort_second_set_index, merge_sort_second_set_data):
    sort_index = argsort(merge_sort_second_set_data)
    assert np.all(merge_sort_second_set_index == sort_index)
