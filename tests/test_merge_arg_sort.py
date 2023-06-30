from core_math.merge_arg_sort import merge_arg_sort


def test_first_set(merge_sort_first_set_index, merge_sort_first_set_data):
    sort_index = merge_arg_sort(merge_sort_first_set_data)
    assert all(merge_sort_first_set_index == sort_index)


def test_second_set(merge_sort_second_set_index, merge_sort_second_set_data):
    sort_index = merge_arg_sort(merge_sort_second_set_data)
    assert all(merge_sort_second_set_index == sort_index)
