from core_math.extremum import min_extremum


def test_min_extremum_firs_set(merge_sort_first_set_index):
    min_index = min_extremum(index=merge_sort_first_set_index, eps=1)
    assert [2, 8] == min_index
