from wsknn.utils.calc import is_there_any_common_element, weight_set_pair


def test_weight_set_pair():
    set_a = {1, 3, 5, 7}
    set_b = {0}
    set_c = {1, 0, 9, 3, 8}

    weights_ab = dict()
    weights_ac = {1: 3,
                  3: 12}
    weights_bc = {0: 1.1}

    expected_ab = 0
    expected_ac = 7.5
    expected_bc = 1.1

    assert weight_set_pair(set_a, set_b, weights_ab) == expected_ab
    assert weight_set_pair(set_a, set_c, weights_ac) == expected_ac
    assert weight_set_pair(set_b, set_c, weights_bc) == expected_bc


def test_is_there_any_common_element():
    set_a = {1, 3, 5, 7}
    set_b = {0}
    set_c = {1, 0, 9, 3, 8}

    expected_ab = 0
    expected_bc = 1

    assert is_there_any_common_element(set_a, set_b) == expected_ab
    assert is_there_any_common_element(set_b, set_c) == expected_bc
