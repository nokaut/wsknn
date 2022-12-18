import numpy as np
from wsknn.weighting.session_weighting import linear_session_score, log_session_score, quadratic_session_score
from wsknn.weighting.item_weighting import linear_item_score, inv_pos_item_score, quadratic_item_score, log_item_score


def test_session_log10():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(0, sequence_size, 1)
    output = [log_session_score(v, sequence_size) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0


def test_session_lin():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(0, sequence_size, 1)
    output = [linear_session_score(v, sequence_size) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0


def test_session_quadratic():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(0, sequence_size, 1)
    output = [quadratic_session_score(v, sequence_size) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0


def test_item_linear():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(1, sequence_size, 1)
    output = [linear_item_score(v) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0


def test_item_log():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(1, sequence_size, 1)
    output = [log_item_score(v) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0


def test_item_inv():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(1, sequence_size, 1)
    output = [inv_pos_item_score(v) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0


def test_item_quadratic():
    sequence_size = np.random.randint(0, 10000)
    vals_range = np.arange(1, sequence_size, 1)
    output = [quadratic_item_score(v) for v in vals_range]
    for res in output:
        assert res <= 1
        assert res >= 0
