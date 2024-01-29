import pytest
from wsknn.model.wsknn import WSKNN
from wsknn.utils.errors import InvalidDimensionsError, InvalidTimestampError


def test_wsknn_invalid_dimensions_exception():

    sessions = {
        'A': [[1, 2, 3, 4]],
        'B': [[4, 3, 4, 3, 3]],
        'C': [[10, 2, 1, 1, 1]]
    }

    items = {
        '0': [0]
    }

    model = WSKNN()

    with pytest.raises(InvalidDimensionsError):
        model.fit(sessions=sessions, items=items)


def test_wsknn_invalid_type_exception():
    sessions = {
        'A': [1, 2, 3],
        'B': [4, 3],
        'C': [10, 2, 1]
    }

    items = {
        '0': [0]
    }

    model = WSKNN()

    with pytest.raises(TypeError):
        model.fit(sessions=sessions, items=items)


def test_wsknn_invalid_datatypes_exception():
    sessions = {
        'A': [[1, 2, 3], [10, 20, 30]]
    }

    items = {
        '1': [['A'], [10]]
    }

    model = WSKNN()

    with pytest.raises(TypeError):
        model.fit(sessions=sessions, items=items)


def test_wsknn_timestamp_type_exception():
    sessions = {
        'A': [[1, 2, 3],
              ['10', '20', '30']]
    }

    items = {
        '0': [
            ['A'],
            ['2022-02-22']
        ]

    }

    model = WSKNN()

    with pytest.raises(InvalidTimestampError):
        model.fit(sessions=sessions, items=items)


def test_wsknn_flow1():
    # Type of sessions is str, type of items is int
    sessions = {
        'a': [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ],
        'b': [
            [2, 3, 4, 5],
            [10, 11, 12, 13]
        ]
    }

    items = {
        1: [['a'], [1]],
        2: [['a', 'b'], [2]],
        3: [['a', 'b'], [3]],
        4: [['a', 'b'], [4]],
        5: [['a', 'b'], [5]]
    }

    some_sessions = [[[1], [100]], [[2, 3], [200, 300]]]

    expected_recommendations = [[(2, 2.0), (3, 2.0), (4, 2.0), (5, 2.0)],
                                [(4, 2.5), (5, 2.5), (1, 1.25)]]

    model = WSKNN(return_events_from_session=False)
    model.fit(sessions, items)

    for idx, sess in enumerate(some_sessions):
        recomms = model.recommend(sess)
        assert recomms == expected_recommendations[idx]


def test_wsknn_flow2():
    # Type of sessions is int, type of items is str
    sessions = {
        0: [
            ['1', '2', '3', '4', '5'],
            [1, 2, 3, 4, 5]
        ],
        1: [
            ['2', '3', '4', '5'],
            [10, 11, 12, 13]
        ]
    }

    items = {
        '1': [[0], [1]],
        '2': [[0, 1], [2]],
        '3': [[0, 1], [3]],
        '4': [[0, 1], [4]],
        '5': [[0, 1], [5]]
    }

    some_sessions = [[['1'], [100]], [['2', '3'], [200, 300]]]
    expected_recommendations = [
        [('2', 2.0), ('3', 2.0), ('4', 2.0), ('5', 2.0)],
        [('4', 2.5), ('5', 2.5), ('1', 1.25)]
    ]

    model = WSKNN(return_events_from_session=False)
    model.fit(sessions, items)

    for idx, sess in enumerate(some_sessions):
        recomms = model.recommend(sess)
        assert recomms == expected_recommendations[idx]


def test_wsknn_with_weights():
    # Type of sessions is int, type of items is str
    sessions = {
        0: [
            ['1', '2', '3', '4', '5'],
            [1, 2, 3, 4, 5],
            [0.9, 0.9, 0.9, 0.1, 0.2]
        ],
        1: [
            ['2', '3', '4', '5'],
            [10, 11, 12, 13],
            [0.3, 0.4, 0.5, 0.1]
        ]
    }

    items = {
        '1': [[0], [1]],
        '2': [[0, 1], [2]],
        '3': [[0, 1], [3]],
        '4': [[0, 1], [4]],
        '5': [[0, 1], [5]]
    }

    some_sessions = [[['1'], [100], [10]], [['2', '3'], [200, 300], [5, 5]]]
    expected_recommendations = [
        [('2', 2.0), ('3', 2.0), ('4', 2.0), ('5', 2.0)],
        [('4', 2.5), ('5', 2.5), ('1', 1.25)]
    ]

    model = WSKNN(return_events_from_session=False,
                  sampling_strategy='weighted_events',
                  sampling_event_weights_index=2)

    model.fit(sessions, items)

    for idx, sess in enumerate(some_sessions):
        recomms = model.recommend(sess)
        assert recomms == expected_recommendations[idx]


def test_wsknn_no_items():
    # Type of sessions is str, type of items is int
    sessions = {
        'a': [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ],
        'b': [
            [2, 3, 4, 5],
            [10, 11, 12, 13]
        ]
    }

    some_sessions = [[[1], [100]], [[2, 3], [200, 300]]]

    expected_recommendations = [[(2, 2.0), (3, 2.0), (4, 2.0), (5, 2.0)],
                                [(4, 2.5), (5, 2.5), (1, 1.25)]]

    model = WSKNN(return_events_from_session=False)
    model.fit(sessions)

    for idx, sess in enumerate(some_sessions):
        recomms = model.recommend(sess)
        assert recomms == expected_recommendations[idx]


def test_batch_recommendations():
    sessions = {
        'a': [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ],
        'b': [
            [2, 3, 4, 5],
            [10, 11, 12, 13]
        ]
    }

    some_sessions = {
        'x': [[1], [100]],
        'y': [[2, 3], [200, 300]]
    }

    expected_recommendations = {
        'x': [(2, 2.0), (3, 2.0), (4, 2.0), (5, 2.0)],
        'y': [(4, 2.5), (5, 2.5), (1, 1.25)]
    }

    model = WSKNN(return_events_from_session=False)
    model.fit(sessions)

    recs = model.recommend(some_sessions)
    for _k, _v in recs.items():
        assert expected_recommendations[_k] == _v


def test_wsknn_random_sampling():
    # Type of sessions is str, type of items is int
    sessions = {
        'a': [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ],
        'b': [
            [2, 3, 4, 5],
            [10, 11, 12, 13]
        ]
    }

    items = {
        1: [['a'], [1]],
        2: [['a', 'b'], [2]],
        3: [['a', 'b'], [3]],
        4: [['a', 'b'], [4]],
        5: [['a', 'b'], [5]]
    }

    model = WSKNN(return_events_from_session=False,
                  sampling_strategy='random')
    model.fit(sessions, items)

    assert isinstance(model, WSKNN)
