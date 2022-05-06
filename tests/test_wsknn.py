import pytest
from wsknn.model.wsknn import WSKNN
from wsknn.utils.errors import InvalidDimensionsError, InvalidTimestampError


def test_vsknn_invalid_dimensions_exception():

    sessions = {
        'A': [1, 2, 3, 4],
        'B': [4, 3, 4, 3, 3],
        'C': [10, 2, 1, 1, 1]
    }

    items = {
        '0': 0
    }

    model = WSKNN()

    with pytest.raises(InvalidDimensionsError):
        model.fit(sessions=sessions, items=items)


def test_vsknn_invalid_type_exception():
    sessions = {
        'A': [1, 2, 3],
        'B': [4, 3],
        'C': [10, 2, 1]
    }

    items = {
        '0': 0
    }

    model = WSKNN()

    with pytest.raises(TypeError):
        model.fit(sessions=sessions, items=items)

def test_vsknn_timestamp_type_exception():
    sessions = {
        'A': [[1, 2, 3],
              [10, 20, 30]]
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

def test_vsknn_flow1():
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

    usessions = {'user1': [[1], [100]],
                'user2': [[2, 3], [200, 300]]}

    model = WSKNN()
    model.fit(sessions, items)

    predictions = model.predict(usessions, number_of_recommendations=10, number_of_closest_neighbors=10)
    expected_predictions = {
        'user1': [(2, 2.0), (3, 2.0), (4, 2.0), (5, 2.0)],
        'user2': [(4, 2.5), (5, 2.5), (1, 1.25)]
    }

    assert predictions == expected_predictions


def test_vsknn_flow2():
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

    usessions = {'user1': [['1'], [100]],
                'user2': [['2', '3'], [200, 300]]}

    model = WSKNN()
    model.fit(sessions, items)

    predictions = model.predict(usessions, number_of_recommendations=10, number_of_closest_neighbors=10)
    expected_predictions = {
        'user1': [('2', 2.0), ('3', 2.0), ('4', 2.0), ('5', 2.0)],
        'user2': [('4', 2.5), ('5', 2.5), ('1', 1.25)]
    }

    assert predictions == expected_predictions


def test_vsknn_flow3():
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

    list_sessions = [
        [['1'], [100]],
        [['2', '3'], [200, 300]]
        ]

    model = WSKNN()
    model.fit(sessions, items)

    predictions = model.predict(list_sessions, number_of_recommendations=10, number_of_closest_neighbors=10)
    expected_predictions = {
        0: [('2', 2.0), ('3', 2.0), ('4', 2.0), ('5', 2.0)],
        1: [('4', 2.5), ('5', 2.5), ('1', 1.25)]
    }

    assert predictions == expected_predictions
