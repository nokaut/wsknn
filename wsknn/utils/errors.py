from typing import Union


def check_data_dimension(record: list, dimensions: int) -> bool:
    """
    The method checks if the dimension of a record is equal to the --dimensions-- parameter.

    Parameters
    ----------
    record : list
             Nested iterable.

    dimensions : int
                 Number of nested dimensions.

    Returns
    -------
    : bool
        True - dimensions are equal to a number of the nested sequences. False otherwise.
    """
    return len(record) >= dimensions


def check_numeric_type_instance(value) -> bool:
    """Method checks if value is int or float

    Parameters
    ----------
    value : Expected Numeric type

    Returns
    -------
    : bool
        True - value is numeric, False - value is other than the number
    """
    if isinstance(value, str):
        return False
    else:
        try:
            _ = int(value)
        except Exception as _e:
            return False
        return True


class InvalidDimensionsError(Exception):
    """
    The exception is raised when the dimension of the input session-items map or input item-sessions map is not valid.

    Attributes
    ----------
    set_type : str
               Item-sessions map or session-items map.

    expected_dimensions : int
    """

    def __init__(self, set_type: str, expected_dimensions: Union[int, list]):
        self.set_type = set_type
        self.expected_dimensions = f'{expected_dimensions}'

    def __str__(self):
        msg = (f'Expected number(s) of nested sequences for {self.set_type} '
               f'must be greater than {self.expected_dimensions}')
        return msg


class InvalidTimestampError(Exception):
    """
    The exception is raised when the timestamp is not a numeric value.
    """

    def __init__(self, timestamp_sample):
        self.timestamp_type = type(timestamp_sample)

    def __str__(self):
        msg = (f'Expected timestamp type should be int or float, '
               f'got {self.timestamp_type} instead.')
        return msg


class TooShortSessionException(Exception):
    """
    The exception is raised by the Mean Reciprocal Rank metrics if session length is shorter than the
    number of top n recommendations."""

    def __init__(self, session_length: int, top_n: int):
        self.s_length = session_length
        self.topn = top_n
        self.msg = f'Given session length {self.s_length} is shorter than the top n recommendations {self.topn}.\n' \
                   f'It is not possible to make a valid evaluation.'

    def __str__(self):
        return self.msg
