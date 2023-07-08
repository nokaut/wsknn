from datetime import datetime
from typing import Any, List


def clean_time(times: Any,
               time_to_numeric: bool,
               time_to_datetime: bool,
               datetime_format: str) -> List:
    """
    Function cleans timestamps if they don't have valid data type.

    Parameters
    ----------
    times : List
        The list with timestamps.

    time_to_numeric : bool
        Should time be transformed to numeric?

    time_to_numeric : bool, default = True
        Transforms input timestamps to float values.

    time_to_datetime : bool, default = False
        Transforms input timestamps to datatime objects. Setting `datetime_format` parameter is required.

    datetime_format : str
        The format of datetime object.

    Returns
    -------
    : List
        Parsed times.
    """

    parsed_times = times

    if isinstance(times, List):
        if time_to_numeric:
            parsed_times = [float(x) for x in times]

        if time_to_datetime:
            parsed_times = [datetime.strptime(x, datetime_format) for x in times]
    else:
        if time_to_numeric:
            parsed_times = float(times)

        if time_to_datetime:
            parsed_times = datetime.strptime(times, datetime_format)

    return parsed_times
