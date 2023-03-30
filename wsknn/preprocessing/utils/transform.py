import datetime
from typing import Union, List, Dict
from more_itertools import sort_together


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Function merges two dicts.

    Parameters
    ----------
    dict1 : Dict

    dict2 : Dict

    Returns
    -------
    : Dict
    """
    merged = dict1.copy()

    for key_d2 in dict2.keys():
        d2vals = dict2[key_d2]
        if key_d2 in dict1:
            d1vals = dict1[key_d2]
            vals = merge_vals(d1vals, d2vals)
            merged[key_d2] = vals
        else:
            merged[key_d2] = d2vals

    return merged


def merge_vals(vals1: List, vals2: List) -> List:
    """
    Function merges lists of values and leave only unique elements.

    Parameters
    ----------
    vals1 : List
        The list with values to merge.

    vals2 : List
        The list with values to merge.

    Returns
    -------
    : List
        Merged and lists.
    """

    vals1[0].extend(vals2[0])
    vals1[1].extend(vals2[1])

    if len(vals2) >= 3:
        vals1[2].extend(vals2[2])

    if len(vals2) == 4:
        vals1[3].extend(vals2[3])

    # Sort by time
    vals1 = sort_together(vals1, key_list=(1,))
    vals_output = []
    for val_row in vals1:
        if isinstance(val_row, tuple):
            vals_output.append(list(val_row))
        else:
            vals_output.append(val_row)

    return vals_output


def parse_dt_to_seconds(event_time: Union[int, str]) -> int:
    """
    Function converts given date to seconds from 1970-01-01. Function rounds returned value to a full second.

    Parameters
    ----------
    event_time : Union[int, str]
        Even time as timestamp or string. Datetime in formats '%Y-%m-%dT%H:%M:%S.%fZ' or '%Y-%m-%dT%H:%M:%S'

    Returns
    -------
    timestamp_ticks : int
        How many seconds to ``event_time`` from 1970-01-01, UTC tz.
    """
    try:
        int(event_time)
        return event_time
    except ValueError:
        if isinstance(event_time, str):
            if len(event_time) >= 24:
                str_scheme = '%Y-%m-%dT%H:%M:%S.%fZ'
            elif len(event_time) == 19:
                str_scheme = '%Y-%m-%dT%H:%M:%S'
            else:
                raise TypeError('Given Datetime Format not allowed! Function uses "%Y-%m-%dT%H:%M:%S.%fZ" and '
                                '"%Y-%m-%dT%H:%M:%S" formats')
            base_time = datetime.datetime.strptime(event_time, str_scheme)
            # Transform to utc

            timestamp_ticks = int(base_time.timestamp())
            return timestamp_ticks
        else:
            raise TypeError(f'Unrecognized datetime format: {type(event_time)}')


def parse_seconds_to_dt(timestamp2transform: int) -> str:
    """
    Function converts given timestamp in seconds to string datetime in format '%Y-%m-%dT%H:%M:%S.%fZ'.

    Parameters
    ----------
    timestamp2transform : int
        Timestamp in seconds to transform.

    Returns
    -------
    s_datetime : str
        Parsed datetime in format '%Y-%m-%dT%H:%M:%S.%fZ'
    """
    s_datetime = datetime.datetime.fromtimestamp(
        timestamp2transform
    ).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    return s_datetime


def parse_event_actions(action: str, all_actions: List) -> Dict:
    """
    Function maps event actions to a vector of actions.

    :param action: (str) name of the user action
    :param all_actions: (list) available actions
    :return: (Dict) {action: 1, other_action: 0} where action is equal to given action parameter and other_action
             is other action type from the all_actions_list.
    """

    actions_dict = {}

    for act in all_actions:
        if act == action:
            actions_dict[act] = 1
        else:
            actions_dict[act] = 0

    return actions_dict
