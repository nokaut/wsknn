from typing import Dict
from more_itertools import sort_together


def map_sessions_to_items(sessions_map: Dict, sort_items_map: bool = True) -> Dict:
    """
    Function transforms sessions map into items map.

    Parameters
    ----------
    sessions_map : Dict
        >>> sessions = {
        ...    session_id: (
        ...        [ sequence_of_items ],
        ...        [ sequence_of_timestamps ],
        ...        [ [OPTIONAL] sequence_of_event_types ],
        ...        [ [OPTIONAL] sequence_of_event_weights]
        ...    )
        ...}

    sort_items_map : bool, default = True
        Sorts item-sessions map by the first timestamp of a session (then sessions are sorted in ascending order, from
        the oldest to the newest).

    Returns
    -------
    items_map : Dict
        Mapped item-sessions dictionary.
        >>> items = {
        ...    item_id: (
        ...        [ sequence_of_sessions ],
        ...        [ sequence_of_the_first_session_timestamps ]
        ...    )
        ...}
    """

    items_map = {}

    for session_k, values in sessions_map.items():

        items = values[0]
        timestamps = values[1]

        for idx, item in enumerate(items):
            ts = timestamps[idx]
            if item not in items_map:
                items_map[item] = [
                    [session_k],
                    [ts]
                ]
            else:
                if session_k not in items_map[item][0]:
                    items_map[item][0].append(session_k)
                    items_map[item][1].append(ts)
                else:
                    session_index = items_map[item][0].index(session_k)
                    the_last_timestamp = items_map[item][1][session_index]
                    if ts < the_last_timestamp:
                        items_map[item][1][session_index] = ts

    if sort_items_map:
        for ikey, ivalues in items_map.items():
            # sort values
            new_ivalues = list(sort_together(ivalues, key_list=(1,)))
            # update mapping
            items_map[ikey] = new_ivalues

    return items_map
