from typing import Dict


def sessions_map_to_items_map(sessions_map: Dict) -> Dict:
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
        first_timestamp = values[1][0]

        for item in items:
            if item not in items_map:
                items_map[item] = [
                    [session_k],
                    [first_timestamp]
                ]
            else:
                items_map[item][0].append(session_k)
                items_map[item][1].append(first_timestamp)

    return items_map
