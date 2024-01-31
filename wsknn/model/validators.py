import random


def validate_mapping_dtypes(sessions, items):
    """
    Function checks if data types between session-items map and item-sessions
    map are the same.

    Parameters
    ----------
    sessions : Dict
        The map of items that occur in the session and their
        timestamps, and (optional) their types and their weights.

        >>> sessions = {
        ...     session_id: (
        ...     [sequence_of_items],
        ...     [sequence_of_timestamps],
        ...     [(optional) event names (types)],
        ...     [(optional) weights]
        ...     )
        ... }

    items : Dict
        The map of items and the sessions where those items are present,
        and the first timestamp of those sessions. If not provided then
        the item-sessions map is created from the ``sessions`` parameter.

        >>> items = {
        ...     item_id: (
        ...     [sequence_of_sessions],
        ...     [sequence_of_the_first_session_timestamps]
        ...     )
        ... }

    Raises
    ------
    TypeError : data types are not the same in both mappings.

    """

    sessions_key = random.choice(list(sessions.keys()))
    items_key = random.choice(list(items.keys()))

    s_type_session_map = type(sessions_key)
    i_type_session_map = type(sessions[sessions_key][0][0])
    ts_type_session_map = type(sessions[sessions_key][1][0])

    i_type_item_map = type(items_key)
    s_type_item_map = type(items[items_key][0][0])
    ts_type_item_map = type(items[items_key][1][0])

    if s_type_item_map != s_type_session_map:
        raise TypeError(f'Sessions in session-items map and item-sessions'
                        f' map have different data types. Session-items '
                        f'map dtype: {s_type_session_map}, and item-sessions '
                        f'map dtype: {s_type_item_map}')

    if i_type_item_map != i_type_session_map:
        raise TypeError(f'Items in session-items map and item-sessions'
                        f' map have different data types. Session-items '
                        f'map dtype: {i_type_session_map}, and item-sessions '
                        f'map dtype: {i_type_item_map}')

    if ts_type_item_map != ts_type_session_map:
        raise TypeError(f'Timestamps in session-items map and item-sessions'
                        f' map have different data types. Session-items '
                        f'map dtype: {ts_type_session_map}, and item-sessions '
                        f'map dtype: {ts_type_item_map}')
