from wsknn.model.wsknn import WSKNN


def fit(sessions: dict, items: dict) -> WSKNN:
    """
    The Function fits given session maps and item maps into the wsknn model.

    Parameters
    ----------
    sessions : dict
               sessions = {
                   session_id: (
                       [sequence_of_items],
                       [sequence_of_timestamps],
                       [sequence_of_event_type]
                   )
               }

    items : dict
            items = {
                item_id: (
                    [sequence_of_sessions],
                    [sequence_of_the_first_session_timestamps]
                )
            }

    Returns
    -------
    wsknn : WSKNN

    Examples
    --------
    >>> input_sessions = {
    ...                     'session_x': (
    ...                                      ['a', 'b', 'c'],
    ...                                      [10001, 10002, 10004],
    ...                                      ['view', 'click', 'click']
    ...                                  )
    ...                  }
    >>> input_items = {
    ...                   'a': (
    ...                            ['session_x'],
    ...                            [10001]
    ...                        ),
    ...                   'b': (
    ...                            ['session_x'],
    ...                            [10001]
    ...                        ),
    ...                   'c': (
    ...                            ['session_x'],
    ...                            [10001]
    ...                        ),
    ...               }
    >>> fitted_model = fit(input_sessions, input_items)
    """

    # Initilize VSKNN model
    wsknn = WSKNN()
    wsknn.fit(
        sessions,
        items
    )

    return wsknn
