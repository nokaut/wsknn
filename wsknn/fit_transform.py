from typing import Dict

from wsknn.model.wsknn import WSKNN


def fit(sessions: Dict,
        items: Dict,
        number_of_recommendations=None,
        number_of_closest_neighbors=None,
        session_sampling_strategy=None,
        possible_neighbors_sample_size=None,
        weighting_strategy=None,
        rank_strategy=None,
        required_sampling_event=None):
    """

    Sets input session-items and item-sessions maps.

    Parameters
    ----------
    sessions : Dict
               sessions = {
                   session_id: (
                       [ sequence_of_items ],
                       [ sequence_of_timestamps ],
                       [ [OPTIONAL] sequence_of_event_types ]
                   )
               }

    items : Dict
            items = {
                item_id: (
                    [ sequence_of_sessions ],
                    [ sequence_of_the_first_session_timestamps ]
                )
            }

    number_of_recommendations : int or None, default=None
                                Resets the number of recommendations.

    number_of_closest_neighbors : int or None, default=None
                                  Resets the number of closest neighbors.

    session_sampling_strategy : str or None, default=None
                                How to filter the initial sample of sessions. Available strategies are:
                                    - 'common_items': sample sessions with the same items as the input session,
                                    - 'recent': sample the most actual sessions,
                                    - 'random': get a random sample of sessions.

    possible_neighbors_sample_size : int or None, default=None
                                     How many sessions from the model are sampled to make a recommendation. If not
                                     set then it is the number given during the class initilization.

    weighting_strategy : str or None, default=None
                         The similarity measurement between sessions. Available options: 'linear', 'log'
                         and 'quadratic'. If not set then weighting strategy is taken from the parameter set
                         during the class initialization.

    rank_strategy : str or None, default=None
                    How we calculate an item rank (based on its position in a session sequence). Available options
                    are: 'inv', 'linear', 'log', 'quadratic'. If not set then model uses rank strategy given
                    during the initilization.

    required_sampling_event : int or str, default = None
                              Set this paramater to the event name if sessions with it must be included in
                              the neighbors selection. For example, this event may be a "purchase".

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

    wsknn = WSKNN(number_of_recommendations,
                  number_of_closest_neighbors,
                  session_sampling_strategy,
                  possible_neighbors_sample_size,
                  weighting_strategy,
                  rank_strategy,
                  required_sampling_event)

    wsknn.fit(sessions, items)

    return wsknn
