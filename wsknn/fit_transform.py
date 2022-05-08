from typing import Dict, Union

from wsknn.model.wsknn import WSKNN


def fit(sessions: Dict,
        items: Dict,
        number_of_recommendations: int = 5,
        number_of_neighbors: int = 10,
        sampling_strategy: str = 'common_items',
        sample_size: int = 1000,
        weighting_func: str = 'linear',
        ranking_strategy: str = 'linear',
        required_sampling_event: Union[int, str] = None):
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

    number_of_recommendations : int, default=5
                                The number of recommended items.

    number_of_neighbors : int, default=10
                          The number of closest sessions to choose the items from.

    sampling_strategy : str, default='common_items'
                        How to filter the initial sample of sessions. Available strategies are:
                        - 'common_items': sample sessions with the same items as the input session,
                        - 'recent': sample the most actual sessions,
                        - 'random': get a random sample of sessions.

    sample_size : int, default=1000
                  How many sessions from the model are sampled to make a recommendation.

    weighting_func : str, default='linear'
                     The similarity measurement between sessions. Available options: 'linear', 'log' and 'quadratic'.

    ranking_strategy : str, default='linear'
                       How we calculate an item rank (based on its position in a session sequence). Available options
                       are: 'inv', 'linear', 'log', 'quadratic'.

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
                  number_of_neighbors,
                  sampling_strategy,
                  sample_size,
                  weighting_func,
                  ranking_strategy,
                  required_sampling_event)

    wsknn.fit(sessions, items)

    return wsknn
