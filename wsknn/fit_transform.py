from typing import Dict, Union

from wsknn.preprocessing.structure.item import Items
from wsknn.preprocessing.structure.session import Sessions
from wsknn.model.wsknn import WSKNN


def fit(sessions: Union[Dict, Sessions],
        items: Union[Dict, Items] = None,
        number_of_recommendations: int = 5,
        number_of_neighbors: int = 10,
        sampling_strategy: str = 'common_items',
        sample_size: int = 1000,
        weighting_func: str = 'linear',
        ranking_strategy: str = 'linear',
        return_events_from_session: bool = True,
        required_sampling_event: Union[int, str] = None,
        required_sampling_event_index: int = None,
        sampling_str_event_weights_index: int = None,
        recommend_any: bool = False):
    """
    Sets input session-items and item-sessions maps.

    Parameters
    ----------
    sessions : Dict or Sessions
        >>> sessions = {
        ...    session_id: (
        ...        [ sequence_of_items ],
        ...        [ sequence_of_timestamps ],
        ...        [ [OPTIONAL] sequence_of_event_types ],
        ...        [ [OPTIONAL] sequence_of_event_weights]
        ...    )
        ...}

    items : Dict or Items, optional
        If not provided then item-sessions map is created from the `sessions` parameter.
        >>> items = {
        ...    item_id: (
        ...        [ sequence_of_sessions ],
        ...        [ sequence_of_the_first_session_timestamps ]
        ...    )
        ...}

    number_of_recommendations : int, default=5
        The number of recommended items.

    number_of_neighbors : int, default=10
        The number of the closest sessions to choose the items from.

    sampling_strategy : str, default='common_items'
        How to filter the initial sample of sessions. Available strategies are:
            - 'common_items': sample sessions with the same items as the input session,
            - 'recent': sample the most actual sessions,
            - 'random': get a random sample of sessions,
            - 'weighted_events': select sessions based on the specific weights assigned to events.

    sample_size : int, default=1000
        How many sessions from the model are sampled to make a recommendation.

    weighting_func : str, default='linear'
        The similarity measurement between sessions. Available options: 'linear', 'log' and 'quadratic'.

    ranking_strategy : str, default='linear'
        How we calculate an item rank (based on its position in a session sequence). Available options
        are: 'inv', 'linear', 'log', 'quadratic'.

    return_events_from_session : bool, default = True
        Should algorithm return the same events as in session if this is only neighbor?

    required_sampling_event : int or str, default = None
        Set this paramater to the event name if sessions with it must be included in the neighbors selection.
        For example, this event may be a "purchase".

    required_sampling_event_index : int, default = None
        If the `required_sampling_event` parameter is filled then you must pass an index of a row with event names.

    sampling_str_event_weights_index : int, default = None
        If `sampling_strategy` is set to `weighted_events` then you must pass an index of a row with event weights.

    recommend_any : bool, default = False
        If recommender returns less than number of recommendations items then return random items.

    Returns
    -------
    wsknn : WSKNN
        The trained Weighted session-based K-nn model.

    Examples
    --------
    >>> input_sessions = {
    ...     'session_x': (
    ...         ['a', 'b', 'c'],
    ...         [10001, 10002, 10004],
    ...         ['view', 'click', 'click']
    ...     )
    ... }
    >>> input_items = {
    ...     'a': (
    ...         ['session_x'],
    ...         [10001]
    ...     ),
    ...     'b': (
    ...         ['session_x'],
    ...         [10001]
    ...     ),
    ...     'c': (
    ...         ['session_x'],
    ...         [10001]
    ...     ),
    ... }
    >>> fitted_model = fit(input_sessions, input_items)
    """

    if isinstance(sessions, Sessions):
        sessions = sessions.session_items_actions_map

    if items is not None:
        if isinstance(items, Items):
            items = items.item_sessions_map

    # Set model parameters
    wsknn = WSKNN(number_of_recommendations=number_of_recommendations,
                  number_of_neighbors=number_of_neighbors,
                  sampling_strategy=sampling_strategy,
                  sample_size=sample_size,
                  weighting_func=weighting_func,
                  ranking_strategy=ranking_strategy,
                  return_events_from_session=return_events_from_session,
                  required_sampling_event=required_sampling_event,
                  required_sampling_event_index=required_sampling_event_index,
                  sampling_event_weights_index=sampling_str_event_weights_index,
                  recommend_any=recommend_any)

    # Fit sessions and items
    wsknn.fit(sessions, items)

    return wsknn
