from typing import Dict, List
from wsknn.model.wsknn import WSKNN


def batch_predict(model: WSKNN,
                  sessions: List[Dict],
                  settings: Dict = None) -> List[Dict]:
    """
    The function predicts multiple sessions at once.

    Parameters
    ----------
    model : WSKNN
        Fitted WSKNN model.

    sessions : List[Dict]
        User-sessions for recommendations:

        >>> [
            ...     {"user A": [*]},
            ...     {"user B": [**]}
            ... ]

        where (*) might be:

        >>> [
            ...     [sequence_of_items],
            ...     [sequence_of_timestamps],
            ...     [(optional) event names (types)],
            ...     [(optional) weights]
            ... ]

    settings : Dict
        Settings of the model. It is worth noticing, that using this parameter
        allows to test different setups. Possible parameters are grouped in
        the ``settings.yml`` file under the ``model`` key.

    Returns
    -------
    inference : List[Dict]
        Recommendations and their weights for each user.

        >>> [
        ...   {"user A": [
        ...     [item a, rank a],
        ...     [item b, rank b]
        ...   ]},
        ...   {"user B": [...]},
        ... ]

    """
    inference = [
        __multi_predict(
            model, settings, s
        ) for s in sessions
    ]

    return inference


def predict(model: WSKNN,
            sessions: List,
            settings: Dict = None) -> List:
    """
    The function is an alias for the `WSKNN.recommend()` method.

    Parameters
    ----------
    model : WSKNN
        Fitted WSKNN model.

    sessions : List
        Sequence of items for recommendation. It must be a nested List of
        lists:
        >>> [
        ...     [items],
        ...     [timestamps],
        ...     [(optional) event names],
        ...     [(optional) weights]
        ... ]

    settings : Dict
        Settings of the model. It is worth noticing, that using this parameter
        allows to test different setups. Possible parameters are grouped in
        the ``settings.yml`` file under the ``model`` key.

    Returns
    -------
    recommendations : List
        Item recommendations and their ranks.

        >>> [
        ...     [item a, rank a], [item b, rank b]
        ... ]

    Raises
    ------
    ValueError
        Model wasn't fitted.
    """
    if model.item_session_map is None or model.session_item_map is None:
        raise ValueError('Given model does not have an item map and a session '
                         'map. Fit those before prediction')

    recommendations = model.recommend(sessions, settings)
    return recommendations


def __multi_predict(_m, _settings, _s):
    _k = list(_s.keys())[0]
    _v = _s[_k]
    return {_k: predict(_m, _v, _settings)}
