from typing import Dict, List
from wsknn.model.wsknn import WSKNN


def predict(model: WSKNN,
            sessions: List,
            settings: Dict = None) -> List:
    """
    The function is an alias for the .predict() method of the WSKNN model.

    Parameters
    ----------
    model : WSKNN
        Fitted WSKNN model.

    sessions : List
        Sequence of items for recommendation. It must be a nested List of lists:
        >>> [
        ...     [items],
        ...     [timestamps],
        ...     [(optional) event names],
        ...     [(optional) weights]
        ... ]

    settings : Dict
        Settings of the model. It is worth noticing, that using this parameter allows to test different setups.
        Possible parameters are grouped in the `settings.yml` file under the `model` key.

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

    Examples
    --------
    >>> sessions = [
            [['item a', 'item b'], []],
            [['item x', 'item a', 'item n'], []]
    ... ]
    >>> recommendations = predict(fitted_model, )
    """
    if model.item_session_map is None or model.session_item_map is None:
        raise ValueError('Given model does not have an item map and a session map. Fit those before prediction')

    recommendations = model.recommend(sessions, settings)
    return recommendations
