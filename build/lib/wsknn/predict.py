from typing import Dict, List
from wsknn.model.wsknn import WSKNN


def predict(model: WSKNN,
            sessions: List,
            settings: Dict = None) -> List:
    """
    The function is an alias to the .predict() method of the WSKNN model.

    Parameters
    ----------
    model : WSKNN
            Fitted VSKNN model.

    sessions : List
               Sequence of items for recommendation. It must be a nested List of lists:
                [
                    [items],
                    [timestamps],
                    [properties]
                ]

    settings : Dict

    Returns
    -------
    recommendations : List
        [
            [item a, rank a], [item b, rank b]
        ]

    Raises
    ------
    ValueError
        Model not fitted.
    """
    if model.item_session_map is None or model.session_item_map is None:
        raise ValueError('Given model does not have an item map and a session map. Fit those before prediction')

    recommendations = model.recommend(sessions, settings)
    return recommendations
