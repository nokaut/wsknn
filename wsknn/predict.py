from wsknn.model.wsknn import WSKNN


def predict(model: WSKNN,
            sessions: dict,
            settings_dict=None) -> dict:
    """
    The function is an alias to the .predict() method of the WSKNN model.

    Parameters
    ----------
    model : WSKNN
            Fitted VSKNN model.

    sessions : dict
               Sequence of items for recommendation. As a dict, it is a user id (key) and products and their timestamps
               (values).

    settings_dict : None or dict
                    If given them it overwrites initial settings of a model. Available fields are:
                    - number_of_recommendations : int
                    - number_of_closest_neighbors : int
                    - session_sampling_strategy: str, available options: 'random', 'recent', 'common_items'
                    - possible_neighbors_sample_size : int
                    - weighting_strategy: str, available options: 'linear', 'log', 'quadratic'
                    - rank_strategy: str, available options: 'linear', 'inv', 'log', 'quadratic'

    Returns
    -------
    predicted : dict or list
                List of tuples (Item, Weight) of the length given by the number_of_recommendations parameter in
                a descending order; or the same

    Raises
    ------
    ValueError
        Model not fitted.
    """
    if model.item_session_map is None or model.session_item_map is None:
        raise ValueError('Given model does not have an item map and a session map. Fit those before prediction')

    if settings_dict is None:
        predicted = model.predict(sessions)
    else:
        predicted = model.predict(sessions, **settings_dict)
    return predicted
