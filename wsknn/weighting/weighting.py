from wsknn.weighting.item_weighting import inv_pos_item_score, linear_item_score, log_item_score, quadratic_item_score
from wsknn.weighting.session_weighting import linear_session_score, log_session_score, quadratic_session_score


# Items

def weight_item_score(fn_name: str, element_pos: int) -> float:
    """Function calculates item weight based on its position in a session.

    Parameters
    ----------
    fn_name : str
              Function used for item weighting. Available options: 'linear', 'div', 'log', 'quadratic'.

    element_pos : int
                  Position of an element in a sequence.

    Returns
    -------
    float
        Item weight.
    """

    weighting = {
        'linear': linear_item_score,
        'inv': inv_pos_item_score,
        'log': log_item_score,
        'quadratic': quadratic_item_score
    }

    if fn_name in weighting:
        return weighting[fn_name](element_pos)
    else:
        return 1


# Sessions

def weight_session_items(fn_name: str, element_pos: int, length: int) -> float:
    """Function weights session items by specific fn.

    Parameters
    ----------
    fn_name : str
              Function used for session weighting. Available options: 'linear', 'log', 'quadratic'.

    element_pos : int
                  Position of an element in a sequence.

    length : int
             Overall session length.

    Returns
    -------
    float
        Session's item weight.
    """

    weighting = {
        'linear': linear_session_score,
        'log': log_session_score,
        'quadratic': quadratic_session_score
    }

    return weighting[fn_name](element_pos, length)
