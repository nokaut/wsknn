from numpy import log10


def linear_item_score(i):
    """Function weights events based on their position in a sequence. Output is normalized to the range [0:1]. For the
       long sequences of events (index 10+) it returns zero.

    Parameters
    ----------
    i : int
        Item position.

    Returns
    -------
    result : float
             Linear rank.
    """
    result = 1 - (0.1 * i) if i < 10 else 0
    result = result / 0.9
    return result


def inv_pos_item_score(i):
    """Function returns the final rank as a simple inverse of the item position.

    Parameters
    ----------
    i : int
        Item position.

    Returns
    -------
    result : float
             Inverted position rank.
    """
    result = 1 / i
    return result


def log_item_score(i):
    """Function returns rank as a transformed inverted log10 of the sum (position + 1.7).
       Result is normalized to the range (0:1].

    Parameters
    ----------
    i : int
        Item position.

    Returns
    -------
    result : float
             Logarithmic rank.
    """
    norm_term = 1 / (log10(2.7))  # The largest possible value
    result = 1 / log10(i + 1.7)
    result = result / norm_term
    return result


def quadratic_item_score(i):
    """Function is similar to inverted and linear functions but weights are decreasing at non-linear rate
       and accelerate with the item position.

    Parameters
    ----------
    i : int
        Item position.

    Returns
    -------
    result : float
             Inverted square position rank.
    """
    result = 1 / (i * i)
    return result
