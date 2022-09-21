from numpy import log10


def linear_session_score(i, length):
    """Newest elements in sequence => largest weights.

    Parameters
    ----------
    i : int
        Element position, i+1 must be less than length.

    length : int
             Length of a sequence.

    Results
    -------
    result : float
             Session rank between 0 and 1.
    """

    result = (i + 1) / length
    return result


def log_session_score(i, length):
    """Newest element in sequence => largest weight.

    Summary
    -------
    Function calculates weights based on the sequence length and a position of element in a sequence.
        Longer sequences create more steps between limits of 0 and 2.31.
        For example:

        a) element position: 1, sequence length: 10   -> weight = 0.97
        b) element position: 1, sequence length: 1000 -> weight = 0.33

    In practice it means that the past elements of the longest sequences have lower weights than the past elements
        of a short sequence. It mimics our short-term memory and fn gives the biggest weights to the newest events and
        past and middle-sequence events are treated as a not relevant, approx.: to 90% of the sequence.

    Parameters
    ----------
    i : int
        Element position, i+1 must be less than length.

    length : int
             Length of a sequence.

    Results
    -------
    result : float
             Session rank between 0 and 1. Normalized from 0:2.31.
    """

    norm_term = 1 / (log10(2.7))  # The largest possible value
    result = 1 / (log10((length - i) + 1.7))

    return result / norm_term


def quadratic_session_score(i, length):
    """Function penalizes old elements in sequence more than the new events. The past elements have much smaller
       weights than the new ones. It follows a quadratic function.

    Parameters
    ----------
    i : int
        Element position, i+1 must be less than length.

    length : int
             Length of a sequence.

    Results
    -------
    result : float
             Session rank between 0 and 1.
    """

    c = i / length
    result = c*c
    return result