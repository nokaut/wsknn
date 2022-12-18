def mrr_func(recommendations, relevant_items):
    """
    Function calculates the mean reciprocal rank of a top k recommendations.

    Parameters
    ----------
    recommendations
        The list of recommended products, sorted from the most relevant items into the least.

    relevant_items
        The set of relevant items.

    Returns
    -------
    rank : float
           ``MRR - ratio: 1 / idx_rel``, where: ``idx_rel`` is an index of the first occurrence of ANY relevant
           product in recommended items list.
    """
    for idx, item in enumerate(recommendations):
        ii = idx + 1
        if item in relevant_items:
            rank = 1 / ii
            return rank
    return 0


def precision_func(recommendations, relevant_items):
    """
    Function calculates the precision of a top k recommendations.

    Parameters
    ----------
    recommendations
        The set recommended products, sorted from the most relevant items into the least.

    relevant_items
        The set of relevant items.

    Returns
    -------
    precision : float
                ``(Number of relevant items in recommendations) / (Number of recommendations)``
    """
    rank = 0

    for item in recommendations:
        if item in relevant_items:
            rank = rank + 1

    precision = rank / len(recommendations)
    return precision


def recall_func(recommendations, relevant_items):
    """
    Function calculates the recall of a top k recommendations.

    Parameters
    ----------
    recommendations
        The set recommended products, sorted from the most relevant items into the least.

    relevant_items
        The set of relevant items.

    Returns
    -------
    recall : float
             ``(Number of relevant items in recommendations) / (Number of relevant items for the user)``
    """

    rank = 0

    for item in recommendations:
        if item in relevant_items:
            rank = rank + 1

    recall = rank / len(relevant_items)
    return recall
