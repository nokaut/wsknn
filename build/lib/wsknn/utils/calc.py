def weight_set_pair(first: set, second: set, mapped_items_weights: dict) -> float:
    """
    The function calculates weighted average of the common items from two sessions based on the dict with items and
        their weights.

    Parameters
    ----------
    first : set
            unique items from the set A

    second : set
             unique items from the set B

    mappped_items_weights : dict
                            {item_id : weight}

    Returns
    -------
    float
        Estimated weight between two item sets. Always 0 or positive.
    """
    if len(mapped_items_weights) >= 1:
        common_items = first & second
        weights_sum = 0
        for i in common_items:
            weights_sum += mapped_items_weights[i]
        result = weights_sum / len(mapped_items_weights)
        return result
    return 0


def is_there_any_common_element(first: set, second: set) -> int:
    """
    The function checks if there are common elements in two sets and if there are any then it returns 1, else 0.

    Parameters
    ----------
    first : set
            unique items from the set A

    second : set
             unique items from the set B

    Returns
    -------
    bool
        True if there is any common element between sets A and B; 0 otherwise
    """

    return int(bool(first & second))
