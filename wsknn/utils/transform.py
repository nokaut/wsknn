import pickle


def load_pickled(filename: str) -> dict:
    """Method loads pickled items / sessions object.

    Parameters
    ----------
    filename : str

    Returns
    -------
    pickled_object : dict
    """
    with open(filename, 'rb') as stored_data:
        pickled_object = pickle.load(stored_data)
    return pickled_object

