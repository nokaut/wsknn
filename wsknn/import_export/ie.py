import joblib

from wsknn import WSKNN


def save(model: WSKNN, save_path: str):
    """
    Function saves model as a joblib object.

    Parameters
    ----------
    model : WSKNN
        Trained model.

    save_path : str

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    Raises
    ------
    ValueError
        Model is not trained.
    """

    # Check model
    if model.session_item_map is None or model.item_session_map is None:
        raise ValueError('Model is not trained yet!')

    filenames = joblib.dump(
        model,
        save_path,
        compress=5
    )
    return filenames


def load(load_path: str):
    """
    Function loads wsknn model.

    Parameters
    ----------
    load_path : str

    Returns
    -------
    model : WSKNN
        Trained model.
    """

    model = joblib.load(load_path)
    return model
