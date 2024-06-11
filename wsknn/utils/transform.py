import gzip
import json
import pickle


def _df_to_mapping(df, main_col, time_col, index_col=None, action_col=None, weights_col=None):
    """
    Transforms dataframe to the dictionary mapping.

    Parameters
    ----------
    df : DataFrame
        Session-items or item-sessions map as pandas DataFrame object.

    main_col :
        Column with mapped items or sessions.

    time_col :
        Column with timestamps.

    index_col : optional
        Column with key values, if not provided then ``DataFrame.index`` is used.

    action_col : optional
        Column with event actions.

    weights_col : optional
        Column with event weights.

    Returns
    -------
    : Dict
    """
    sessions_map = {}

    if index_col is not None:
        indexes = df[index_col].values
    else:
        indexes = df.index

    # Prepare values
    cols = [main_col, time_col]
    if action_col:
        cols.append(action_col)
    if weights_col:
        cols.append(weights_col)

    vals = df[cols].values

    # Parse
    for _no, _id in enumerate(indexes):
        row = vals[_no].tolist()
        sessions_map[_id] = row

    return sessions_map


def dataframe_to_item_sessions_map(df, main_col, time_col, index_col=None):
    """
    Function transforms given item sessions dataframe to dictionary used by the WSKNN model.

    Parameters
    ----------
    df : DataFrame
        Session-items or item-sessions map as pandas DataFrame object.

    main_col :
        Column with mapped items or sessions.

    time_col :
        Column with timestamps.

    index_col : optional
        Column with key values, if not provided then ``DataFrame.index`` is used.

    Returns
    -------
    : Dict
    """

    item_sessions_map = _df_to_mapping(
        df=df,
        main_col=main_col,
        time_col=time_col,
        index_col=index_col
    )
    return item_sessions_map


def dataframe_to_session_items_map(df, main_col, time_col, index_col=None, action_col=None, weights_col=None):
    """
    Function transforms given session items dataframe to dictionary used by the WSKNN model.

    Parameters
    ----------
    df : DataFrame
        Session-items or item-sessions map as pandas DataFrame object.

    main_col :
        Column with mapped items or sessions.

    time_col :
        Column with timestamps.

    index_col : optional
        Column with key values, if not provided then ``DataFrame.index`` is used.

    action_col : optional
        Column with event actions.

    weights_col : optional
        Column with event weights.

    Returns
    -------
    : Dict
    """
    session_items_map = _df_to_mapping(
        df=df,
        main_col=main_col,
        time_col=time_col,
        index_col=index_col,
        action_col=action_col,
        weights_col=weights_col
    )
    return session_items_map


def load_pickled(filename: str) -> dict:
    """
    The function loads pickled items / sessions object.

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


def load_jsonl(filename: str) -> dict:
    """
    Function loads data stored in JSON Lines.

    Parameters
    ----------
    filename : str
               Path to the file.

    Returns
    -------
    datadict : dict
        Python dictionary with unique records.
    """
    datadict = {}
    with open(filename, 'r') as fstream:
        for fline in fstream:
            pdict = json.loads(fline)
            datadict.update(pdict)
    return datadict


def load_gzipped_jsonl(filename: str, encoding: str = 'UTF-8') -> dict:
    """
    Function loads data stored in gzipped JSON Lines.

    Parameters
    ----------
    filename : str
               Path to the file.

    encoding : str, default = 'utf-8'

    Returns
    -------
    datadict : dict
        Python dictionary with unique records.
    """

    datadict = {}

    with gzip.open(filename, 'rt', encoding=encoding) as fstream:
        for fline in fstream:
            datadict.update(json.loads(fline))

    return datadict


def load_gzipped_pickle(filename: str) -> dict:
    """
    The function loads gzipped and pickled items / sessions object.

    Parameters
    ----------
    filename : str

    Returns
    -------
    pickled_object : dict
    """
    with gzip.open(filename, 'rb') as fstream:
        pickled_object = pickle.load(fstream)
    return pickled_object
