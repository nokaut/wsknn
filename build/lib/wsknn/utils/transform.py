import gzip
import json
import pickle


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
    : dict
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
    : dict
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
