from typing import Dict

from wsknn.preprocessing.static_parsers.parse import parse_stream


def _get_header(header, sep, session_index: int, product_index: int, time_index: int, action_index: int = None):
    """
    Function gets header names.

    Parameters
    ----------
    header : str
        Line with a header.

    sep : str
        Separator.

    session_index : int

    product_index : int

    time_index : int

    action_index : int, optional

    Returns
    -------
    header_map : Dict
        Dictionary with pairs {index: name}
    """
    try:
        header_names = header.split(sep)
    except Exception as _:
        raise AttributeError('Cannot parse header from file, wrong separator!')

    # Create dictionary

    if action_index is not None:
        header_map = {
            session_index: header_names[session_index],
            action_index: header_names[action_index],
            product_index: header_names[product_index],
            time_index: header_names[time_index]
        }
    else:
        header_map = {
            session_index: header_names[session_index],
            product_index: header_names[product_index],
            time_index: header_names[time_index]
        }

    return header_map


def parse_flat_file_fn(dataset: str,
                       sep: str,
                       session_index: int,
                       product_index: int,
                       time_index: int,
                       action_index: int = None,
                       use_header_row: bool = False,
                       time_to_numeric=False,
                       time_to_datetime=False,
                       datetime_format='',
                       allowed_actions: Dict = None,
                       purchase_action_name=None,
                       ignore_errors: bool = True,
                       progress_bar: bool = False):
    """
    Function parses data from flat file into item-sessions and session-items maps.

    Parameters
    ----------
    dataset : str
        Input file.

    sep : str
        Separator used to separate values.

    session_index : int
        The index of the session.

    product_index : int
        The index of the product.

    time_index : int
        The index of the event timestamp.

    action_index : int, optional
        The index of the event action.z

    use_header_row : bool, default = False
        Use first row values as a header.

    time_to_numeric : bool, default = False
        Transforms input timestamps to float values.

    time_to_datetime : bool, default = False
        Transforms input timestamps to datatime objects. Setting ``datetime_format`` parameter is required.

    datetime_format : str
        The format of datetime object.

    allowed_actions : Dict, optional
        Allowed actions and their weights.

    purchase_action_name: Any, optional
        The name of the final action (it is required to apply weight into the session vector).

    ignore_errors : bool, default=True
        Ignore rows that raise exceptions.

    progress_bar : bool
        Show parsing progress.

    Returns
    -------
    items, sessions : Items, Sessions
        The mappings of item-session and session-items.
    """

    with open(dataset, 'r') as fin:
        if use_header_row:
            header = _get_header(next(fin),
                                 sep=sep,
                                 session_index=session_index,
                                 product_index=product_index,
                                 time_index=time_index,
                                 action_index=action_index)
        else:
            header = None

        items, sessions = parse_stream(
            events=fin,
            sep=sep,
            allowed_actions=allowed_actions,
            purchase_action_name=purchase_action_name,
            session_index=session_index,
            product_index=product_index,
            time_index=time_index,
            action_index=action_index,
            time_to_numeric=time_to_numeric,
            time_to_datetime=time_to_datetime,
            datetime_format=datetime_format,
            ignore_errors=ignore_errors,
            header_names=header,
            progress_bar=progress_bar
        )

    return items, sessions
