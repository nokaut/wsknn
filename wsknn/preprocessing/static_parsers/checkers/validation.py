from typing import Dict


def check_event_keys_and_values(event: Dict,
                                session_id_key: str,
                                product_key: str,
                                action_key: str,
                                time_key: str):
    """
    Function checks if event has its all keys.

    Parameters
    ----------
    event : Dict

    session_id_key : str

    product_key : str

    action_key : str

    time_key : str

    Returns
    -------
    : Dict
        Empty dict if there is a missing key.
    """

    keys = {session_id_key, product_key, action_key, time_key}

    if not keys.issubset(set(event.keys())):
        return {}

    return event
