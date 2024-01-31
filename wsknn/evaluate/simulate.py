import gzip
import json

import numpy as np
import tqdm


def generate_input(number_of_sessions: int,
                   number_of_items: int,
                   min_session_length: int,
                   max_session_length: int):
    """
    Function generates session map and item map.

    Parameters
    ----------
    number_of_sessions : int
        The number of sessions.

    number_of_items : int
        The number of items.

    min_session_length : int
        The minimum session length.

    max_session_length : int
        The maximum session length.

    Returns
    -------
    : items_map, sessions_map
    """
    items_map = {}
    sessions_map = {}

    for session_idx in tqdm.tqdm(range(0, number_of_sessions)):
        session_length = np.random.randint(min_session_length,
                                           max_session_length + 1)
        items = np.random.randint(0,
                                  high=number_of_items,
                                  size=session_length)
        tstart = np.random.randint(1, 10)
        tend = tstart * max_session_length
        times = np.linspace(tstart, tend, session_length)

        sessions_map[session_idx] = [
            items.tolist(), times.tolist()
        ]

        for item in items:
            item = int(item)
            if item in items_map:
                items_map[item][0].append(session_idx)
                items_map[item][1].append(float(times[0]))
            else:
                items_map[item] = [
                    [session_idx], [float(times[0])]
                ]
    return items_map, sessions_map


def simulate_input(number_of_sessions: int,
                   number_of_items: int,
                   min_session_length: int,
                   max_session_length: int,
                   output_sessions_json_path: str = None,
                   output_items_json_path: str = None,
                   item_occurence_distribution="uniform",
                   compress=True):
    """
    Function generates fake sessions and items for testing purposes.

    Parameters
    ----------
    number_of_sessions : int
        The number of sessions.

    number_of_items : int
        The number of items.

    min_session_length : int
        The minimum session length.

    max_session_length : int
        The maximum session length.

    output_sessions_json_path : str
        The path to the output sessions file.

    output_items_json_path : str
        The path to the output items file.

    item_occurence_distribution : str
        How items are sampled, for now the only possible distribution is
        "uniform".

    compress : bool, default=True
        Should jsonl be compressed to gzip?
    """
    items_map, sessions_map = generate_input(
        number_of_sessions=number_of_sessions,
        number_of_items=number_of_items,
        min_session_length=min_session_length,
        max_session_length=max_session_length
    )

    if compress:
        output_sessions_json_path = output_sessions_json_path + '.gz'
        with gzip.open(
                output_sessions_json_path, 'wt', encoding='UTF-8'
        ) as zipfile:
            json.dump(sessions_map, zipfile)

        output_items_json_path = output_items_json_path + '.gz'
        with gzip.open(
                output_items_json_path, 'wt', encoding='UTF-8'
        ) as zipfile:
            json.dump(items_map, zipfile)

    del sessions_map
    del items_map

    return output_items_json_path, output_sessions_json_path
