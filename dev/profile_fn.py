import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple
from wsknn.fit_transform import fit
from wsknn.predict import predict
from wsknn.model.wsknn import WSKNN
from wsknn.utils.meta import parse_settings
from wsknn.utils.transform import load_pickled


DEFAULT_SETTINGS = os.path.join(os.getcwd(), '../settings.yml')
INPUT_DATA = '/Users/szymonsare/Documents/data/open/profile-wsknn/'
FILE_TYPE = 'pkl'
EXPERIMENT_OUTPUT = 'testtimes.csv'
EXPERIMENT_RANGE = 5


def get_files_paths(directory: str, file_type='') -> list:
    """Function prepares list of paths to the files within a given directory.

    Parameters
    ----------
    directory : str

    file_type : str, default=''
                If default empty string is passed then all files will be selected.

    Returns
    -------
    list
    """
    if len(file_type) > 0:
        files = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith(file_type)]
        return files
    else:
        return [os.path.join(directory, x) for x in os.listdir(directory)]


def get_item_sessions_pairs(files: list) -> list:
    """Function retrieves items, sessions and metadata part of a file.

    Parameters
    ----------
    files : list

    Returns
    -------
    list
        (item-sessions map file path, session-items map file path, file metadata)

    """

    metas = list()
    items = list()
    sessions = list()

    for ffile in files:
        fname = Path(ffile).stem
        # Get ID
        id_meta = fname[1+fname.find('_'):fname.rfind('.')]
        if id_meta in metas:
            pass
        else:
            metas.append(id_meta)
            for _f in files:
                items_prefix = 'items_' + id_meta
                sessions_prefix = 'sessions_' + id_meta
                if items_prefix in _f:
                    items.append(_f)
                elif sessions_prefix in _f:
                    sessions.append(_f)
            # Sanity check
            if len(metas) != len(items) != len(sessions):
                raise ValueError(f'Missing files, number of sessions: {len(sessions)}, '
                                 f'number of items: {len(items)}')

    output = list(zip(items, sessions, metas))
    return output


def _get_map(map_path: str) -> dict:
    """Function gets a dict with item-sessions or session-items map.

    Parameters
    ----------
    map_path : str

    Returns
    -------
    map : dict
    """
    loaded = load_pickled(map_path)
    _map = loaded['map']
    return _map


def _get_meta(fname: str) -> Tuple[int, int, int]:
    """Function gets metadata from a filename.

    Example: ge_s_20000_i_100_is_5_2022-03-03
             [ge=generated events]_[s=sessions]_[number of sessions]_[i=items]_[number of items]_[is=item space]_
             [number of closest items]_[experiment date YYYY-MM-DD]

    Parameters
    ----------
    fname : str

    Returns
    -------
    tuple
        (number of sessions, number of items, item space)
    """
    arguments = fname.split('_')
    sessions_n = int(arguments[2])
    items_n = int(arguments[4])
    space = int(arguments[6])
    return sessions_n, items_n, space


def build_session_items_list(directory: str, file_type='') -> Iterator[dict]:
    """Function prepares list of sessions and items for profiling.

    session name : sessions_ge_s_20000_i_100_is_5_2022-03-03
    item name    : items_ge_s_20000_i_100_is_5_2022-03-03

    schema       : [sessions or items]_[ge=generated events]_[s=sessions]_[number of sessions]
                   _[i=items]_[number of items]_[is=item space]_[number of closest items]_[experiment date YYYY-MM-DD]

    Parameters
    ----------
    directory : str
                Directory with sessions and items as a pkl files

    file_type : str, default=''
                If default empty string is passed then all files will be selected.

    Yields
    ------
    input_data : dict
                 Dict with session map, number of sessions, number of items and item space
                 {session_items_map: dict, item_sessions_map: dict, n_sessions: int, n_items: int, item_space: int}
    """

    # Prepare files
    all_files = [x for x in get_files_paths(directory, file_type)]
    session_items_pairs = get_item_sessions_pairs(all_files)

    # Parse into data
    for pair in session_items_pairs:

        items_map = _get_map(pair[0])
        sessions_map = _get_map(pair[1])
        n_sessions, n_items, i_space = _get_meta(pair[2])

        outputd = {
            'session_items_map': sessions_map,
            'item_sessions_map': items_map,
            'n_sessions': n_sessions,
            'n_items': n_items,
            'item_space': i_space
        }

        yield outputd


def test_model(model: WSKNN, settings: dict, number_of_repetitions=1000):
    """Function tests model locally by random sampling of sessions

    Parameters
    ----------
    model : VSKNN
            fitted model

    settings : dict
               parsed settings

    number_of_repetitions : int
                            How many times prediction must be performed.
    """
    kl = list(model.session_item_map.keys())

    for _ in range(0, number_of_repetitions):
        sample = random.randint(0, len(kl)-1)
        test_session = model.session_item_map[kl[sample]]
        test_session = {'a': test_session[0]}
        _ = predict(model, test_session)


def createcsv(filename: str, header: str):
    with open(filename, 'w') as csv:
        csv.write(header)
        csv.write('\n')


def store_data(csvfile: str, times: list, n_sessions: int, n_items: int, i_space: int):

    data_ = [n_sessions, n_items, i_space]
    data_.extend(times)
    data_ = [str(x) for x in data_]
    str_data = ','.join(data_)
    with open(csvfile, 'a') as csv:
        csv.write(str_data)
        csv.write('\n')


if __name__ == '__main__':
    _settings = parse_settings(DEFAULT_SETTINGS)
    maps = build_session_items_list(INPUT_DATA, FILE_TYPE)

    # Create output data file
    header_times = [f't{x}' for x in range(0, EXPERIMENT_RANGE)]
    _header = ['sessions', 'items', 'item_neighbors']
    _header.extend(header_times)
    str_header = ','.join(_header)
    createcsv(EXPERIMENT_OUTPUT, str_header)

    # Start experiment
    while True:
        try:
            a = next(maps)
            print('Next iteration starts')

            nsess = a['n_sessions']
            nitms = a['n_items']
            ispac = a['item_space']

            in_sess = a['session_items_map']
            in_itms = a['item_sessions_map']

            # Clock is ticking
            _times = []
            for _ in range(0, EXPERIMENT_RANGE):
                t0 = datetime.now()
                fitted_model = fit(in_sess, in_itms)
                test_model(fitted_model, _settings)
                tx = (datetime.now() - t0).seconds
                _times.append(tx)

            # Store info
            store_data(EXPERIMENT_OUTPUT, _times, nsess, nitms, ispac)

        except StopIteration:
            print('End...')
            break
