import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path


PREPROCESSING_FILENAME = 'preprocessing_settings.yml'
IO_KEY = 'io'
PROPERTIES_KEY = 'properties'
DATE_FORMAT = '%Y-%m-%d'


def get_files_list(directory: str, ftype: str) -> list:
    """
    Function returns list of files with a given type.

    Parameters
    ----------
    directory : str
        The path to directory with files.

    ftype : str
        File type.

    Returns
    -------
    files : List
        The list with files of specified type.
    """
    files = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith(ftype)]
    return files


def get_files_list_from_daterange(directory: str, ftype: str, date_start=None, date_end=None) -> list:
    """
    Function returns list of selected file types from a given directory within a specified dates. Filename MUST contain
    date in the format YYYY-MM-DD.

    Parameters
    ----------
    directory : str
        The path to directory with files.

    ftype : str
        File type.

    date_start : str, optional
        Date in ISO format YYYY-MM-DD. If you provide ``start_date`` then files in the past from this date won't
        be included.

    date_end : str, optional
        Date in ISO format YYYY-MM-DD. If you provide ``end_date`` then files in the future from this date won't
        be included.

    Returns
    -------
    files : List
        The list with files of specified type within date ranges.
    """
    files = get_files_list(directory, ftype)
    if date_start is not None or date_end is not None:
        files = _filter_files_by_date(files, date_start, date_end)

    # Sort files
    files = sorted(files)
    return files


def _filter_files_by_date(files, date_start, date_end):
    if date_start is None:
        possible_dates = _get_possible_dates_upper_bound_only(date_end, files)
    elif date_end is None:
        possible_dates = _get_possible_dates_lower_bound_only(date_start, files)
    else:
        possible_dates = _get_possible_dates(date_start, date_end)

    files_new = []
    for pdate in possible_dates:
        for fname in files:
            if pdate in fname:
                files_new.append(fname)

    return files_new


def _get_possible_dates_upper_bound_only(upper_date, fileslist):
    sorted_files = sorted(fileslist)
    flist = []
    check = 0
    for fname in sorted_files:
        if check == 0:
            flist.append(fname)
            if upper_date in fname:
                check = 1
        else:
            if upper_date in fname:
                flist.append(fname)
            else:
                break

    return flist


def _get_possible_dates_lower_bound_only(lower_date, fileslist):
    sorted_files = sorted(fileslist)
    flist = []
    check = 0
    for fname in sorted_files:
        if check == 0:
            if lower_date in fname:
                flist.append(fname)
                check = 1
        else:
            flist.append(fname)

    return flist


def _get_possible_dates(lower_date, upper_date):
    date_low = datetime.strptime(lower_date, DATE_FORMAT)

    dates = [lower_date]
    d0 = date_low
    while True:
        d0 = d0 + timedelta(days=1)
        dstr = d0.strftime(DATE_FORMAT)
        dates.append(dstr)
        if dstr == upper_date:
            break

    return dates


def search_upwards_for_file(filename: str):
    """
    Search in the current directory and all directories above it
    for a file of a particular name.

    Parameters
    ----------
    filename : str
        The filename to look for.

    Returns
    -------
    : pathlib.Path
        The location of the first file found or ``None``, if none was found
    """
    d = Path.cwd()
    root = Path(d.root)

    while d != root:
        attempt = d / filename
        if attempt.exists():
            return attempt
        d = d.parent

    return None


def parse_settings(fpath=None):
    """
    Function parses ``preprocessing_settings`` file.

    Parameters
    ----------
    fpath : str, optional
        If given settings are retrieved from file, otherwise function tries to find ``preprocessing_settings`` in a
        working directory.

    Returns
    -------
    : Dict
        Parsed settings.

    Raises
    ------
    RunetimeError
        File not found or it has missing keys.
    """

    if fpath is None:
        settings_file = PREPROCESSING_FILENAME
        path_to_file = search_upwards_for_file(settings_file)
    else:
        path_to_file = fpath

    with open(path_to_file, 'r') as settings_stream:
        parsed_settings = yaml.safe_load(settings_stream)

    settings_keys = list(parsed_settings.keys())
    if IO_KEY in settings_keys:
        if PROPERTIES_KEY in settings_keys:
            return parsed_settings

    msg = f'Does the {PREPROCESSING_FILENAME} file exist in a top level of the module? ' \
          f'Does it have "{IO_KEY}" and "{PROPERTIES_KEY}" keys?'
    raise RuntimeError(msg)
