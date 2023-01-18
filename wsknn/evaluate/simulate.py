from datetime import datetime
import tqdm


def simulate_input(number_of_sessions: int,
                   number_of_items: int,
                   min_session_length: int,
                   max_session_length: int,
                   time_start: datetime,
                   time_end: datetime,
                   output_sessions_jsonl_path: str,
                   output_items_jsonl_path: str,
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

    time_start : datetime
        Time when session could possibly start.

    time_end : datetime
        Time when session could possibly end.

    output_sessions_jsonl_path : str
        The path to the output sessions file.

    output_items_jsonl_path : str
        The path to the output items file.

    compress : bool, default=True
        Should jsonl be compressed to gzip?
    """


