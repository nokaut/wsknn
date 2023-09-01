import pickle
from typing import Union, Dict, Iterable

from wsknn.preprocessing.utils.calc import get_larger_value, get_smaller_value
from wsknn.preprocessing.utils.transform import merge_dicts, parse_seconds_to_dt


class Sessions:
    """
    Class stores session-items mapping and its basic properties. The core object is a dictionary of unique
    sessions (keys) that are pointing to the session-items and their timestamps (lists).

    Parameters
    ----------
    event_session_key
        The name of a session key.

    event_product_key
        The name of an item key.

    event_time_key
        The name of a timestamp key.

    event_action_key : str, optional
        The name of a key with actions that may be used for neighbors filtering.

    event_action_weights : Dict, optional
        Dictionary with weights that should be assigned to event actions.

    Attributes
    ----------
    session_items_actions_map : Dict
        The session-items mapper ``{session_id: [[items], [timestamps], [optional - actions], [optional - weights]]}``.

    time_start : int, default = 1_000_000_000_000_000
        The initial timestamp, first event in whole dataset.

    time_end : int, default = 0
        The timestamp of the last event in dataset.

    longest_items_vector_size: int, default = 0
        The longest sequence of products.

    number_of_sessions : int, default = 0
        The number of sessions within a ``session_items_actions_map`` object.

    event_session_key
        See the ``event_session_key`` parameter.

    event_product_key
        See the ``event_product_key`` parameter.

    event_time_key
        See the ``event_time_key`` parameter.

    event_action_key
        See the ``event_action_key`` parameter.

    action_weights
        See the ``event_action_weights`` parameter.

    metadata : str
        A description of the ``Sessions`` class.

    Methods
    -------
    append(event)
        Appends a single event to the session-items map.

    export(filename)
        Method exports created mapping to pickled dictionary.

    load(filename)
        Loads pickled ``Sessions`` object into a new instance of a class.

    save_object(filename)
        Users object is stored as a Python pickle binary file.

    update_weights_of_purchase_session()
        Adds constant value to all weights if session ended up with purchase.

    __add__(Users)
        Adds other Sessions object.

    __str__()
        The basic info about the class.

    """

    def __init__(self,
                 event_session_key: str,
                 event_product_key: str,
                 event_time_key: str,
                 event_action_key: str = None,
                 event_action_weights: Dict = None):

        self.session_items_actions_map = dict()
        self.time_start = 1_000_000_000_000_000
        self.time_end = 0
        self.longest_items_vector_size = 0
        self.number_of_sessions = 0

        self.event_session_key = event_session_key
        self.event_product_key = event_product_key
        self.event_time_key = event_time_key
        self.event_action_key = event_action_key
        self.action_weights = event_action_weights

        self.metadata = self._get_metadata()

    @staticmethod
    def _get_metadata():
        meta = """
        The sessions object is a dictionary of unique sessions (keys) that are pointing to the specific items and their 
        timestamps and, optionally, weighting factors.

        Key map:

        map = {
            session_id: [
                [items],
                [timestamps],
                [optional - actions],
                [optional - action weights]
            ]
        }

        Other keys:
        - time_start: the first date in a dataset.
        - time_end: the last date in a dataset.
        - longest_sessions_vector_size: size of the longest sequence sessions,
        - number_of_items: the length of item_sessions_map.
        """
        return meta

    def _update_number_of_sessions(self):
        self.number_of_sessions = self.number_of_sessions + 1

    def _update_longest_event_sequence_size(self, seq_length: int):
        if seq_length > self.longest_items_vector_size:
            self.longest_items_vector_size = seq_length

    def _update_first_timestamp(self, ts):
        self.time_start = ts

    def _update_last_timestamp(self, ts):
        self.time_end = ts

    def _update_timestamps(self, ts: int):

        # First
        if ts < self.time_start:
            self._update_first_timestamp(ts)

        # Last
        if ts > self.time_end:
            self._update_last_timestamp(ts)

    def _append_session_item_and_timestamp(self,
                                           session: str,
                                           item: str,
                                           timestamp: int,
                                           action: Union[str, None] = None):
        """
        Function appends item session and timestamp to existing list of sessions and timestamps.
        """
        # Append item
        self.session_items_actions_map[session][0].append(item)
        # Append timestamp
        self.session_items_actions_map[session][1].append(timestamp)

        if action is not None:
            # Append action type
            self.session_items_actions_map[session][2].append(action)

        if self.action_weights is not None:
            self.session_items_actions_map[session][3].append(self.action_weights.get(action))

        # Update the first and the last time reading if needed
        self._update_timestamps(timestamp)
        # Update longest session info
        self._update_longest_event_sequence_size(len(self.session_items_actions_map[session][0]))

    def append(self, event: dict):
        """
        Method appends the event into session-items map.

        Parameters
        ----------
        event : Dict
            Single event that consists: session id, timestamp, product, and optionally action type.
        """

        item = event[self.event_product_key]
        session = event[self.event_session_key]
        dt = event[self.event_time_key]

        if self.event_action_key is not None:
            action = event[self.event_action_key]
        else:
            action = None

        if session in self.session_items_actions_map:
            self._append_session_item_and_timestamp(session, item, dt, action)
        else:
            # Create Item-sessions map

            if action is None:
                self.session_items_actions_map[session] = (
                    [item],
                    [dt]
                )
            else:
                if self.action_weights is None:
                    self.session_items_actions_map[session] = (
                        [item],
                        [dt],
                        [action]
                    )
                else:
                    self.session_items_actions_map[session] = (
                        [item],
                        [dt],
                        [action],
                        [self.action_weights.get(action, 0.001)]
                    )
            # Update number of sessions in the dictionary
            self._update_number_of_sessions()
            # Update the first and the last time reading if needed
            self._update_timestamps(dt)
            # Update longest session info
            self._update_longest_event_sequence_size(len(self.session_items_actions_map[session][0]))

    def update_weights_of_purchase_session(self, session_id,
                                           additive_factor=0.0,
                                           bought_products: Iterable = None):
        """
        Method updates item weights accordingly to the given additive factor. Running this method is required when
        we generate optional weights list.

        Parameters
        ----------
        session_id : Any
            The index of a session.

        additive_factor : float
            Weight added to the item.

        bought_products : List, optional
            If provided, then only those items weights are updated.
        """
        for idx, item in enumerate(self.session_items_actions_map[session_id][0]):
            old_value = self.session_items_actions_map[session_id][3][idx]
            new_value = old_value + additive_factor
            if bought_products is not None:
                if item in bought_products:
                    self.session_items_actions_map[session_id][3][idx] = new_value
            else:
                self.session_items_actions_map[session_id][3][idx] = new_value

    def export_to_dict(self, filename: str):
        """
        Method exports object's attributes as a dictionary in a pickled object.

        Parameters
        ----------
        filename : str
            The path to the pickled object. If suffix .pkl is not given then methods appends it into the file.

        Notes
        -----
        Dictionary keys are:
            * ``map``: session-items map,
            * ``time_start``: the earliest event time,
            * ``time_end``: the latest event time,
            * ``longest_items_vector_size``: the number of items in the longest session-sequence,
            * ``number_of_sessions``: total number of parsed sessions,
            * ``metadata``: additional class info.

        """
        pkl = '.pkl'
        if not filename.endswith(pkl):
            filename = filename + pkl

        objects_att_dict = {
            'map': self.session_items_actions_map,
            'time_start': self.time_start,
            'time_end': self.time_end,
            'longest_items_vector_size': self.longest_items_vector_size,
            'number_of_sessions': self.number_of_sessions,
            'metadata': self.metadata
        }

        with open(filename, 'wb') as storage:
            pickle.dump(objects_att_dict, storage)

    def save(self, filename: str):
        """
        Method saves ``Sessions`` class object into a pickled file.

        Parameters
        ----------
        filename : str
            The path to the pickled object. If suffix .pkl is not given then methods appends it into the file.
        """

        pkl = '.pkl'
        if not filename.endswith(pkl):
            filename = filename + pkl

        with open(filename, 'wb') as storage:
            pickle.dump(self, storage)

    def load(self, filename: str):
        """
        Method loads pickled ``Sessions`` class object and assigns its properties and data into the class instance.

        Parameters
        ----------
        filename : str
            The path to the file.

        Raises
        ------
        IOError
            Cannot load data into ``Sessions`` object with parsed events.
        """
        # Check if current object has any items to avoid overwriting
        if self.number_of_sessions > 0:
            raise IOError('Cannot load data into object with sessions! '
                          'Create empty object to load data from pickled file.')

        # Load
        with open(filename, 'rb') as stored_data:
            self.__dict__.update(pickle.load(stored_data).__dict__)

    def __add__(self, other):
        """
        Adds sessions and mapped items from one ``Sessions`` object into the other.

        Parameters
        ----------
        other : Sessions

        Returns
        -------
        merged : Sessions
            The merged and updated ``Sessions`` object.
        """
        merged = Sessions(other.event_session_key,
                          other.event_product_key,
                          other.event_time_key,
                          other.event_action_key)

        merged.session_items_actions_map = merge_dicts(self.session_items_actions_map,
                                                       other.session_items_actions_map)

        merged.time_start = get_smaller_value(self.time_start, other.time_start)
        merged.time_end = get_larger_value(self.time_end, other.time_end)
        merged.longest_items_vector_size = get_larger_value(self.longest_items_vector_size,
                                                            other.longest_items_vector_size)
        merged.number_of_sessions = len(merged.session_items_actions_map)
        return merged

    def __str__(self):
        if self.session_items_actions_map:
            try:
                ts = parse_seconds_to_dt(self.time_start)
                te = parse_seconds_to_dt(self.time_end)
            except ValueError:
                ts = self.time_start
                te = self.time_end
            msg = f'Sessions object statistics:\n' \
                  f'*Number of unique sessions: {self.number_of_sessions}\n' \
                  f'*The longest event stream size per session: {self.longest_items_vector_size}\n' \
                  f'*Period start: {ts}\n' \
                  f'*Period end: {te}'
            return msg
        else:
            msg = 'Empty object. Append events or load pickled session-items map!'
            return msg
