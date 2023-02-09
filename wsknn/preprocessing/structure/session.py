import pickle
from typing import Union, Dict, List

# TODO: from preprocessing.core.structure.models import SessionItemsMapModel -> def of data types
from preprocessing.utils.calc import get_larger_value, get_smaller_value
from preprocessing.utils.transform import merge_dicts, parse_seconds_to_dt


class Sessions:

    def __init__(self,
                 event_session_key: str,
                 event_product_key: str,
                 event_time_key: str,
                 event_action_key: Union[str, None] = None,
                 event_action_weights: Dict = None):
        """
        Class stores Sessions dict and its basic properties. The core object is a dictionary of unique sessions (keys)
            that are pointing to the specific items and their timestamps and event types (lists).

        Parameters:
        :param event_session_key: (str)
        :param event_product_key: (str)
        :param event_time_key: (str)
        :param event_action_key: (str or None)
        :param event_action_weights: (Dict)

        sessions_dict = {
            session_id: [
                [items],
                [timestamps],
                [actions],
                [action weights]
            ]
        }

        Other information stored by this class are:
            - time_start: the first date in a dataset.
            - time_end: the last date in a dataset.
            - longest_items_vector_size: size of the longest sequence items,
            - number_of_sessions: the length of session-items-map.

        Available methods:

            - append(event): appends given dict with event to the existing dictionary,
            - save(): Sessions object is stored as a Python pickle binary file,
            - __add__(Items): adds other Sessions object. It is a set operation. Therefore, item that is assigned to
                the same session within Sessions(1) and Sessions(2) is not duplicated.
            - __str__(): basic info about the class.

            """
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
        timestamps and, optionally, weighting factor.

        Key map:

        map = {
            session_id: [
                [items],
                [timestamps],
                [actions]
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

        :param session: (str),
        :param item: (str),
        :param action: (str),
        :param timestamp: (int).
        """
        # Append item
        self.session_items_actions_map[session][0].append(item)
        # Append timestamp
        self.session_items_actions_map[session][1].append(timestamp)

        if action is not None:
            # Append action type
            self.session_items_actions_map[session][2].append(action)

        # Update the first and the last time reading if needed
        self._update_timestamps(timestamp)
        # Update longest session info
        self._update_longest_event_sequence_size(len(self.session_items_actions_map[session][0]))

    def append(self, event: dict):
        """
        Method appends given event into internal structure of the class.
        :param event: (dict) keys: event_type, event_time, event_session, event_product
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
                self.session_items_actions_map[session] = (
                    [item],
                    [dt],
                    [action],
                    [self.action_weights[action]]
                )
            # Update number of sessions in the dictionary
            self._update_number_of_sessions()
            # Update the first and the last time reading if needed
            self._update_timestamps(dt)
            # Update longest session info
            self._update_longest_event_sequence_size(len(self.session_items_actions_map[session][0]))

    def clean_map(self, session_length: int):
        """
        Method cleans session-items map and leaves only sessions longer or equal to session_length.

        :param session_length: (int)
        """
        #
        # self.time_start = 1_000_000_000_000_000
        # self.time_end = 0
        # self.longest_items_vector_size = 0
        # self.number_of_sessions = 0

        sessions = list(self.session_items_actions_map.keys())
        sessions_to_pop = list()

        for session in sessions:
            item = self.session_items_actions_map[session]
            if len(item[0]) < session_length:
                # remove record from a dict
                sessions_to_pop.append(session)
            else:
                # TODO: update class parameters
                pass

        # Pop sessions
        for spop in sessions_to_pop:
            self.session_items_actions_map.pop(spop)

    def update_weights(self, session_id, products: List, additive_factor: float):
        """
        Method updates all weights accordingly to the given additive factor and a list of products with weights to
            be updated.

        :param session_id: id of a session.
        :param products: (List) ids of items (usually those items that have been purchased).
        :param additive_factor: (float)
        """
        for idx, item in enumerate(self.session_items_actions_map[session_id][0]):
            if item in products:
                old_value = self.session_items_actions_map[session_id][3][idx]
                new_value = old_value + additive_factor
                self.session_items_actions_map[session_id][3][idx] = new_value

    def export_to_dict(self, filename: str):
        """
        Method saves object's attributes dict in a pickled object.
        :param filename: (str) path to the pickled object. If suffix .pkl is not given then methods appends it into the
            file.
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
        Method saves object in a pickled object.
        :param filename: (str) path to the pickled object. If suffix .pkl is not given then methods appends it into the
            file.
        """

        pkl = '.pkl'
        if not filename.endswith(pkl):
            filename = filename + pkl

        with open(filename, 'wb') as storage:
            pickle.dump(self, storage)

    def load(self, filename: str):
        """
        Method loads pickled object and assigns its properties and data into the class instance.

        :param filename:
        :return:
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
        Adds sessions and mapped items from one Sessions object into the other.
        :param other: (Sessions)
        :return: (Sessions)
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
