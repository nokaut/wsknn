import pickle
from preprocessing.utils.calc import get_larger_value, get_smaller_value
from preprocessing.utils.transform import merge_dicts, parse_seconds_to_dt


class Items:
    """
    Class stores Items dict and its basic properties. The core object is a dictionary of unique items (keys) that are
        pointing to the specific sessions and their timestamps (lists).

    Parameters:
        :param event_session_key: (str)
        :param event_product_key: (str)
        :param event_time_key: (str)

    item_sessions_map = {
        item_id: {
            sessions: [sequence_of_sessions],
            timestamps: [sequence_of_the_first_session_timestamps]
        }
    }

    Other information stored by this class are:
    - time_start: the first date in a dataset.
    - time_end: the last date in a dataset.
    - longest_sessions_vector_size: size of the longest sequence sessions,
    - number_of_items: the length of item_sessions_map.

    Available methods:

    - append(event): appends given dict with event to the existing dictionary,
    - save(): Items object is stored as a Python pickle binary file,
    - __add__(Items): adds other Items object. It is a set operation. Therefore, session that is assigned to the same
        item within Items(1) and Items(2) is not duplicated.
    - __str__(): basic info about the class.

    """

    def __init__(self,
                 event_session_key: str,
                 event_product_key: str,
                 event_time_key: str):

        self.item_sessions_map = dict()
        self.time_start = 1_000_000_000_000_000
        self.time_end = 0
        self.longest_sessions_vector_size = 0
        self.number_of_items = 0

        self.event_session_key = event_session_key
        self.event_product_key = event_product_key
        self.event_time_key = event_time_key

        self.metadata = self._get_metadata()

    @staticmethod
    def _get_metadata():
        meta = """
        The items object is a dictionary of unique items (keys) that are pointing to the specific sessions and their 
        timestamps (lists).

        Key map:

        item_sessions_map = {
            item_id: (
                [sequence_of_sessions],
                [sequence_of_the_first_session_timestamps]
            )
        }

        Other keys:
        - time_start: the first date in a dataset.
        - time_end: the last date in a dataset.
        - longest_sessions_vector_size: size of the longest sequence sessions,
        - number_of_items: the length of item_sessions_map.
        """
        return meta

    def _update_number_of_items(self):
        self.number_of_items = self.number_of_items + 1

    def _update_longest_session_size(self, session_length: int):
        if session_length > self.longest_sessions_vector_size:
            self.longest_sessions_vector_size = session_length

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

    def _append_item_session_and_timestamp(self, item: str, session: str, timestamp: int):
        """
        Function appends item session and timestamp to existing list of sessions and timestamps.

        :param item: (str),
        :param session: (str),
        :param timestamp: (int)
        """
        if session in self.item_sessions_map[item][0]:

            # Session exists, upload timestamp if needed
            idx = self.item_sessions_map[item][0].index(session)

            past_ts = self.item_sessions_map[item][1][idx]
            if timestamp < past_ts:
                self.item_sessions_map[item][1][idx] = timestamp

            # Update the first and the last time reading if needed
            self._update_timestamps(timestamp)
        else:
            # Append session to sessions
            self.item_sessions_map[item][0].append(session)
            # Append timestamp to timestamps
            self.item_sessions_map[item][1].append(timestamp)
            # Update the first and the last time reading if needed
            self._update_timestamps(timestamp)
            # Update longest session info
            self._update_longest_session_size(len(self.item_sessions_map[item][0]))

    def append(self, event: dict):
        """
        Method appends given event into internal structure of the class.
        :param event: (dict) keys: event_type, event_time, event_session, event_product
        """

        item = event[self.event_product_key]
        session = event[self.event_session_key]
        dt = event[self.event_time_key]

        if item in self.item_sessions_map:
            self._append_item_session_and_timestamp(item, session, dt)
        else:
            # Create Item-sessions map
            self.item_sessions_map[item] = (
                [session],
                [dt]
            )
            # Update number of items in the dictionary
            self._update_number_of_items()
            # Update the first and the last time reading if needed
            self._update_timestamps(dt)
            # Update longest session info
            self._update_longest_session_size(len(self.item_sessions_map[item][0]))

    def clean_map(self, no_of_sessions: int):
        """
        Method cleans session-items map and leaves only sessions longer or equal to session_length.

        :param no_of_sessions: (int)
        """

        items = list(self.item_sessions_map.keys())
        items_to_pop = list()

        for itm in items:
            item = self.item_sessions_map[itm]
            if len(item[0]) < no_of_sessions:
                # remove record from a dict
                items_to_pop.append(itm)
            else:
                # TODO: update class parameters
                pass

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
            'map': self.item_sessions_map,
            'time_start': self.time_start,
            'time_end': self.time_end,
            'longest_sessions_vector_size': self.longest_sessions_vector_size,
            'number_of_items': self.number_of_items,
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
        if self.item_sessions_map:
            raise IOError('Cannot load data into a populated object! '
                          'Create empty object to load data from pickled file.')

        # Load
        with open(filename, 'rb') as stored_data:
            self.__dict__.update(pickle.load(stored_data).__dict__)

    def __add__(self, other):
        """
        Adds items and mapped sessions from one Items object into the other.
        :param other: (Items)
        :return: (Items)
        """
        merged = Items(other.event_session_key, other.event_product_key, other.event_time_key)
        merged.item_sessions_map = merge_dicts(self.item_sessions_map, other.item_sessions_map)
        merged.time_start = get_smaller_value(self.time_start, other.time_start)
        merged.time_end = get_larger_value(self.time_end, other.time_end)
        merged.longest_sessions_vector_size = get_larger_value(self.longest_sessions_vector_size,
                                                               other.longest_sessions_vector_size)
        merged.number_of_items = len(merged.item_sessions_map)
        return merged

    def __str__(self):
        if self.item_sessions_map:
            try:
                ts = parse_seconds_to_dt(self.time_start)
                te = parse_seconds_to_dt(self.time_end)
            except ValueError:
                # TODO
                ts = self.time_start
                te = self.time_end
            msg = f'Items object statistics:\n' \
                  f'*Number of unique items: {self.number_of_items}\n' \
                  f'*The longest sessions vector size: {self.longest_sessions_vector_size}\n' \
                  f'*Period start: {ts}\n' \
                  f'*Period end: {te}'
            return msg
        else:
            msg = 'Empty object. Append events or load pickled item-sessions map!'
            return msg
