import pickle

from preprocessing.utils.transform import merge_dicts


class Users:
    """
    Class stores Users dict and its basic properties. The core object is a dictionary of unique users (keys) that are
        pointing to the specific sessions and their timestamps (lists).

    Parameters:
        :param event_session_key: (str)
        :param event_user_key: (str)
        :param event_time_key: (str)

    user_sessions_map = {
        user_id: {
            sessions: [sequence_of_sessions],
            timestamps: [sequence_of_the_first_session_timestamps]
        }
    }

    Available methods:

    - append(event): appends given dict with event to the existing dictionary,
    - save(): Items object is stored as a Python pickle binary file,
    - __add__(Items): adds other Users object. It is a set operation. Therefore, sessions that is assigned to the same
        item within Users(1) and Users(2) is not duplicated.
    - __str__(): basic info about the class.

    """

    def __init__(self,
                 event_session_key: str,
                 event_user_key: str,
                 event_time_key: str):

        self.user_sessions_map = dict()

        self.event_session_key = event_session_key
        self.event_user_key = event_user_key
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

    def _append_user_session_and_timestamp(self, user: str, session: str, timestamp: int):
        """
        Function appends item session and timestamp to existing list of sessions and timestamps.

        :param user: (str),
        :param session: (str),
        :param timestamp: (int)
        """
        if session in self.user_sessions_map[user][0]:

            # Session exists, upload timestamp if needed
            idx = self.user_sessions_map[user][0].index(session)

            past_ts = self.user_sessions_map[user][1][idx]
            if timestamp < past_ts:
                self.user_sessions_map[user][1][idx] = timestamp
        else:
            # Append session to sessions
            self.user_sessions_map[user][0].append(session)
            # Append timestamp to timestamps
            self.user_sessions_map[user][1].append(timestamp)

    def append(self, event: dict):
        """
                Method appends given event into internal structure of the class.
                :param event: (dict) keys: event_type, event_time, event_session, event_product
                """

        uid = event[self.event_user_key]
        session = event[self.event_session_key]
        dt = event[self.event_time_key]

        if uid in self.user_sessions_map:
            self._append_user_session_and_timestamp(uid, session, dt)
        else:
            # Create Item-sessions map
            self.user_sessions_map[uid] = (
                [session],
                [dt]
            )

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
            'map': self.user_sessions_map,
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
        if self.user_sessions_map:
            raise IOError('Cannot load data into object with users! '
                          'Create empty object to load data from pickled file.')

        # Load
        with open(filename, 'rb') as stored_data:
            self.__dict__.update(pickle.load(stored_data).__dict__)

    def __add__(self, other):
        """
        Adds users and mapped sessions from one Users object into the other.
        :param other: (Users)
        :return: (Users)
        """
        merged = Users(other.event_session_key, other.event_user_key, other.event_time_key)
        merged.item_sessions_map = merge_dicts(self.user_sessions_map, other.user_sessions_map)
        return merged

    def __str__(self):
        if self.user_sessions_map:
            msg = f'Items object statistics:\n' \
                  f'*Number of unique users: {len(list(self.user_sessions_map.keys()))}'
            return msg
        else:
            msg = 'Empty object. Append events or load pickled item-sessions map!'
            return msg
