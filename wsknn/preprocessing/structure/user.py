import pickle
from typing import Dict

from wsknn.preprocessing.utils.transform import merge_dicts


class Users:
    """
    Class stores Users dict and its basic properties. The core object is a dictionary of unique users (keys) that are
    pointing to the specific sessions and their timestamps (lists).

    Parameters
    ----------
    event_session_key
        The name of a session key.

    event_user_key
        The name of a user key.

    event_time_key
        The name of a timestamp key.

    Attributes
    ----------
    user_sessions_map : Dict
        ``{user_id: {sessions: List, timestamps: List}}``

    event_session_key
        The name of a session key.

    event_user_key
        The name of a user key.

    event_time_key
        The name of a timestamp key.

    metadata : str
        The general description of the ``Users`` object.

    Methods
    -------
    append(event)
        Appends given event to the user-sessions map.

    export(filename)
        Method exports created mapping to pickled dictionary.

    load(filename)
        Loads pickled ``Users`` object into a new instance of a class.

    save_object(filename)
        The ``Users`` object is stored as a Python pickle binary file.

    __add__(Users)
        Intersection with other ``Users`` object. It is a set operation. Therefore, sessions that are assigned to
        the same user within ``Users(1)`` and ``Users(2)`` won't be duplicated.

    __str__()
        The basic info about the class.
    """

    def __init__(self,
                 event_session_key,
                 event_user_key,
                 event_time_key):

        self.user_sessions_map = dict()

        self.event_session_key = event_session_key
        self.event_user_key = event_user_key
        self.event_time_key = event_time_key

        self.metadata = self._get_metadata()

    @staticmethod
    def _get_metadata():
        meta = """
            The users object is a dictionary of unique items (keys) that are pointing to the specific sessions and their 
            timestamps.

            Key map:

            ```
            user_sessions_map = {
                user_id: (
                    [sequence_of_sessions],
                    [sequence_of_the_first_session_timestamps]
                )
            }
            ```

            Other keys:
            - time_start: the first date in a dataset.
            - time_end: the last date in a dataset.
            - longest_sessions_vector_size: size of the longest sequence sessions,
            - number_of_sessions: the length of the ``user_sessions_map``.
            """
        return meta

    def _append_user_session_and_timestamp(self, user, session, timestamp):
        """
        Function appends user - session and user - timestamp to existing list of sessions and timestamps.

        Parameters
        ----------
        user
            User index.

        session
            Session index.

        timestamp
            Event time.
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

    def append(self, event: Dict):
        """
        Method appends given event into user-sessions and user-timestamps map.

        Parameters
        ----------
        event : Dict
            Python dictionary with values of: event time, event session index, event product (optional).
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

    def export(self, filename: str):
        """
        Method saves object's attributes in a dictionary, within a pickled object.

        Parameters
        ----------
        filename : str
            Path to the pickled object. Method appends suffix .pkl if it is not given.
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

    def save_object(self, filename: str):
        """
        Method saves the whole object in a pickled file.

        Parameters
        ----------
        filename : str
            Path to the pickled object. Method appends suffix .pkl if it is not given.

        """

        pkl = '.pkl'
        if not filename.endswith(pkl):
            filename = filename + pkl

        with open(filename, 'wb') as storage:
            pickle.dump(self, storage)

    def load(self, filename: str):
        """
        Method loads a pickled ``Users`` object and assigns its properties and data into the class instance.

        Parameters
        ----------
        filename : str
            The path to the pickled ``Users`` object.

        Raises
        ------
        IOError
            Method can't load pickled object into processed mapping.
        """
        # Check if current object has any items to avoid overwriting
        if self.user_sessions_map:
            raise IOError('Cannot load data into object with users! '
                          'Created empty object to load data from pickled file.')

        # Load
        with open(filename, 'rb') as stored_data:
            self.__dict__.update(pickle.load(stored_data).__dict__)

    def __add__(self, other):
        """
        Intersection of two user-sessions mappings. Map records are not duplicated.

        Parameters
        ----------
        other : Users
            Other ``Users`` object.

        Returns
        -------
        merged : Users
            Merged objects.
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
