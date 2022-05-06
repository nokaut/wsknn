import random
from typing import Iterable, Union, List, Set, Dict

from wsknn.weighting import weight_session_items, weight_item_score
from wsknn.utils.calc import weight_set_pair
from wsknn.utils.errors import check_data_dimension, check_numeric_type_instance,\
    InvalidDimensionsError, InvalidTimestampError


class WSKNN:
    """
    The class represents the Weighted Session-Based k-nn model.

    Parameters
    ----------
    number_of_recommendations : int, default=5
                                The number of recommended items.

    number_of_neighbors : int, default=10
                          The number of closest sessions to choose the items from.

    sampling_strategy : str, default='common_items'
                        How to filter the initial sample of sessions. Available strategies are:
                        - 'common_items': sample sessions with the same items as the input session,
                        - 'recent': sample the most actual sessions,
                        - 'random': get a random sample of sessions.

    sample_size : int, default=1000
                  How many sessions from the model are sampled to make a recommendation.

    weighting_func : str, default='linear'
                     The similarity measurement between sessions. Available options: 'linear', 'log' and 'quadratic'.

    ranking_strategy : str, default='linear'
                       How we calculate an item rank (based on its position in a session sequence). Available options
                       are: 'inv', 'linear', 'log', 'quadratic'.

    required_sampling_event : int or str, default = None
                              Set this paramater to the event name if sessions with it must be included in
                              the neighbors selection. For example, this event may be a "purchase".

    Attributes
    ----------
    weighting_functions: List
                         List with weighting functions: ['linear', 'log', 'quadratic'].

    ranking_strategies : List
                         List with ranking strategies: ['linear', 'log', 'quadratic', 'inv'].

    session_item_map : Dict
                       sessions = {
                           session_id: (
                               [sequence_of_items],
                               [sequence_of_timestamps],
                               [sequence_of_event_type]
                           )
                       }
                       Dict populated with fit() method.

    item_session_map : Dict
                       items = {
                           item_id: (
                               [sequence_of_sessions],
                               [sequence_of_the_first_session_timestamps]
                           )
                       }
                       Dict populated with fit() method.

    n_of_recommendations : int
                           Number of items recommended.

    number_of_closest_neighbors : int

    sampling_strategy : str
                        See sampling_strategy parameter.

    possible_neighbors_sample_size : int
                                     See sample_size parameter.

    weighting_function : str
                         See weighting_func parameter.

    ranking_strategy : str
                       See ranking_strategy parameter.

    required_sampling_event : Union[int, str], default = None
                              See required_sampling_event parameter.

    Methods
    -------
    fit()

    predict()

    Raises
    ------
    InvalidDimensionsError : wrong number of nested sequences within session-items or item-sessions maps.

    InvalidTimestampError : wrong type of timestamp - int is required.

    TypeError : wrong type of nested structures within session-items or item-sessions maps.

    """

    def __init__(self,
                 number_of_recommendations: int = 5,
                 number_of_neighbors: int = 10,
                 sampling_strategy: str = 'common_items',
                 sample_size: int = 1000,
                 weighting_func: str = 'linear',
                 ranking_strategy: str = 'linear',
                 required_sampling_event: Union[int, str] = None):

        self.sampling_strategies = ['common_items', 'recent', 'random']
        self.weighting_functions = ['linear', 'log', 'quadratic']
        self.ranking_strategies = ['linear', 'log', 'quadratic', 'inv']

        self.session_item_map = None
        self.item_session_map = None

        self.n_of_recommendations = number_of_recommendations
        self.number_of_closest_neighbors = number_of_neighbors
        self.possible_neighbors_sample_size = sample_size

        self.required_sampling_event = required_sampling_event
        self.sampling_strategy = self._is_sampling_strategy_valid(sampling_strategy)
        self.weighting_function = self._is_weighting_function_valid(weighting_func)
        self.ranking_strategy = self._is_ranking_strategy_valid(ranking_strategy)

    # Core methods

    def fit(self, sessions: Dict, items: Dict):
        """Sets input session-items and item-sessions maps.

        Parameters
        ----------
        sessions : Dict
                   sessions = {
                       session_id: (
                           [ sequence_of_items ],
                           [ sequence_of_timestamps ],
                           [ [OPTIONAL] sequence_of_event_types ]
                       )
                   }

        items : Dict
                items = {
                    item_id: (
                        [ sequence_of_sessions ],
                        [ sequence_of_the_first_session_timestamps ]
                    )
                }
        """

        # Check input data
        self._check_sessions_input(sessions)
        self._check_items_input(items)

        self.session_item_map = sessions
        self.item_session_map = items

    def predict(self, sessions: Union[Dict, List],
                number_of_recommendations=None,
                number_of_closest_neighbors=None,
                session_sampling_strategy=None,
                possible_neighbors_sample_size=None,
                weighting_strategy=None,
                rank_strategy=None,
                required_sampling_event=None) -> Dict:
        """
        The method predicts n next recommendations from a given session.

        Parameters
        ----------
        sessions : Union[Dict, List]
                   Sequence of items for recommendation. As a dict, it is a user id (key) and products and
                   their timestamps (values). As a List it is a nested List of user actions and their timestamps,
                   and additional properties.

        number_of_recommendations : int or None, default=None
                                    Resets the number of recommendations.

        number_of_closest_neighbors : int or None, default=None
                                      Resets the number of closest neighbors.

        session_sampling_strategy : str or None, default=None
                                    How to filter the initial sample of sessions. Available strategies are:
                                        - 'common_items': sample sessions with the same items as the input session,
                                        - 'recent': sample the most actual sessions,
                                        - 'random': get a random sample of sessions.
                                        
        possible_neighbors_sample_size : int or None, default=None
                                         How many sessions from the model are sampled to make a recommendation. If not
                                         set then it is the number given during the class initilization.

        weighting_strategy : str or None, default=None
                             The similarity measurement between sessions. Available options: 'linear', 'log'
                             and 'quadratic'. If not set then weighting strategy is taken from the parameter set
                             during the class initialization.

        rank_strategy : str or None, default=None
                        How we calculate an item rank (based on its position in a session sequence). Available options
                        are: 'inv', 'linear', 'log', 'quadratic'. If not set then model uses rank strategy given
                        during the initilization.

        required_sampling_event : int or str, default = None
                                  Set this paramater to the event name if sessions with it must be included in
                                  the neighbors selection. For example, this event may be a "purchase".

        Returns
        -------
        : Dict
          ranked items in descending order
          - {user ID: [[item, rank] ...]} (input as a Dict)
          - {int: [[item, rank] ...]} (input as a List)
        """

        # Set class consts
        self._reset_model_params(number_of_recommendations,
                                 number_of_closest_neighbors,
                                 session_sampling_strategy,
                                 possible_neighbors_sample_size,
                                 weighting_strategy,
                                 rank_strategy,
                                 required_sampling_event)

        output_ranks = dict()
        if isinstance(sessions, dict):
            for _key, session in sessions.items():
                recommendations = self._part_predict(session)
                output_ranks[_key] = recommendations
        elif isinstance(sessions, list):
            for idx, session in enumerate(sessions):
                recommendations = self._part_predict(session)
                output_ranks[idx] = recommendations
        else:
            raise TypeError('Not supported input type for prediction')
        return output_ranks

    def _part_predict(self, session):
        neighbors = self._nearest_neighbors(session)
        ranked_items = self._rank_items(neighbors, session)
        ranked_items.sort(key=lambda x: x[1], reverse=True)
        recommendations = ranked_items[:self.n_of_recommendations]
        return recommendations

    # Settings
    @staticmethod
    def _check_sessions_input(sessions: Dict):
        """Check if sessions have required dimensions.

        Parameters
        ----------
        sessions : Dict
                   sessions = {
                       session_id: (
                           [ sequence_of_items ],
                           [ sequence_of_timestamps ],
                           [ [OPTIONAL] sequence_of_event_type ]
                       )
                   }

        Raises
        ------
        InvalidDimensionsError
            Number of nested sequences per record is different than 2 or 3.

        InvalidTimestampError
            Wrong type of timestamp - int is required.

        TypeError
            Wrong type of a given structure.
        """

        # Get sample record
        sample_key = random.choice(list(sessions.keys()))

        sample_rec = sessions[sample_key]

        # Check dimensions
        test_dims = check_data_dimension(sample_rec, 2) or check_data_dimension(sample_rec, 3)

        if not test_dims:
            raise InvalidDimensionsError('Session-items map', [2, 3])

        # Check type
        for subrec in sample_rec:
            if not isinstance(subrec, Iterable):
                msg = f'Expected iterable as a part of session-items map record, got {type(subrec)} instead.'
                raise TypeError(msg)

        # Check timestamp type
        for tstamp in sample_rec[1]:
            if not check_numeric_type_instance(tstamp):
                raise InvalidTimestampError(tstamp)

    @staticmethod
    def _check_items_input(items: dict):
        """Check if sessions have required dimensions.

        Parameters
        ----------
        items : dict
                items = {
                    item_id: (
                        [ sequence_of_sessions ],
                        [ sequence_of_the_first_session_timestamps ]
                    )
                }

        Raises
        ------
        InvalidDimensionsError
            Number of nested sequences per record is different than 2.

        InvalidTimestampError
            Wrong type of timestamp - int is required.

        TypeError
            Wrong type of a given structure.

        """

        # Get sample record
        sample_key = random.choice(list(items.keys()))

        sample_rec = items[sample_key]

        # Check dimensions
        test_dims = check_data_dimension(sample_rec, 2)

        if not test_dims:
            raise InvalidDimensionsError('Item-sessions map', 2)

        # Check type
        for subrec in sample_rec:
            if not isinstance(subrec, Iterable):
                msg = f'Expected iterable as a part of item-sessions map record, got {type(subrec)} instead.'
                raise TypeError(msg)

        # Check timestamp type
        for tstamp in sample_rec[1]:
            if not check_numeric_type_instance(tstamp):
                raise InvalidTimestampError(tstamp)

    def _is_sampling_strategy_valid(self, sampling_strategy):
        """Check sampling strategy.

        Parameters
        ----------
        sampling_strategy : str

        Raises
        ------
        KeyError
            Strategy not in list defined in self.sampling_strategies

        Returns
        -------
        sampling_strategy : str
        """
        if sampling_strategy in self.sampling_strategies:
            return sampling_strategy
        else:
            msg = f"Given sampling strategy {sampling_strategy} not implemented in the package. " \
                  f"Use one of {self.sampling_strategies} instead"
            raise KeyError(msg)

    def _is_ranking_strategy_valid(self, strategy: str) -> str:
        """Check if ranking strategy is valid.

        Parameters
        ----------
        strategy : str

        Raises
        ------
        KeyError
            Strategy not in list ['linear', 'log', 'quadratic', 'inv']

        Returns
        strategy : str
        """
        if strategy in self.ranking_strategies:
            return strategy
        else:
            msg = f"Given ranking strategy {strategy} not implemented in the package. " \
                  f"Use 'linear', 'log', 'quadratic', 'inv' instead"
            raise KeyError(msg)

    def _is_weighting_function_valid(self, wfunc: str) -> str:
        """Check if weighting function is valid.

        Parameters
        ----------
        wfunc : str

        Raises
        ------
        KeyError
            Weighting function not in list ['linear', 'log', 'quadratic']

        Returns
        wfunc : str
        """
        if wfunc in self.weighting_functions:
            return wfunc
        else:
            msg = f"Given weighting function {wfunc} is not implemented in the package. " \
                  f"Use 'linear', 'log', 'quadratic', instead"
            raise KeyError(msg)

    def _reset_model_params(self,
                            number_of_recommendations,
                            number_of_closest_neighbors,
                            session_sampling_strategy,
                            possible_neighbors_sample_size,
                            weighting_strategy,
                            rank_strategy,
                            required_sampling_event):
        """Methods resets and maps new model parameters.

        Parameters
        ----------
        number_of_recommendations : int

        number_of_closest_neighbors : int

        session_sampling_strategy : str

        possible_neighbors_sample_size : int

        weighting_strategy : str

        rank_strategy : str

        required_sampling_event : str or int

        Raises
        -------
        TypeError
            Wrong type of an input parameter.

        KeyError
            Wrong name of sampling strategy, ranking strategy or weighting function.

        """
        if number_of_recommendations is not None:
            if isinstance(number_of_recommendations, int):
                self._set_n_of_recs(number_of_recommendations)
            else:
                raise TypeError(f'Number of output recommendations should be an integer, '
                                f'got {type(number_of_recommendations)} instead')

        if number_of_closest_neighbors is not None:
            if isinstance(number_of_closest_neighbors, int):
                self._set_number_of_closest_neighbors(number_of_closest_neighbors)
            else:
                raise TypeError(f'Number of closest neighbors should be an integer, '
                                f'got {type(number_of_closest_neighbors)} instead')

        if session_sampling_strategy is not None:
            if isinstance(session_sampling_strategy, str):
                session_sampling_strategy = self._is_sampling_strategy_valid(session_sampling_strategy)
                self._set_sampling_strategy(session_sampling_strategy)
            else:
                raise TypeError(f'Defined sampling strategy should be a string. '
                                f'Got {type(session_sampling_strategy)} instead')

        if possible_neighbors_sample_size is not None:
            if isinstance(possible_neighbors_sample_size, int):
                self._set_possible_neighbors_sample_size(possible_neighbors_sample_size)
            else:
                raise TypeError(f'Number of possible neighbors should be an integer, '
                                f'got {type(possible_neighbors_sample_size)} instead')

        if weighting_strategy is not None:
            if isinstance(weighting_strategy, str):
                weighting_strategy = self._is_weighting_function_valid(weighting_strategy)
                self._set_weighting_strategy(weighting_strategy)
            else:
                raise TypeError(f'Defined weighting function should be a string. '
                                f'Got {type(weighting_strategy)} instead')

        if rank_strategy is not None:
            if isinstance(rank_strategy, str):
                rank_strategy = self._is_ranking_strategy_valid(rank_strategy)
                self._set_ranking_strategy(rank_strategy)
            else:
                raise TypeError(f'Defined ranking function should be a string. '
                                f'Got {type(rank_strategy)} instead')

        if required_sampling_event is not None:
            if isinstance(required_sampling_event, str) or isinstance(required_sampling_event, int):
                self.required_sampling_event = required_sampling_event
            else:
                raise TypeError('Defined required sampling event can be int or str, other datatypes are not supported')

    def _set_n_of_recs(self, n):
        self.n_of_recommendations = n

    def _set_number_of_closest_neighbors(self, n):
        self.number_of_closest_neighbors = n

    def _set_possible_neighbors_sample_size(self, sample_size):
        self.possible_neighbors_sample_size = sample_size

    def _set_sampling_strategy(self, sampling_strategy):
        self.sampling_strategy = sampling_strategy

    def _set_weighting_strategy(self, weighting_func):
        weighting_func = self._is_weighting_function_valid(weighting_func)
        self.weighting_function = weighting_func

    def _set_ranking_strategy(self, rank_strategy):
        self.ranking_strategy = rank_strategy

    # Transform, sample, rank - core

    def _nearest_neighbors(self, session: List) -> List:
        """Method searches for nearest neighbors for a given session.

        Parameters
        ----------
        session : List
                  Session is a sequence of products, events (product view / click) and timestamps of those events:

            session = [
                [sequence_of_items],
                [sequence_of_timestamps],
                [sequence_of_event_type]
            ]

        Returns
        -------
        : List
          n closest sessions, where n - number of closest sessions or length of ranked session if smaller than
          number of closest sessions.
        """
        possible_neighbor_sessions = self._possible_neighbors(session)
        items_sequence = session[0]
        rank = self._calculate_similarity(items_sequence, possible_neighbor_sessions)
        rank.sort(key=lambda x: x[1], reverse=True)
        length = len(rank)
        idx = min(self.number_of_closest_neighbors, length)
        return rank[0:idx]

    def _possible_neighbors(self, session: List) -> List:
        """Get set of possible neighbors based on the item similarity.

        Parameters
        ----------
        session : List
                  Session is a sequence of products, events (product view / click) and timestamps of those events:

            session = [
                [sequence_of_items],
                [sequence_of_timestamps],
                [sequence_of_event_type]
            ]

        Returns
        -------
        : List
          Sample of possible neighbors. Sampling controlled by the sampling_strategy attribute.
        """
        session_items = set(session[0])
        common_sessions = set()
        for s_item in session_items:
            if s_item in self.item_session_map:
                s_item_sessions = set(self.item_session_map[s_item][0])
                common_sessions |= s_item_sessions

        # Filter session by event if needed
        if self.required_sampling_event is not None:
            common_sessions = self._get_sessions_with_event(common_sessions)

        sample_subset = self._sample_possible_neighbors(common_sessions, session)
        return sample_subset

    def _rank_items(self, closest_neighbors: List, session: List) -> List:
        """Function ranks given items to return the best recommendation results.

        Parameters
        ----------
        closest_neighbors : List
                            The closest sessions ranked by similarity to a given session.

        session : List
                  User session.

        Returns
        -------
        : List
          List of rated items in descending order.
        """
        session_items = session[0]
        scores = dict()

        for neighbor in closest_neighbors:
            n_items = self.session_item_map[neighbor[0]][0]
            step = 1
            decay = 1
            for s_item in reversed(session_items):
                if s_item in n_items:
                    decay = weight_item_score(self.ranking_strategy, step)
                    break
                step = step + 1

            for n_item in n_items:
                if n_item in session_items:
                    # We do not want to score items viewed by the user
                    # TODO: condition to control this behavior
                    pass
                else:
                    old_score = scores.get(n_item)
                    new_score = neighbor[1]
                    # TODO: idf weighting
                    new_score = new_score * decay

                    if old_score is not None:
                        new_score = old_score + new_score

                    scores.update({n_item: new_score})

        rank = list()
        for k, v in scores.items():
            rank.append((k, v))
        return rank

    # Transform, sample, rank - additional

    def _calculate_similarity(self, session_items: List, possible_neighbours: List) -> List:
        """Function calculates similarity between sessions based on the items ranking.

        Parameters
        ----------
        session_items : List
                        List of items from the customer session.

        possible_neighbours : List
                              List of sessions from the possible neighbors pool (based on a sampling strategy).

        Returns
        -------
        neighbors : List
                    List of similar sessions to the customer session.
        """

        pos_weights = dict()
        length = len(session_items)

        for idx, item in enumerate(session_items):
            count = idx + 1
            pos_weights[item] = weight_session_items(self.weighting_function, count, length)

        items = set(session_items)
        neighbours = []
        for other in possible_neighbours:
            other_items = set(self.session_item_map[other][0])
            similarity = weight_set_pair(items, other_items, pos_weights)
            neighbours.append([other, similarity])

        return neighbours

    def _get_sessions_with_event(self, raw_sessions: Set) -> Set:
        """Method parses available input sessions based on the occurence of a specific event and returns list of
        unique sessions with this event.

        Parameters
        ----------
        raw_sessions : Set

        Returns
        -------
        unique_session_with_event : Set

        """
        new_sessions = set()

        for sess in raw_sessions:
            session_sample = self.session_item_map[sess]
            if self.required_sampling_event in session_sample[2]:
                new_sessions.add(sess)

        return new_sessions

    def _sampling_common(self, sessions: Set, session: List) -> List:
        """Function gets the most similar sessions based on the number of common elements between sessions.

        Parameters
        ----------
        sessions : set
                   Unique sessions.

        session : List
                  The customer session.

        Returns
        -------
        : List
          List of n possible sessions with the same items as a customer session.
        """
        rank = [(ses, len(set(self.session_item_map[ses][0]) & set(session[0]))) for ses in sessions]
        rank.sort(key=lambda x: x[1])
        result = [x[0] for x in rank]
        sample_size = min(self.possible_neighbors_sample_size, len(sessions))
        return result[:sample_size]

    def _sample_possible_neighbors(self, all_sessions: Set, session: List) -> List:
        """Method samples possible neighbors.

        Parameters
        ----------
        all_sessions : set
                       All sessions to sample possible neighbors.

        session : List
                  Customer session.

        Returns
        -------
        : List
          subset of possible neighbors
        """

        if self.sampling_strategy == 'random':
            return self._sampling_random(all_sessions)
        elif self.sampling_strategy == 'recent':
            return self._sampling_recent(all_sessions)
        elif self.sampling_strategy == 'common_items':
            return self._sampling_common(all_sessions, session)
        else:
            err_msg = f'Defined sampling strategy {self.sampling_strategy} not implemented. Available strategies are:' \
                      f' {self.sampling_strategies}.'
            raise TypeError(err_msg)

    def _sampling_random(self, sessions: Set) -> List:
        """Get random sessions from the sessions space. This method is good to estimate model performance or to test it.

        Parameters
        ----------
        sessions : set
                   Unique sessions.

        Returns
        -------
        : List
          Random sample of self.possible_neighbors_sample_size sessions.
        """

        sample_size = min(self.possible_neighbors_sample_size, len(sessions))
        sample = random.sample(sessions, sample_size)

        return sample

    def _sampling_recent(self, sessions: Set) -> List:
        """Get most recent sessions from the possible neighbors.

        Parameters
        ----------
        sessions : set
                   Unique sessions.

        Returns
        -------
        : List
          Most recent sessions. Sample of size possible_neighbors_sample_size.
        """
        rank = [(sid, self.session_item_map[sid][1]) for sid in sessions]
        rank.sort(key=lambda x: x[1], reverse=True)
        result = [x[0] for x in rank]
        sample_size = min(self.possible_neighbors_sample_size, len(sessions))
        return result[:sample_size]
