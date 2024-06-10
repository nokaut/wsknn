import random
import numpy as np
from typing import Iterable, Union, List, Set, Dict, Tuple

from wsknn.model.validators import validate_mapping_dtypes
from wsknn.preprocessing.structure.session_to_item_map import (
    map_sessions_to_items)
from wsknn.weighting import weight_session_items, weight_item_score
from wsknn.utils.calc import weight_set_pair
from wsknn.utils.errors import (check_data_dimension,
                                check_numeric_type_instance,
                                InvalidDimensionsError,
                                InvalidTimestampError)


class WSKNN:
    """
    The class represents the Weighted Session-Based k-nn model.

    Parameters
    ----------
    number_of_recommendations : int, default=5
        The number of recommended items.

    number_of_neighbors : int, default=10
        The number of the closest sessions to choose the items from.

    sampling_strategy : str, default='common_items'
        How to filter the initial sample of sessions. Available strategies are:

        - ``'common_items'``: sample sessions with the same items as the input
          session,
        - ``'recent'``: sample the most actual sessions,
        - ``'random'``: get a random sample of sessions,
        - ``'weighted_events'``: select sessions based on the specific weights
          assigned to events.

    sample_size : int, default=1000
        How many sessions from the model are sampled to make a recommendation.

    weighting_func : str, default='linear'
        The similarity measurement between sessions. Available options:
        ``'linear'``, ``'log'`` and ``'quadratic'``.

    ranking_strategy : str, default='linear'
        How we calculate an item rank (based on its position in a session
        sequence). Available options are:
        ``'inv'``, ``'linear'``, ``'log'``, ``'quadratic'``.

    return_events_from_session : bool, default = True
        Should algorithm return the same items as in session if there are no
        neighbors?

    required_sampling_event : int or str, default = None
        Set this parameter to the event name if sessions with this event must
        be included in the neighbors' selection.
        For example, an event name may be the ``"purchase"``.

    required_sampling_event_index : int, default = None
        If ``required_sampling_event`` parameter is filled then you must pass
        an index of a row with event names.

    sampling_event_weights_index : int, default = None
        If ``sampling_strategy`` is set to ``weighted_events`` then you must
        pass an index of a row with event weights.

    recommend_any : bool, default = False
        If recommender picks fewer items than required by the
        ``number_of_recommendations`` then add random items to the results.

    Attributes
    ----------
    weighting_functions: List
        The weighting functions: ``['linear', 'log', 'quadratic']``.

    ranking_strategies : List
        The ranking strategies: ``['linear', 'log', 'quadratic', 'inv']``.

    session_item_map : Dict
        The map of items that occur in the session and their timestamps,
        and (optional) their types and their
        weights.
        >>> sessions = {
        ...     session_id: (
        ...     [sequence_of_items],
        ...     [sequence_of_timestamps],
        ...     [(optional) event names (types)],
        ...     [(optional) weights]
        ...     )
        ... }

    item_session_map : Dict
        The map of items and the sessions where those items are present,
        and the first timestamp of those sessions.
        >>> items = {
        ...     item_id: (
        ...     [sequence_of_sessions],
        ...     [sequence_of_the_first_session_timestamps]
        ...     )
        ... }

    n_of_recommendations : int
        The number of items recommended.

    number_of_closest_neighbors : int
        See ``number_of_neighbors`` parameter.

    sampling_strategy : str
        See ``sampling_strategy`` parameter.

    possible_neighbors_sample_size : int
        See ``sample_size`` parameter.

    weighting_function : str
        See ``weighting_func`` parameter.

    ranking_strategy : str
        See ``ranking_strategy`` parameter.

    return_events_from_session : bool, default = True
        See ``return_events_from_session`` parameter.

    required_sampling_event : Union[int, str], default = None
        See ``required_sampling_event`` parameter.

    required_sampling_event_index : int, default = None
        See ``required_sampling_event_index`` parameter.

    sampling_event_weights_index : int, default = None
        See ``sampling_str_event_weights_index`` parameter.

    recommend_any : bool, default = False
        See ``recommend_any`` parameter.

    Methods
    -------
    fit()
        Sets input session-items and item-sessions maps.

    recommend()
        The method predicts the ``n`` next recommendations from a given
        session.

    set_model_params()
        Methods resets and maps the new model parameters.

    Raises
    ------
    InvalidDimensionsError
        Wrong number of nested sequences within session-items or
        item-sessions maps.

    InvalidTimestampError
        Wrong type of timestamp - ``int`` type is required.

    TypeError
        Wrong type of nested structures within session-items or
        item-sessions maps.

    IndexError
        Wrong index of event names or wrong index of event weights.

    """

    def __init__(self,
                 number_of_recommendations: int = 5,
                 number_of_neighbors: int = 10,
                 sampling_strategy: str = 'common_items',
                 sample_size: int = 1000,
                 weighting_func: str = 'linear',
                 ranking_strategy: str = 'linear',
                 return_events_from_session: bool = True,
                 required_sampling_event: Union[int, str] = None,
                 required_sampling_event_index: int = None,
                 sampling_event_weights_index: int = None,
                 recommend_any: bool = False):

        # CHECKS

        # Check if all parameters are given: required sampling event

        if required_sampling_event is not None:
            # User must provide index of event names
            if required_sampling_event_index is None:
                msg = ('With required sampling event given you must '
                       'provide index of a row with names of events!')
                raise IndexError(msg)

        # Check if all parameters are given:
        # sampling_strategy == 'weighted_events'
        if sampling_strategy == 'weighted_events':
            if sampling_event_weights_index is None:
                msg = ('If you want to sample sessions based '
                       'on the weights then you must provide index of '
                       'the row with weights')
                raise IndexError(msg)

        # INITILIAZATION

        self.sampling_strategies = ['common_items',
                                    'recent',
                                    'random',
                                    'weighted_events']
        self.weighting_functions = ['linear',
                                    'log',
                                    'quadratic']
        self.ranking_strategies = ['linear',
                                   'log',
                                   'quadratic',
                                   'inv']

        self.session_item_map = None
        self.item_session_map = None

        self.n_of_recommendations = number_of_recommendations
        self.number_of_closest_neighbors = number_of_neighbors
        self.possible_neighbors_sample_size = sample_size

        self.required_sampling_event = required_sampling_event
        self.sampling_strategy = self._is_sampling_strategy_valid(
            sampling_strategy)
        self.weighting_function = self._is_weighting_function_valid(
            weighting_func)
        self.ranking_strategy = self._is_ranking_strategy_valid(
            ranking_strategy)
        self.return_events_from_session = return_events_from_session
        self.sampling_event_weights_index = sampling_event_weights_index
        self.required_sampling_event_index = required_sampling_event_index
        self.recommend_any = recommend_any

    # Core methods
    def fit(self,
            sessions: Dict,
            items: Dict = None):
        """
        Sets input session-items and item-sessions maps.

        Parameters
        ----------
        sessions : Dict
            The map of items that occur in the session and their
            timestamps, and (optional) their types and their weights.

            >>> sessions = {
            ...     session_id: (
            ...     [sequence_of_items],
            ...     [sequence_of_timestamps],
            ...     [(optional) event names (types)],
            ...     [(optional) weights]
            ...     )
            ... }

        items : Dict
            The map of items and the sessions where those items are present,
            and the first timestamp of those sessions. If not provided then
            the item-sessions map is created from the ``sessions`` parameter.

            >>> items = {
            ...     item_id: (
            ...     [sequence_of_sessions],
            ...     [sequence_of_the_first_session_timestamps]
            ...     )
            ... }
        """

        # Check input data
        self._check_sessions_input(sessions)

        if items is None:
            items = map_sessions_to_items(sessions)
        else:
            # Validate dtypes
            validate_mapping_dtypes(
                sessions=sessions,
                items=items
            )

        self._check_items_input(items)

        self.session_item_map = sessions
        self.item_session_map = items

    def recommend(self,
                  event_stream: Union[List, Tuple, np.ndarray, Dict],
                  settings: dict = None) -> Union[List, Dict]:
        """
        The method predicts n next recommendations from a given session.

        Parameters
        ----------
        event_stream : ArrayLike, Dict
            Sequence of items for recommendation. If list then it is treated
            as a single recommendation:

            >>> [
            ...     [sequence_of_items],
            ...     [sequence_of_timestamps],
            ...     [(optional) event names (types)],
            ...     [(optional) weights]
            ... ]

            If it is a dictionary then recommendations are done in batch.
            Every key in a dictionary is user-index, and value is a list
            with sequence of items, timestamps and optional features.

            >>> {
            ...     "user A": [...],
            ...     "user B": [...]
            ... }

        settings : Dict, default = None
            Model settings and parameters.

        Returns
        -------
        recommendations : List, Dict

            Output for the single input (list):
            >>> [
            ...     (item a, rank a), (item b, rank b)
            ... ]

            Output for the input with multiple users (dictionary):
            >>> {
            ...     "user A": [(item a, rank a), (item b, rank b)],
            ...     "...": [...]
            ... }
        """

        if settings is not None:
            self.set_model_params(**settings)

        if isinstance(event_stream, Dict):
            recommendations = {
                _k: self._predict(rec) for _k, rec in event_stream.items()
            }
        elif isinstance(event_stream, List) or isinstance(event_stream, Tuple) or isinstance(event_stream, np.ndarray):
            recommendations = self._predict(event_stream)
        else:
            raise NotImplementedError('Recommendation can be done only'
                                      ' for list or dictionary as an '
                                      'input.')

        return recommendations

    def set_model_params(self,
                         number_of_recommendations=None,
                         number_of_neighbors=None,
                         sampling_strategy=None,
                         sample_size=None,
                         weighting_func=None,
                         ranking_strategy=None,
                         return_events_from_session=None,
                         required_sampling_event=None,
                         recommend_any=False):
        """
        Methods resets and maps the new model parameters.

        Parameters
        ----------
        number_of_recommendations : int, default = None

        number_of_neighbors : int, default = None

        sampling_strategy : str, default = None

        sample_size : int, default = None

        weighting_func : str, default = None

        ranking_strategy : str, default = None

        return_events_from_session : bool, default = None

        required_sampling_event : str or int, default = None

        recommend_any : bool, default = None

        Raises
        ------
        TypeError
            Wrong input parameter type.

        KeyError
            Wrong name of sampling strategy, ranking strategy or weighting
            function.

        """
        if number_of_recommendations is not None:
            if isinstance(number_of_recommendations, int):
                self._set_n_of_recs(number_of_recommendations)
            else:
                raise TypeError(f'Number of output recommendations should '
                                f'be an integer, '
                                f'got {type(number_of_recommendations)} '
                                f'instead')

        if number_of_neighbors is not None:
            if isinstance(number_of_neighbors, int):
                self._set_number_of_closest_neighbors(number_of_neighbors)
            else:
                raise TypeError(f'Number of closest neighbors should '
                                f'be an integer, '
                                f'got {type(number_of_neighbors)} instead')

        if sampling_strategy is not None:
            if isinstance(sampling_strategy, str):
                session_sampling_strategy = self._is_sampling_strategy_valid(
                    sampling_strategy)
                self._set_sampling_strategy(session_sampling_strategy)
            else:
                raise TypeError(f'Defined sampling strategy should be '
                                f'a string. '
                                f'Got {type(sampling_strategy)} instead')

        if sample_size is not None:
            if isinstance(sample_size, int):
                self._set_possible_neighbors_sample_size(sample_size)
            else:
                raise TypeError(f'Number of possible neighbors should be '
                                f'an integer, '
                                f'got {type(sample_size)} instead')

        if weighting_func is not None:
            if isinstance(weighting_func, str):
                weighting_strategy = self._is_weighting_function_valid(
                    weighting_func)
                self._set_weighting_strategy(weighting_strategy)
            else:
                raise TypeError(f'Defined weighting function should be '
                                f'a string. '
                                f'Got {type(weighting_func)} instead')

        if ranking_strategy is not None:
            if isinstance(ranking_strategy, str):
                rank_strategy = self._is_ranking_strategy_valid(
                    ranking_strategy)
                self._set_ranking_strategy(rank_strategy)
            else:
                raise TypeError(f'Defined ranking function should be a string.'
                                f' Got {type(ranking_strategy)} instead')

        if return_events_from_session is not None:
            if isinstance(return_events_from_session, bool):
                self.return_events_from_session = return_events_from_session
            else:
                raise TypeError(f'return_events_from_session parameter should '
                                f'be set only to True or False (bool). '
                                f'Got type {type(return_events_from_session)} '
                                f'instead')

        if required_sampling_event is not None:
            if isinstance(required_sampling_event, str) or isinstance(
                    required_sampling_event, int):
                self.required_sampling_event = required_sampling_event
            else:
                raise TypeError('Defined required sampling event can be int '
                                'or str, other datatypes are not supported')

        if recommend_any is not None:
            if isinstance(recommend_any, bool):
                self.recommend_any = recommend_any
            else:
                raise TypeError(f'recommend_any parameter should be set only '
                                f'to True or False (bool). '
                                f'Got type {type(recommend_any)} instead')

    def _predict(self, session):
        neighbors = self._nearest_neighbors(session)

        if len(neighbors) == 0:
            if self.recommend_any:
                recs = list()
                recommendations = self._get_more_items(recs)
                return recommendations
        else:
            ranked_items = self._rank_items(neighbors, session)
            ranked_items.sort(key=lambda x: x[1], reverse=True)
            recommendations = ranked_items[:self.n_of_recommendations]

            if self.recommend_any:
                if len(recommendations) < self.n_of_recommendations:
                    recommendations = self._get_more_items(recommendations)

            return recommendations

    def _get_more_items(self, recommendations):
        add_items_size = self.n_of_recommendations - len(recommendations)
        possible_items = list(self.item_session_map.keys())
        for _ in range(add_items_size):
            rnd_item = random.choice(possible_items)
            rnd_rec = (rnd_item, 0.0)
            recommendations.append(rnd_rec)
        return recommendations

    # Settings
    @staticmethod
    def _check_sessions_input(sessions: Dict):
        """Check if sessions have required dimensions.

        Parameters
        ----------
        sessions : Dict
            The map of items that occur in the session and their
            timestamps, and (optional) their types and their weights.

            >>> sessions = {
            ...     session_id: (
            ...     [sequence_of_items],
            ...     [sequence_of_timestamps],
            ...     [(optional) event names (types)],
            ...     [(optional) weights]
            ...     )
            ... }

        Raises
        ------
        InvalidDimensionsError
            Number of nested sequences per record is different from 2 or 3.

        InvalidTimestampError
            Wrong type of timestamp - int is required.

        TypeError
            Wrong type of given structure.
        """

        # Get sample record
        sample_key = random.choice(list(sessions.keys()))

        sample_rec = sessions[sample_key]

        # Check dimensions
        test_dims = check_data_dimension(sample_rec, 2)

        if not test_dims:
            raise InvalidDimensionsError('Session-items map', 2)

        # Check type
        for subrec in sample_rec:
            if not isinstance(subrec, Iterable):
                msg = (f'Expected iterable as a part of session-items map '
                       f'record, got {type(subrec)} instead.')
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
        items : Dict
            The map of items and the sessions where those items are present,
            and the first timestamp of those sessions.

            >>> items = {
            ...     item_id: (
            ...     [sequence_of_sessions],
            ...     [sequence_of_the_first_session_timestamps]
            ...     )
            ... }

        Raises
        ------
        InvalidDimensionsError
            Number of nested sequences per record is different from 2.

        InvalidTimestampError
            Wrong type of timestamp - int is required.

        TypeError
            Wrong type of given structure.

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
                msg = (f'Expected iterable as a part of item-sessions map '
                       f'record, got {type(subrec)} instead.')
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
            Strategy not in list defined in ``self.sampling_strategies``

        Returns
        -------
        sampling_strategy : str
        """
        if sampling_strategy in self.sampling_strategies:
            return sampling_strategy
        else:
            msg = (f"Given sampling strategy {sampling_strategy} not "
                   f"implemented in the package. "
                   f"Use one of {self.sampling_strategies} instead")
            raise KeyError(msg)

    def _is_ranking_strategy_valid(self, strategy: str) -> str:
        """Check if ranking strategy is valid.

        Parameters
        ----------
        strategy : str

        Raises
        ------
        KeyError
            Strategy not in list ``['linear', 'log', 'quadratic', 'inv']``

        Returns
        strategy : str
        """
        if strategy in self.ranking_strategies:
            return strategy
        else:
            msg = (f"Given ranking strategy {strategy} not implemented "
                   f"in the package. "
                   f"Use 'linear', 'log', 'quadratic', 'inv' instead")
            raise KeyError(msg)

    def _is_weighting_function_valid(self, wfunc: str) -> str:
        """Check if weighting function is valid.

        Parameters
        ----------
        wfunc : str

        Raises
        ------
        KeyError
            Weighting function not in list ``['linear', 'log', 'quadratic']``

        Returns
        wfunc : str
        """
        if wfunc in self.weighting_functions:
            return wfunc
        else:
            msg = (f"Given weighting function {wfunc} is not "
                   f"implemented in the package. "
                   f"Use 'linear', 'log', 'quadratic', instead")
            raise KeyError(msg)

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
            Session is a sequence of products, events (product view / click)
            and timestamps of those events:

            >>> session = [
            ...     [sequence_of_items],
            ...     [sequence_of_timestamps],
            ...     [optional sequence_of_event_type],
            ...     [optional sequence of weights]
            ... ]

        Returns
        -------
        rank : List
            The ``n`` closest sessions, where ``n`` - number of the closest
            sessions or length of ranked session if
            smaller than number of the closest sessions.
        """
        possible_neighbor_sessions = self._possible_neighbors(session)
        items_sequence = session[0]
        rank = self._calculate_similarity(items_sequence,
                                          possible_neighbor_sessions)
        rank.sort(key=lambda x: x[1], reverse=True)
        length = len(rank)
        idx = min(self.number_of_closest_neighbors, length)
        return rank[0:idx]

    def _possible_neighbors(self, session: List) -> List:
        """Get set of possible neighbors based on the item similarity.

        Parameters
        ----------
        session : List
            Session is a sequence of products, events (product view / click)
            and timestamps of those events:

            >>> session = [
            ...     [sequence_of_items],
            ...     [sequence_of_timestamps],
            ...     [optional sequence_of_event_type],
            ...     [optional sequence of weights]
            ... ]

        Returns
        -------
        sample_subset : List
          Sample of possible neighbors. Sampling controlled by the
          ``sampling_strategy`` attribute.
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

        sample_subset = self._sample_possible_neighbors(common_sessions,
                                                        session)
        return sample_subset

    def _rank_items(self, closest_neighbors: List, session: List) -> List:
        """Function ranks given items to return the best recommendation
        results.

        Parameters
        ----------
        closest_neighbors : List
            The closest sessions ranked by similarity to a given session.

        session : List
            User session.

        Returns
        -------
        rank : List
            The list of rated items in descending order.
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
                if (
                        n_item in session_items
                ) and (
                        not self.return_events_from_session
                ):
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

    def _calculate_similarity(self, session_items: List,
                              possible_neighbours: List) -> List:
        """Function calculates similarity between sessions based on the items
        ranking.

        Parameters
        ----------
        session_items : List
            List of items from the customer session.

        possible_neighbours : List
            List of sessions from the possible neighbors pool (based on the
            sampling strategy).

        Returns
        -------
        neighbors : List
            List of similar sessions to the customer session.
        """

        pos_weights = dict()
        length = len(session_items)

        for idx, item in enumerate(session_items):
            count = idx + 1
            pos_weights[item] = weight_session_items(self.weighting_function,
                                                     count,
                                                     length)

        items = set(session_items)
        neighbours = []
        for other in possible_neighbours:
            other_items = set(self.session_item_map[other][0])
            similarity = weight_set_pair(items, other_items, pos_weights)
            neighbours.append([other, similarity])

        return neighbours

    def _get_sessions_with_event(self, raw_sessions: Set) -> Set:
        """Method parses available input sessions based on the occurrence of
        a specific event and returns list of unique sessions with this event.

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
            if self.required_sampling_event in session_sample[
                self.required_sampling_event_index
            ]:
                new_sessions.add(sess)

        return new_sessions

    def _sampling_common(self, sessions: Set, session: List) -> List:
        """Function gets the most similar sessions based on the number of
        common elements between sessions.

        Parameters
        ----------
        sessions : Set
            Unique sessions.

        session : List
            The customer session.

        Returns
        -------
        result : List
            List of ``n`` possible sessions with the same items as a customer
            session.
        """
        rank = [(ses, len(
            set(self.session_item_map[ses][0]) & set(session[0]))
                 ) for ses in sessions]
        rank.sort(key=lambda x: x[1])
        result = [x[0] for x in rank]
        sample_size = min(self.possible_neighbors_sample_size, len(sessions))
        return result[:sample_size]

    def _sample_possible_neighbors(self,
                                   all_sessions: Set,
                                   session: List) -> List:
        """Method samples possible neighbors.

        Parameters
        ----------
        all_sessions : set
            All sessions to sample possible neighbors.

        session : List
            Customer session.

        Returns
        -------
        sample : List
            The subset of possible neighbors
        """

        if self.sampling_strategy == 'random':
            return self._sampling_random(all_sessions)
        elif self.sampling_strategy == 'recent':
            return self._sampling_recent(all_sessions)
        elif self.sampling_strategy == 'common_items':
            return self._sampling_common(all_sessions, session)
        elif self.sampling_strategy == 'weighted_events':
            return self._sampling_weighted_events(all_sessions)
        else:
            err_msg = (f'Defined sampling strategy {self.sampling_strategy} '
                       f'not implemented. Available strategies are:'
                       f' {self.sampling_strategies}.')
            raise TypeError(err_msg)

    def _sampling_random(self, sessions: Set) -> List:
        """Get random sessions from the sessions space. This method is good to
        estimate model performance.

        Parameters
        ----------
        sessions : set
            Unique sessions.

        Returns
        -------
        sample : List
            Random sample of ``self.possible_neighbors_sample_size`` sessions.
        """

        sessions = list(sessions)
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
        result : List
            The most recent sessions. Sample of size
            ``self.possible_neighbors_sample_size``.
        """
        rank = [(sid, self.session_item_map[sid][1]) for sid in sessions]
        rank.sort(key=lambda x: x[1], reverse=True)
        result = [x[0] for x in rank]
        sample_size = min(self.possible_neighbors_sample_size, len(sessions))
        return result[:sample_size]

    def _sampling_weighted_events(self, sessions: Set) -> List:
        """Get sessions with the highest weights.

        Parameters
        ----------
        sessions : set
            Unique sessions.

        Returns
        -------
        result : List
            Sessions with the highest weights. Sample of size
            ``self.possible_neighbors_sample_size``.
        """

        rank = [(
            ses,
            np.mean(
                self.session_item_map[ses][self.sampling_event_weights_index]
            )
        ) for ses in sessions]

        rank.sort(key=lambda x: x[1], reverse=True)
        result = [x[0] for x in rank]
        sample_size = min(self.possible_neighbors_sample_size, len(sessions))
        return result[:sample_size]
