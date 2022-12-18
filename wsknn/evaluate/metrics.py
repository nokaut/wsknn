from typing import Dict, List, Tuple

import numpy as np
from wsknn.model.wsknn import WSKNN
from wsknn.evaluate.scores.scores import mrr_func, precision_func, recall_func
from wsknn.utils.errors import TooShortSessionException


def score_model(sessions: List,
                trained_model: WSKNN,
                k=0,
                skip_short_sessions=True,
                calc_mrr: bool = True,
                calc_precision: bool = True,
                calc_recall: bool = True,
                sliding_window: bool = False) -> Dict:
    """
    Function get Precision@k, Recall@k and MRR@k.

    Parameters
    ----------
    sessions : List of sessions
        >>> [
        ...     [
        ...         [ sequence_of_items ],
        ...         [ sequence_of_timestamps ],
        ...         [ [OPTIONAL] sequence_of_event_type ]
        ...     ],
        ... ]

    trained_model : WSKNN
        The trained WSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate **MRR**. Default is 0 and
        when it is set, then ``k`` is equal to the number of recommendations from a trained model. If ``k``
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
        Should the algorithm skip short sessions when calculating MRR or should it raise an error?

    calc_mrr : bool, default = True
        Should **MRR** be calculated?

    calc_precision : bool, default = True
        Should **precision** be calculated?

    calc_recall : bool, default = True
        Should **recall** be calculated?

    sliding_window : bool, default = False
        When calculating metrics slide through a single session up to the point when it is not possible to have the
        same number of evaluation products as the number of recommendations.

    Returns
    -------
    scores : Dict
        ``{'MRR': float, 'Recall': float, 'Precision': float}``
    """
    mrrs = list()
    precisions = list()
    recalls = list()

    k, trained_model = _set_number_of_recommendations(k, trained_model)

    for session in sessions:

        s_length = len(session[0])
        session_length_test = _should_skip_short_session(s_length, k, skip_short_sessions)

        if session_length_test:
            # Session is too short to make any valuable scoring
            pass
        else:
            eval_items, predictions = _prepare_metrics_data(session, trained_model, sliding_window)

            for i in range(len(eval_items)):
                recommendations = predictions[i]
                recommendations = [x[0] for x in recommendations]  # We are not interested in the weights
                evaluation_items = eval_items[i]

                # Get rank
                if calc_mrr:
                    partial_rank = mrr_func(recommendations, evaluation_items)
                    mrrs.append(partial_rank)

                # Get precisions
                if calc_precision:
                    partial_precision = precision_func(recommendations, evaluation_items)
                    precisions.append(partial_precision)

                # Get recalls
                if calc_recall:
                    partial_recall = recall_func(recommendations, evaluation_items)
                    recalls.append(partial_recall)

    mrr = float(np.mean(mrrs))
    prec = float(np.mean(precisions))
    rec = float(np.mean(recalls))

    scores = {
        'MRR': mrr,
        'Precision': prec,
        'Recall': rec
    }

    return scores


def get_mean_reciprocal_rank(sessions: List,
                             trained_model: WSKNN,
                             k=0,
                             skip_short_sessions=True,
                             sliding_window=False) -> float:
    """
    The function calculates the mean reciprocal rank of a top ``k`` recommendations. Given session must be longer
    than ``k`` events.

    Parameters
    ----------
    sessions : List
        >>> [
        ...     [
        ...         [ sequence_of_items ],
        ...         [ sequence_of_timestamps ],
        ...         [ [OPTIONAL] sequence_of_event_type ]
        ...     ],
        ... ]

    trained_model : WSKNN
        The trained WSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate **MRR**. Default is 0 and
        when it is set, then ``k`` is equal to the number of recommendations from a trained model. If ``k``
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
        Should the algorithm skip short sessions when calculating **MRR** or should it raise an error?

    sliding_window : bool, default = False
        When calculating metrics slide through a single session up to the point when it is not possible to have the
        same number of evaluation products as the number of recommendations.

    Returns
    -------
    mrr : float
        Mean Reciprocal Rank: The average score of **MRR** per ``n`` sessions.
    """
    mrrs = list()

    k, trained_model = _set_number_of_recommendations(k, trained_model)

    for session in sessions:

        s_length = len(session[0])
        session_length_test = _should_skip_short_session(s_length, k, skip_short_sessions)
        if session_length_test:
            # Session is too short to make any valuable scoring
            pass
        else:
            eval_items, predictions = _prepare_metrics_data(session, trained_model, sliding_window)

            # Get rank
            for i in range(len(eval_items)):
                recommendations = predictions[i]
                recommendations = [x[0] for x in recommendations]  # We are not interested in the weights
                evaluation_items = eval_items[i]
                partial_rank = mrr_func(recommendations, evaluation_items)
                mrrs.append(partial_rank)

    mrr = np.mean(mrrs)
    return float(mrr)


def get_precision(sessions: List,
                  trained_model: WSKNN,
                  k=0,
                  skip_short_sessions=True,
                  sliding_window=False) -> float:
    """
    The function calculates the precision score of a top ``k`` recommendations. Given session must be longer than
    ``k`` events.

    Parameters
    ----------
    sessions : List
        >>> [
        ...     [
        ...         [ sequence_of_items ],
        ...         [ sequence_of_timestamps ],
        ...         [ [OPTIONAL] sequence_of_event_type ]
        ...     ],
        ... ]

    trained_model : WSKNN
        The trained WSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate **precision**. Default is 0 and
        when it is set, then ``k`` is equal to the number of recommendations from a trained model. If ``k``
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
        Should the algorithm skip short sessions when calculating **precision** or should it raise an error?

    sliding_window : bool, default = False
        When calculating metrics slide through a single session up to the point when it is not possible to have the
        same number of evaluation products as the number of recommendations.

    Returns
    -------
    precision : float
        Precision: The average score of **precision** per ``n`` sessions.

    Notes
    -----
    Precision is defined as ``(no of recommendations that are relevant) / (number of items recommended)``.
    """

    precisions = list()

    k, trained_model = _set_number_of_recommendations(k, trained_model)

    for session in sessions:

        s_length = len(session[0])

        session_length_test = _should_skip_short_session(s_length, k, skip_short_sessions)

        if session_length_test:
            # Session is too short to make any valuable scoring
            pass
        else:
            eval_items, predictions = _prepare_metrics_data(session, trained_model, sliding_window)

            # Get rank
            for i in range(len(eval_items)):
                recommendations = predictions[i]
                recommendations = [x[0] for x in recommendations]  # We are not interested in the weights
                evaluation_items = eval_items[i]
                partial_precision = precision_func(recommendations, evaluation_items)
                precisions.append(partial_precision)

    precision = np.mean(precisions)
    return float(precision)


def get_recall(sessions: List,
               trained_model: WSKNN,
               k=0,
               skip_short_sessions=True,
               sliding_window=False) -> float:
    """
    The function calculates the **recall** score of a top ``k`` recommendations. Given session must be longer than
    ``k`` events.

    Parameters
    ----------
    sessions : List
        >>> [
        ...     [
        ...         [ sequence_of_items ],
        ...         [ sequence_of_timestamps ],
        ...         [ [OPTIONAL] sequence_of_event_type ]
        ...     ],
        ... ]

    trained_model : WSKNN
        The trained WSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate **recall**. Default is 0 and
        when it is set, then ``k`` is equal to the number of recommendations from a trained model. If ``k``
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
        Should the algorithm skip short sessions when calculating **recall** or should it raise an error?

    sliding_window : bool, default = False
        When calculating metrics slide through a single session up to the point when it is not possible to have the
        same number of evaluation products as the number of recommendations.

    Returns
    -------
    recall : float
        The average score of Recall per ``n`` sessions.

    Notes
    -----
    Recall is defined as (no of recommendations that are relevant) / (all relevant items for a user).
    """
    recalls = list()

    k, trained_model = _set_number_of_recommendations(k, trained_model)

    for session in sessions:

        s_length = len(session[0])

        session_length_test = _should_skip_short_session(s_length, k, skip_short_sessions)

        if session_length_test:
            # Session is too short to make any valuable scoring
            pass
        else:
            eval_items, predictions = _prepare_metrics_data(session, trained_model, sliding_window)

            # Get rank
            for i in range(len(eval_items)):
                recommendations = predictions[i]
                recommendations = [x[0] for x in recommendations]  # We are not interested in the weights
                evaluation_items = eval_items[i]
                partial_recall = recall_func(recommendations, evaluation_items)

                recalls.append(partial_recall)

    recall = np.mean(recalls)
    return float(recall)


def _prepare_metrics_data(session, trained_model, sliding_window):
    """
    Function prepares metrics data.

    Parameters
    ----------
    session : Any
              Array or list with a session.

    trained_model : WSKNN
                    Model to make predictions.

    sliding_window : bool, default = False
                     When calculating metrics slide through a single session up to the point when it is not possible
                     to have the same number of evaluation products as the number of recommendations.

    Returns
    -------
    : Tuple[List, List]
        (relevant items, recommended items)
    """
    relevant_items, recommends = _get_test_eval_preds(session, trained_model, sliding_window)
    return relevant_items, recommends


def _get_test_eval_preds(session, trained_model: WSKNN, sliding_window: bool):
    """
    Function parses session into test session, evaluation items (relevant items), and recommendations.

    Parameters
    ----------
    session : Any
              Array or list with a session.

    trained_model : WSKNN
                    Model to make predictions.

    sliding_window : bool, default = False
                     When calculating metrics slide through a single session up to the point when it is not possible
                     to have the same number of evaluation products as the number of recommendations.

    Returns
    -------
    : Tuple[List, List]
        (relevant items, recommended items)
    """
    recommended_items_list = list()
    relevant_items_list = list()

    session_range = len(session[0])
    k = trained_model.n_of_recommendations

    if sliding_window:
        srange = range(session_range-k, session_range)
        for i in srange:
            test_session = [x[:i] for x in session]
            relevant_items = session[0][i:]
            recommended_items = trained_model.recommend(test_session)

            recommended_items_list.append(recommended_items)
            relevant_items_list.append(relevant_items)
    else:

        test_session = [x[:k] for x in session]
        recommended_items = trained_model.recommend(test_session)
        relevant_items = session[0][k:]

        recommended_items_list.append(recommended_items)
        relevant_items_list.append(relevant_items)

    return relevant_items_list, recommended_items_list


def _set_number_of_recommendations(k: int, wsknn_model: WSKNN) -> Tuple[int, WSKNN]:
    """
    Function checks if k parameter is different than no_of_neighbors and sets it to

    Parameters
    ----------
    k : int
        Number of recommendations.

    wsknn_model : WSKNN
                  Trained wsknn model.

    Returns
    -------
    : Tuple[int, WSKNN]
        Number of recommendations, Trained model with selected number of recommendations
    """

    # First, check k
    if k == 0:
        k = wsknn_model.n_of_recommendations

    if k != wsknn_model.n_of_recommendations:
        wsknn_model.n_of_recommendations = k

    return k, wsknn_model


def _should_skip_short_session(s_length: int, k: int, skip_short: bool) -> bool:
    """
    Function checks if session length is smaller than number of recommendations -> in this case it is not possible
    to clip session into a prediction and evaluation parts.

    Parameters
    ----------
    s_length : int
               The length of a session.

    k : int
        The number of neighbors.

    skip_short : bool
                 If True then nothing happens even if session is short. Else error is raised.

    Returns
    -------
    : bool
        True, if session is too short and should be skipped.

    Raises
    ------
    TooShortSessionException : Raised if session length is < k and if skip_short parameter is set to False.
    """
    if s_length <= k:
        if skip_short:
            return True
        else:
            raise TooShortSessionException(s_length, k)
    else:
        return False
