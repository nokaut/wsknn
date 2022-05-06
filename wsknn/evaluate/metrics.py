from typing import Dict

import numpy as np
from wsknn.model.wsknn import WSKNN
from wsknn.utils.errors import TooShortSessionException


def score_model(sessions: dict,
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
    sessions : dict
               {'session_id': [
                   [ sequence_of_items ],
                   [ sequence_of_timestamps ],
                   [ [OPTIONAL] sequence_of_event_type ]
               ]}

    trained_model : WSKNN
                    Trained VSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate MRR. Default is 0 and
        when it is set, then k is equal to the number of recommendations from a trained model. If k
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
                          Should the algorithm skip short sessions when calculating MRR or should raise an error?

    calc_mrr : bool, default = True
               Should MRR be calculated?

    calc_precision : bool, default = True
                     Should Precision be calculated?

    calc_recall : bool, default = True
                  Should Recall be calculated?

    sliding_window : bool, default = False
                     When calculating metrics slide through a single session up to the point when it is not possible
                     to have the same number of evaluation products as the number of recommendations.

    Returns
    -------
    : Dict
    {'MRR': float, 'Recall': float, 'Precision': float}
    """
    mrrs = list()
    precisions = list()
    recalls = list()

    trained_model = _set_number_of_recommendations(k, trained_model)

    for session_k, session in sessions.items():

        s_length = len(session[0])

        _should_raise_short_session_exception(s_length, k, skip_short_sessions)

        eval_items, predictions = list(), list()

        if skip_short_sessions:
            if s_length < k + 1:
                pass
            else:
                eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)
        else:
            eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)

        for i in range(len(eval_items)):

            preds = predictions[i][session_k]
            eitmes = eval_items[i]

            # Get rank
            if calc_mrr:
                partial_rank = __mrr(preds, eitmes)
                mrrs.append(partial_rank)

            # Get precisions
            if calc_precision:
                partial_precision = __precision(preds, eitmes, k)
                precisions.append(partial_precision)

            # Get recalls
            if calc_recall:
                partial_recall = __recall(preds, eitmes)
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


def get_mean_reciprocal_rank(sessions: dict, trained_model: WSKNN, k=0, skip_short_sessions=True,
                             sliding_window=False) -> float:
    """
    The function calculates the mean reciprocal rank of a top k recommendations.
    Given session must be longer than k events.

    Parameters
    ----------
    sessions : dict
               {'session_id': [
                   [ sequence_of_items ],
                   [ sequence_of_timestamps ],
                   [ [OPTIONAL] sequence_of_event_type ]
               ]}

    trained_model : WSKNN
                    Trained VSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate MRR. Default is 0 and
        when it is set, then k is equal to the number of recommendations from a trained model. If k
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
                          Should the algorithm skip short sessions when calculating MRR or should raise an error?

    sliding_window : bool, default = False
                     When calculating metrics slide through a single session up to the point when it is not possible
                     to have the same number of evaluation products as the number of recommendations.

    Returns
    -------
    : float
        Mean Reciprocal Rank: The average score of MRR per n sessions.
    """
    mrrs = list()

    trained_model = _set_number_of_recommendations(k, trained_model)

    for session_k, session in sessions.items():

        s_length = len(session[0])

        _should_raise_short_session_exception(s_length, k, skip_short_sessions)

        eval_items, predictions = list(), list()

        if skip_short_sessions:
            if s_length < k + 1:
                pass
            else:
                eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)
        else:
            eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)

        # Get rank
        for i in range(len(eval_items)):
            partial_rank = __mrr(predictions[i][session_k], eval_items[i])

            mrrs.append(partial_rank)

    mrr = np.mean(mrrs)
    return float(mrr)


def get_precision(sessions: dict, trained_model: WSKNN, k=0, skip_short_sessions=True, sliding_window=False) -> float:
    """
    The function calculates the precision score of a top k recommendations.
    Given session must be longer than k events.

    Parameters
    ----------
    sessions : dict
               {'session_id': [
                   [ sequence_of_items ],
                   [ sequence_of_timestamps ],
                   [ [OPTIONAL] sequence_of_event_type ]
               ]}

    trained_model : WSKNN
                    Trained VSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate Precision. Default is 0 and
        when it is set, then k is equal to the number of recommendations from a trained model. If k
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
                          Should the algorithm skip short sessions when calculating Precision or should raise an error?

    sliding_window : bool, default = False
                     When calculating metrics slide through a single session up to the point when it is not possible
                     to have the same number of evaluation products as the number of recommendations.

    Returns
    -------
    : float
        Precision: The average score of Precision per n sessions.

    Notes
    -----
    Precision is defined as (no of recommendations that are relevant) / (number of items recommended).
    """

    precisions = list()

    trained_model = _set_number_of_recommendations(k, trained_model)

    for session_k, session in sessions.items():

        s_length = len(session[0])

        _should_raise_short_session_exception(s_length, k, skip_short_sessions)

        eval_items, predictions = list(), list()

        if skip_short_sessions:
            if s_length < k + 1:
                pass
            else:
                eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)
        else:
            eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)

        for i in range(len(eval_items)):
            # Get precision
            partial_precision = __precision(predictions[i][session_k], eval_items[i], k)
            precisions.append(partial_precision)

    precision = np.mean(precisions)
    return float(precision)


def get_recall(sessions: dict, trained_model: WSKNN, k=0, skip_short_sessions=True, sliding_window=False) -> float:
    """
    The function calculates the recall score of a top k recommendations.
    Given session must be longer than k events.

    Parameters
    ----------
    sessions : dict
               {'session_id': [
                   [ sequence_of_items ],
                   [ sequence_of_timestamps ],
                   [ [OPTIONAL] sequence_of_event_type ]
               ]}

    trained_model : WSKNN
                    Trained VSKNN model.

    k : int, default=0
        Number of top recommendations. Session must have n+1 items minimum to calculate Recall. Default is 0 and
        when it is set, then k is equal to the number of recommendations from a trained model. If k
        is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
                          Should the algorithm skip short sessions when calculating Recall or should raise an error?

    sliding_window : bool, default = False
                     When calculating metrics slide through a single session up to the point when it is not possible
                     to have the same number of evaluation products as the number of recommendations.

    Returns
    -------
    : float
        Precision: The average score of Recall per n sessions.

    Notes
    -----
    Recall is defined as (no of recommendations that are relevant) / (all relevant items for a user).
    """
    recalls = list()

    trained_model = _set_number_of_recommendations(k, trained_model)

    for session_k, session in sessions.items():

        s_length = len(session[0])

        _should_raise_short_session_exception(s_length, k, skip_short_sessions)

        eval_items, predictions = list(), list()

        if skip_short_sessions:
            if s_length < k + 1:
                pass
            else:
                eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)
        else:
            eval_items, predictions = _prepare_metrics_data(session, session_k, trained_model, sliding_window)

        for i in range(len(eval_items)):
            # Get recall
            partial_recall = __recall(predictions[i][session_k], eval_items[i])

            recalls.append(partial_recall)

    recall = np.mean(recalls)
    return float(recall)


def __mrr(preds, rel_items):
    rank = 0
    for idx, prod in enumerate(preds):
        ii = idx + 1
        if prod[0] in rel_items:
            rank = 1 / ii
            return rank
    return rank


def __precision(preds, rel_items, k):
    rank = 0

    for recommendation in preds:
        ritem = recommendation[0]
        if ritem in rel_items:
            rank = rank + 1

    session_precision = rank / k
    return session_precision


def __recall(preds, rel_items):
    rank = 0

    for recommendation in preds:
        ritem = recommendation[0]
        if ritem in rel_items:
            rank = rank + 1

    session_recall = rank / len(rel_items)
    return session_recall


def _prepare_metrics_data(session, session_key, trained_model, sliding_window):
    """
    Function prepares metrics data.

    Parameters
    ----------
    session : Any
              Array or list with a session.

    session_key : str
                  Unique key of a session (for example could be a user ID)

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
    relevant_items, recommends = _get_test_eval_preds(session, session_key, trained_model, sliding_window)
    return relevant_items, recommends


def _get_test_eval_preds(session, session_key: str, trained_model: WSKNN, sliding_window: bool):
    """
    Function parses session into test session, evaluation items (relevant items), and recommendations.

    Parameters
    ----------
    session : Any
              Array or list with a session.

    session_key : str
                  Unique key of a session (for example could be a user ID)

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
    relevants_items_list = list()

    session_range = len(session[0])

    if sliding_window:
        for i in range(session_range):
            test_session = [x[:i] for x in session]
            relevant_items = session[0][i:]
            recommended_items = trained_model.predict(
                {session_key: test_session}
            )

            recommended_items_list.append(recommended_items)
            relevants_items_list.append(relevant_items)
    else:
        k = trained_model.n_of_recommendations
        test_session = [x[:-k] for x in session]
        recommended_items = trained_model.predict({session_key: test_session})
        relevant_items = session[0][k:]

        recommended_items_list.append(recommended_items)
        relevants_items_list.append(relevant_items)

    return relevants_items_list, recommended_items_list


def _set_number_of_recommendations(k: int, wsknn_model: WSKNN) -> WSKNN:
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

    return wsknn_model


def _should_raise_short_session_exception(s_length: int, k: int, skip_short: bool) -> None:
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

    Raises
    ------
    TooShortSessionException : Raised if session length is < k and if skip_short parameter is set to False.
    """
    if s_length <= k:
        if skip_short:
            pass
        else:
            raise TooShortSessionException(s_length, k)
