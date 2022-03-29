from wsknn.model.wsknn import WSKNN
from wsknn.utils.errors import TooShortSessionException


def mean_reciprocal_rank(sessions: dict, trained_model: WSKNN, r_number=0, skip_short_sessions=True) -> float:
    """
    Function calculates mean reciprocal rank of a top r_number recommendations.
    Given session must be longer than r_number.

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

    r_number : int, default=0
               Number of top recommendations. Session must have n+1 items minimum to calculate MRR. Default is 0 and
               when it is set, then r_number is equal to number of recommendations from a trained model. If r_number
               is greater than the number of recommendations then the latter is adjusted to it.

    skip_short_sessions : bool, default=True
                          Should algorithm skip short sessions when calculates MRR or should raise an error?

    Returns
    -------
    : float
        Mean Reciprocal Rank: The average score of MRR per n sessions.
    """
    mrr = 0
    denom = 0

    # Check r
    if r_number == 0:
        r_number = trained_model.n_of_recommendations

    if r_number > trained_model.n_of_recommendations:
        trained_model.n_of_recommendations = r_number

    for k, session in sessions.items():

        s_length = len(session[0])

        if s_length <= r_number:
            if skip_short_sessions:
                pass
            else:
                raise TooShortSessionException(s_length, r_number)
        else:
            test_session = [x[:-r_number] for x in session]
            eval_items = session[0][-r_number:]
            predictions = trained_model.predict(
                {k: test_session}
            )

            # Get rank
            rank = 0
            for idx, prod in enumerate(predictions[k]):
                if prod[0] in eval_items:
                    rank = 1 / idx
                    break

            mrr = mrr + rank
            denom = denom + 1

    mrr = mrr / denom
    return mrr
