from wsknn import WSKNN, batch_predict
from wsknn.evaluate.simulate import generate_input


def test_batch():
    imap, smap = generate_input(
        number_of_sessions=10000,
        number_of_items=200,
        min_session_length=3,
        max_session_length=12
    )

    model = WSKNN()
    model.fit(smap, imap)
    _, sess = generate_input(
        number_of_sessions=100,
        number_of_items=20,
        min_session_length=3,
        max_session_length=12
    )
    sess_dicts = [{k: v} for k, v in sess.items()]
    predictions = batch_predict(model, sess_dicts)
    assert isinstance(predictions, list)
    assert len(predictions) == len(sess_dicts)
