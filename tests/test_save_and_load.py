import os
import tempfile

from wsknn import WSKNN
from wsknn.import_export.ie import save, load


def test_save_and_load():
    model_name = 'wsknn_model.joblib'
    # Type of sessions is str, type of items is int
    sessions = {
        'a': [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ],
        'b': [
            [2, 3, 4, 5],
            [10, 11, 12, 13]
        ]
    }

    items = {
        1: [['a'], [1]],
        2: [['a', 'b'], [2]],
        3: [['a', 'b'], [3]],
        4: [['a', 'b'], [4]],
        5: [['a', 'b'], [5]]
    }

    model = WSKNN(return_events_from_session=False)
    model.fit(sessions, items)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, model_name)

        # Save
        mpath = save(model, tmppath)

        assert mpath[0] == tmppath

        # Load
        model = load(tmppath)
        assert model.item_session_map is not None
        assert model.session_item_map is not None
