import os
import random

from wsknn.fit_transform import fit
from wsknn.predict import predict
from wsknn.model.wsknn import WSKNN
from wsknn.utils.meta import parse_settings
from wsknn.utils.transform import load_pickled


DEFAULT_SETTINGS = os.path.join(os.getcwd(), '../settings.yml')


def build_model(settings: dict) -> WSKNN:
    """Function builds model from the input data.

    Parameters
    ----------
    settings : dict
               dict with settings file

    Returns
    -------
    fitted_model : WSKNN
    """

    # Load data
    raw_sessions = load_pickled(settings['input_sessions'])
    raw_items = load_pickled(settings['input_items'])

    # Fit data
    fitted_model = fit(raw_sessions['map'], raw_items['map'])
    return fitted_model


def _test_model(model: WSKNN, settings: dict):
    """Function tests model locally by random sampling of sessions

    Parameters
    ----------
    model : WSKNN
            fitted model

    settings : dict
               parsed settings
    """
    kl = list(model.session_item_map.keys())
    sample = random.randint(0, len(kl))
    test_session = model.session_item_map[kl[sample]]
    prediction = predict(model, test_session, settings_dict=settings['model'])
    print(prediction)


if __name__ == '__main__':
    _parsed_settings = parse_settings(DEFAULT_SETTINGS)
    _model = build_model(_parsed_settings)
    _test_model(_model, _parsed_settings)
