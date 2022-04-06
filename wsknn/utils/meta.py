import yaml


def parse_settings(settings_file: str) -> dict:
    """
    The function parses settings file into dict

    Parameters
    ----------
    settings_file : str
                    File with the model settings, must be in yaml.

    Returns
    -------
    ydict : dict
            Parsed settings used for modeling.
    """

    with open(settings_file, 'r') as fstream:
        ydict = yaml.safe_load(fstream)

    return ydict
