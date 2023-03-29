import gzip
import json


def to_json(datadict: dict, filename: str, compress=True) -> None:
    """
    Function saves mapped object into jsonl file.

    Parameters
    ----------
    datadict : Dict
        Data to be stored.

    filename : str
        The path to the stored object. If suffix ``.json`` or ``.jsonl`` is not given then methods appends
        ``.json`` suffix into the file.

    compress : bool, default=True
        Should file be compressed into a gzip archive?
    """

    sjson = '.json'
    sjsonl = '.jsonl'
    sgz = '.gz'

    if (not filename.endswith(sjson)) or (not filename.endswith(sjsonl)):
        filename = filename + sjson

    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for kdx, vals in datadict.items():
                d = {kdx: vals}
                jout = json.dumps(d) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w') as out:
            for kdx, vals in datadict.items():
                d = {kdx: vals}
                jout = json.dumps(d) + '\n'
                out.write(jout)
