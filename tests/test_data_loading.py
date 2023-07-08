from wsknn.utils.transform import load_jsonl, load_gzipped_jsonl


EXPECTED_DICT = dict(id123=10.0002, id234=-90, id345="zzz")
JSONL_FILE = 'tdata/test.jsonl'
GZ_FILE = 'tdata/test.jsonl.gz'


def test_load_jsonl():
    loaded_data = load_jsonl(JSONL_FILE)
    assert loaded_data == EXPECTED_DICT


def test_load_gzip():
    loaded_data = load_gzipped_jsonl(GZ_FILE)
    assert loaded_data == EXPECTED_DICT

