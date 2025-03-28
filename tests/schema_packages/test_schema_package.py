import os.path

from nomad.client import normalize_all, parse

EXPECTED_VALUE = 3.4

def test_schema_package():
    test_file = os.path.join('tests', 'data', 'test.archive.yaml')
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.value.m == EXPECTED_VALUE
