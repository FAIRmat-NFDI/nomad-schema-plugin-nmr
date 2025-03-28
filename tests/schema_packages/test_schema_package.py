import glob
import os.path

import pytest
from nomad.client import normalize_all, parse

test_files = glob.glob(os.path.join('tests', 'data', '*.archive.yaml'))
EXPECTED_VALUE = 3.4


@pytest.mark.parametrize('test_file', test_files)
def test_schema_package(test_file):
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.value.m == EXPECTED_VALUE
