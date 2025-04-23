import glob
import os.path

import numpy as np
import pytest
from nomad.client import normalize_all, parse

test_files = glob.glob(os.path.join('tests', 'data', '*.archive.yaml'))
EXPECTED_VALUE = [
    [-16.0397, 0.0000, 0.0000],
    [0.0000, -16.1160, 0.0000],
    [0.0000, 0.0000, -48.9908],
]


@pytest.mark.parametrize('test_file', test_files)
def test_schema_package(test_file):
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    # assert entry_archive.data.value.m == EXPECTED_VALUE
    assert np.array_equal(entry_archive.data.value.m, EXPECTED_VALUE)
