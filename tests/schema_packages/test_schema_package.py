import glob
import os.path

import numpy as np
import pytest
from nomad.client import normalize_all, parse

from nomad_nmr_schema.schema_packages.tensor_utils import (
    NMRTensor,
    TensorConvention,
)
from tests.schema_packages.expected_values import (
    EXPECTED_GRADIENT_DERIVED,
    EXPECTED_GRADIENT_VALUE,
    EXPECTED_ISC_DERIVED,
    EXPECTED_ISC_VALUE,
    EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE,
    EXPECTED_SHIELDING_DERIVED,
    EXPECTED_SHIELDING_VALUE,
)

test_files = glob.glob(os.path.join('tests', 'data', '*.archive.yaml'))


def check_magnetic_susceptibility(data):
    assert np.array_equal(data.value.m, EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE), (
        f'Magnetic Susceptibility tensor mismatch: '
        f'expected {EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE}, '
        f'got {data.value.m}'
    )


def check_magnetic_shielding(data):
    assert np.array_equal(data.value.m, EXPECTED_SHIELDING_VALUE), (
        f'Magnetic Shielding tensor mismatch: '
        f'expected {EXPECTED_SHIELDING_VALUE}, got {data.value.m}'
    )
    matrix = np.array(data.value.m, dtype=float)
    tensor = NMRTensor(matrix, TensorConvention.Haeberlen)
    data.isotropy = tensor.isotropy
    data.anisotropy = tensor.anisotropy
    data.reduced_anisotropy = tensor.reduced_anisotropy
    data.asymmetry = tensor.asymmetry
    data.span = tensor.span
    data.skew = tensor.skew

    assert data.isotropy.m == pytest.approx(
        EXPECTED_SHIELDING_DERIVED['isotropy'], abs=1e-2
    )
    assert data.anisotropy.m == pytest.approx(
        EXPECTED_SHIELDING_DERIVED['anisotropy'], abs=1e-2
    )
    assert data.reduced_anisotropy.m == pytest.approx(
        EXPECTED_SHIELDING_DERIVED['reduced_anisotropy'], abs=1e-2
    )
    assert data.asymmetry.m == pytest.approx(
        EXPECTED_SHIELDING_DERIVED['asymmetry'], abs=1e-2
    )
    assert 0 <= data.asymmetry.m <= 1, (
        f'Asymmetry value {data.asymmetry.m} is not between 0 and 1'
    )
    assert data.span.m == pytest.approx(EXPECTED_SHIELDING_DERIVED['span'], abs=1e-2)
    assert data.skew.m == pytest.approx(EXPECTED_SHIELDING_DERIVED['skew'], abs=1e-2)


def check_electric_field_gradient(data):
    assert np.array_equal(data.value.m, EXPECTED_GRADIENT_VALUE), (
        f'Electric Field Gradient tensor mismatch: '
        f'expected {EXPECTED_GRADIENT_VALUE}, got {data.value.m}'
    )
    matrix = np.array(data.value.m, dtype=float)
    tensor = NMRTensor(matrix, TensorConvention.NQR)
    data.Vzz = tensor.eigenvalues[2]
    data.asymmetry = tensor.asymmetry
    assert data.Vzz.m == pytest.approx(EXPECTED_GRADIENT_DERIVED['Vzz'], abs=1e-2)
    assert data.asymmetry.m == pytest.approx(
        EXPECTED_GRADIENT_DERIVED['asymmetry'], abs=1e-2
    )
    assert 0 <= data.asymmetry.m <= 1, (
        f'Asymmetry value {data.asymmetry.m} is not between 0 and 1'
    )


def check_indirect_spin_spin_coupling(data):
    assert np.array_equal(data.value.m, EXPECTED_ISC_VALUE), (
        f'Electric Field Gradient tensor mismatch: '
        f'expected {EXPECTED_ISC_VALUE}, got {data.value.m}'
    )
    component_files = glob.glob(
        os.path.join('tests', 'data', 'test_indirect_spin_spin_coupling_*.archive.yaml')
    )
    summed_matrix = None
    for comp_file in component_files:
        comp_archive = parse(comp_file)[0]
        normalize_all(comp_archive)
        matrix = np.array(comp_archive.data.value.m)
        if summed_matrix is None:
            summed_matrix = matrix
        else:
            summed_matrix += matrix
    assert np.allclose(summed_matrix, EXPECTED_ISC_VALUE, rtol=1e-3), (
        f'Indirect Spin-Spin Coupling tensor mismatch: '
        f'expected {EXPECTED_ISC_VALUE}, got {summed_matrix}'
    )
    matrix = np.array(data.value.m, dtype=float)
    tensor = NMRTensor(matrix, TensorConvention.Haeberlen)
    data.isotropy = tensor.isotropy
    data.anisotropy = tensor.anisotropy
    data.asymmetry = tensor.asymmetry
    assert data.isotropy.m == pytest.approx(EXPECTED_ISC_DERIVED['isotropy'], abs=1e-2)
    assert data.anisotropy.m == pytest.approx(
        EXPECTED_ISC_DERIVED['anisotropy'], abs=1e-2
    )
    assert data.asymmetry.m == pytest.approx(
        EXPECTED_ISC_DERIVED['asymmetry'], abs=1e-2
    )
    assert 0 <= data.asymmetry.m <= 1, (
        f'Asymmetry value {data.asymmetry.m} is not between 0 and 1'
    )


def check_atoms_state(data):
    pass  # Placeholder for future checks on atoms state


@pytest.mark.parametrize('test_file', test_files)
def test_schema_package(test_file):
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)
    name = entry_archive.data.m_def.name

    # Perform tests on the normalized entry_archive
    # Test Magnetic Susceptibility
    if name == 'MagneticSusceptibility':
        check_magnetic_susceptibility(entry_archive.data)
    # Test Magnetic Shielding
    elif name == 'MagneticShielding':
        print(entry_archive.data)
        check_magnetic_shielding(entry_archive.data)
    # Test Electric Field Gradient
    elif name == 'ElectricFieldGradient':
        check_electric_field_gradient(entry_archive.data)
    # Test Indirect Spin-Spin Coupling
    elif name == 'IndirectSpinSpinCoupling':
        check_indirect_spin_spin_coupling(entry_archive.data)
