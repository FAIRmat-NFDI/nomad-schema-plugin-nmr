import glob
import os.path

import numpy as np
import pytest
from nomad.client import normalize_all, parse
from nomad_simulations.schema_packages.model_system import (
    AtomsState,
    ModelSystem,
)

from nomad_nmr_schema.schema_packages.schema_package import (
    IndirectSpinSpinCoupling,
    MagneticShielding,
    resolve_name_from_entity_ref,
)
from nomad_nmr_schema.schema_packages.tensor_utils import (
    NMRTensor,
    TensorConvention,
)
from tests.schema_packages.expected_values import (
    EXPECTED_DELTA_G_PARATEC_VALUE,
    EXPECTED_DELTA_G_VALUE,
    EXPECTED_GRADIENT_DERIVED,
    EXPECTED_GRADIENT_VALUE,
    EXPECTED_HYPERFINE_DIPOLAR_VALUE,
    EXPECTED_HYPERFINE_FERMI_CONTACT_VALUE,
    EXPECTED_ISC_DERIVED,
    EXPECTED_ISC_VALUE,
    EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE,
    EXPECTED_SHIELDING_DERIVED,
    EXPECTED_SHIELDING_VALUE,
    EXPECTED_UNPAIRED_SPINS_VALUE,
)
from tests.schema_packages.sample_magres_data import (
    ISC_DATA,
    MS_DATA,
)

test_files = glob.glob(os.path.join('tests', 'data', '*.archive.yaml'))
magnetic_shielding = MagneticShielding
indirect_spin_spin_coupling = IndirectSpinSpinCoupling
atoms_state = AtomsState
model_system = ModelSystem


def check_magnetic_susceptibility(data):
    # Assert that the parsed magnetic susceptibility tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE), (
        f'Magnetic Susceptibility tensor mismatch: '
        f'expected {EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE}, '
        f'got {data.value.m}'
    )


def check_magnetic_shielding(data):
    # Assert that the parsed magnetic shielding tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_SHIELDING_VALUE), (
        f'Magnetic Shielding tensor mismatch: '
        f'expected {EXPECTED_SHIELDING_VALUE}, got {data.value.m}'
    )

    # Convert the tensor to a numpy array and create an NMRTensor for derived property
    # calculations
    matrix = np.array(data.value.m, dtype=float)
    tensor = NMRTensor(matrix, TensorConvention.Haeberlen)

    # Assign derived tensor properties to the data object for further checks
    data.isotropy = tensor.isotropy
    data.anisotropy = tensor.anisotropy
    data.reduced_anisotropy = tensor.reduced_anisotropy
    data.asymmetry = tensor.asymmetry
    data.span = tensor.span
    data.skew = tensor.skew

    # Assert that all derived properties match the expected values within a tolerance
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
    assert data.span.m == pytest.approx(EXPECTED_SHIELDING_DERIVED['span'], abs=1e-2)
    assert data.skew.m == pytest.approx(EXPECTED_SHIELDING_DERIVED['skew'], abs=1e-2)

    # Ensure asymmetry is within the physically meaningful range [0, 1]
    assert 0 <= data.asymmetry.m <= 1, (
        f'Asymmetry value {data.asymmetry.m} is not between 0 and 1'
    )


def check_electric_field_gradient(data):
    # Assert that the parsed magnetic shielding tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_GRADIENT_VALUE), (
        f'Electric Field Gradient tensor mismatch: '
        f'expected {EXPECTED_GRADIENT_VALUE}, got {data.value.m}'
    )

    # Convert the tensor to a numpy array and create an NMRTensor for derived property
    # calculations
    matrix = np.array(data.value.m, dtype=float)
    tensor = NMRTensor(matrix, TensorConvention.NQR)

    # Assign derived tensor properties to the data object for further checks
    data.Vzz = tensor.eigenvalues[2]
    data.asymmetry = tensor.asymmetry

    # Assert that all derived properties match the expected values within a tolerance
    assert data.Vzz.m == pytest.approx(EXPECTED_GRADIENT_DERIVED['Vzz'], abs=1e-2)
    assert data.asymmetry.m == pytest.approx(
        EXPECTED_GRADIENT_DERIVED['asymmetry'], abs=1e-2
    )

    # Ensure asymmetry is within the physically meaningful range [0, 1]
    assert 0 <= data.asymmetry.m <= 1, (
        f'Asymmetry value {data.asymmetry.m} is not between 0 and 1'
    )


def check_indirect_spin_spin_coupling(data):
    # Assert that the parsed magnetic shielding tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_ISC_VALUE), (
        f'Electric Field Gradient tensor mismatch: '
        f'expected {EXPECTED_ISC_VALUE}, got {data.value.m}'
    )

    # Check if the component files exist and sum the tensors for decomposed indirect
    # spin-spin coupling contributions
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

    # Assert that the summed tensor matches the expected indirect spin-spin coupling
    # value total contribution
    assert np.allclose(summed_matrix, EXPECTED_ISC_VALUE, rtol=1e-3), (
        f'Indirect Spin-Spin Coupling tensor mismatch: '
        f'expected {EXPECTED_ISC_VALUE}, got {summed_matrix}'
    )

    # Convert the tensor to a numpy array and create an NMRTensor for derived property
    # calculations
    matrix = np.array(data.value.m, dtype=float)
    tensor = NMRTensor(matrix, TensorConvention.Haeberlen)

    # Assign derived tensor properties to the data object for further checks
    data.isotropy = tensor.isotropy
    data.anisotropy = tensor.anisotropy
    data.reduced_anisotropy = tensor.reduced_anisotropy
    data.asymmetry = tensor.asymmetry
    data.span = tensor.span

    # Assert that all derived properties match the expected values within a tolerance
    assert data.isotropy.m == pytest.approx(EXPECTED_ISC_DERIVED['isotropy'], abs=1e-2)
    assert data.anisotropy.m == pytest.approx(
        EXPECTED_ISC_DERIVED['anisotropy'], abs=1e-2
    )
    assert data.reduced_anisotropy.m == pytest.approx(
        EXPECTED_ISC_DERIVED['reduced_anisotropy'], abs=1e-2
    )
    assert data.asymmetry.m == pytest.approx(
        EXPECTED_ISC_DERIVED['asymmetry'], abs=1e-2
    )
    assert data.span.m == pytest.approx(EXPECTED_ISC_DERIVED['span'], abs=1e-2)

    # Ensure asymmetry is within the physically meaningful range [0, 1]
    assert 0 <= data.asymmetry.m <= 1, (
        f'Asymmetry value {data.asymmetry.m} is not between 0 and 1'
    )


def check_hyperfine_dipolar(data):
    # Assert that the parsed hyperfine dipolar tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_HYPERFINE_DIPOLAR_VALUE), (
        f'Hyperfine Dipolar tensor mismatch: '
        f'expected {EXPECTED_HYPERFINE_DIPOLAR_VALUE}, '
        f'got {data.value.m}'
    )


def check_hyperfine_fermi_contact(data):
    # Assert that the parsed hyperfine fermi contact scalar matches the expected value
    assert np.equal(data.value.m, EXPECTED_HYPERFINE_FERMI_CONTACT_VALUE), (
        f'Hyperfine Fermi Contact scalar mismatch: '
        f'expected {EXPECTED_HYPERFINE_FERMI_CONTACT_VALUE}, '
        f'got {data.value.m}'
    )


def check_delta_g(data):
    # Assert that the parsed delta g tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_DELTA_G_VALUE), (
        f'Delta G tensor mismatch: '
        f'expected {EXPECTED_DELTA_G_VALUE}, '
        f'got {data.value.m}'
    )


def check_delta_g_paratec(data):
    # Assert that the parsed delta g a la Paratec tensor matches the expected value
    assert np.array_equal(data.value.m, EXPECTED_DELTA_G_PARATEC_VALUE), (
        f'Delta G a la Paratec tensor mismatch: '
        f'expected {EXPECTED_DELTA_G_PARATEC_VALUE}, '
        f'got {data.value.m}'
    )


def check_unpaired_spins(data):
    # Assert that the parsed number of unpaired spins matches the expected value
    assert np.equal(data.value.m, EXPECTED_UNPAIRED_SPINS_VALUE), (
        f'Number of Unpaired Spins mismatch: '
        f'expected {EXPECTED_UNPAIRED_SPINS_VALUE}, '
        f'got {data.value.m}'
    )


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
        check_magnetic_shielding(entry_archive.data)
    # Test Electric Field Gradient
    elif name == 'ElectricFieldGradient':
        check_electric_field_gradient(entry_archive.data)
    # Test Indirect Spin-Spin Coupling
    elif name == 'IndirectSpinSpinCoupling':
        check_indirect_spin_spin_coupling(entry_archive.data)
    # Test Hyperfine Dipolar
    elif name == 'HyperfineDipolar':
        check_hyperfine_dipolar(entry_archive.data)
    # Test Hyperfine Fermi Contact
    elif name == 'HyperfineFermiContact':
        check_hyperfine_fermi_contact(entry_archive.data)
    # Test Delta G
    elif name == 'DeltaG':
        check_delta_g(entry_archive.data)
    # Test Delta G a la Paratec
    elif name == 'DeltaGParatec':
        check_delta_g_paratec(entry_archive.data)
    # Test number of Unpaired Spins
    elif name == 'UnpairedSpins':
        check_unpaired_spins(entry_archive.data)


def test_atoms_state_name_resolution_mag_shielding():
    # Dummy MS data for a single H atom
    data = MS_DATA

    # Extract atom data from the MS_data
    _, atom_data = list(enumerate(data))[0]

    # Create AtomsState object to represent the atom
    atoms_state_data = atoms_state(label=f"{atom_data[0]}_{atom_data[1]}")

    # Create a dummy AtomicCell and assign the AtomsState object to it
    model_system.particle_states = [atoms_state_data]
    atoms_state_data.m_parent = model_system

    # Create MagneticShielding instance, referencing the atom
    magnetic_shieldings = magnetic_shielding(entity_ref=model_system.particle_states[0])

    # Use the resolve_name_from_entity_ref function to get the resolved name
    test_name = resolve_name_from_entity_ref(
        [magnetic_shieldings.entity_ref], logger=None
    )

    # Check if the resolved name matches the expected name
    assert test_name == 'H4_4', f'Expected name "H4_4", got "{test_name}"'

def test_atoms_state_name_resolution_isc():
    # Dummy ISC data for a single coupling between C and H
    data = ISC_DATA

    # Extract the atom data from the dummy ISC_DATA
    _, atom_data = list(enumerate(data))[0]

    # Create AtomsState objects for both atoms involved in the coupling
    atoms_state_1 = atoms_state(label=f"{atom_data[0]}_{atom_data[1]}")
    atoms_state_2 = atoms_state(label=f"{atom_data[2]}_{atom_data[3]}")

    # Create a dummy AtomicCell and assign both AtomsState objects to it
    model_system.particle_states = [atoms_state_1, atoms_state_2]
    atoms_state_1.m_parent = model_system
    atoms_state_2.m_parent = model_system

    # Create a dummy IndirectSpinSpinCoupling instance, referencing both atoms
    isc = indirect_spin_spin_coupling(
        entity_ref_1=model_system.particle_states[0],
        entity_ref_2=model_system.particle_states[1],
    )

    # Use the resolve_name_from_entity_ref function to get the resolved name
    test_name = resolve_name_from_entity_ref(
        entities=[isc.entity_ref_1, isc.entity_ref_2],
        logger=None,
    )

    # Check if the resolved name matches the expected name
    assert test_name == 'C2_2-H4_4', f'Expected name "C2_2-H4_4", got "{test_name}"'
