import numpy as np
import pytest

from nomad_nmr_schema.schema_packages.tensor_utils import (NMRTensor,
                                                           TensorConvention,
                                                           _anisotropy,
                                                           _asymmetry,
                                                           _evals_sort, _skew,
                                                           _span)


def test_tensor_initialization():
    # Test initialization with 3x3 matrix
    matrix = np.array([[1.0, 0.5, 0.0],
                      [0.5, 2.0, 0.0],
                      [0.0, 0.0, 3.0]])
    tensor = NMRTensor(matrix)
    assert isinstance(tensor, NMRTensor)

    # Test initialization with eigenvalues and eigenvectors
    evals = np.array([1.0, 2.0, 3.0])
    evecs = np.eye(3)  # Identity matrix as eigenvectors
    tensor = NMRTensor((evals, evecs))
    assert isinstance(tensor, NMRTensor)

    # Test invalid initialization
    with pytest.raises(ValueError):
        NMRTensor(np.array([1, 2, 3]))  # Wrong shape

def test_eigenvalue_ordering():
    # Create a tensor with known eigenvalues
    matrix = np.array([[2.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 3.0]])

    # Test increasing order
    tensor_inc = NMRTensor(matrix, order=TensorConvention.Increasing)
    np.testing.assert_array_almost_equal(
        tensor_inc.eigenvalues,
        np.array([1.0, 2.0, 3.0])
    )

    # Test decreasing order
    tensor_dec = NMRTensor(matrix, order=TensorConvention.Decreasing)
    np.testing.assert_array_almost_equal(
        tensor_dec.eigenvalues,
        np.array([3.0, 2.0, 1.0])
    )

def test_haeberlen_convention():
    # Create a tensor that follows Haeberlen convention: |σzz - σiso| ≥ |σxx - σiso| ≥ |σyy - σiso|
    matrix = np.array([[10.0, 0.0, 0.0],
                      [0.0, 20.0, 0.0],
                      [0.0, 0.0, 30.0]])

    tensor = NMRTensor(matrix, order=TensorConvention.Haeberlen)
    evals = tensor.eigenvalues

    # Calculate isotropic value
    iso = np.mean(evals)

    # Verify Haeberlen ordering: |σzz - σiso| ≥ |σxx - σiso| ≥ |σyy - σiso|
    dev_zz = abs(evals[2] - iso)
    dev_xx = abs(evals[0] - iso)
    dev_yy = abs(evals[1] - iso)

    assert dev_zz >= dev_xx >= dev_yy

def test_nqr_convention():
    # Create a tensor that follows NQR convention: |Vzz| ≥ |Vyy| ≥ |Vxx|
    matrix = np.array([[-2.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, 3.0]])

    tensor = NMRTensor(matrix, order=TensorConvention.NQR)
    evals = tensor.eigenvalues

    # Verify NQR ordering: |Vzz| ≥ |Vyy| ≥ |Vxx|
    assert abs(evals[2]) >= abs(evals[1]) >= abs(evals[0])

def test_invalid_convention():
    matrix = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]])

    with pytest.raises(ValueError):
        NMRTensor(matrix, order="invalid")


def test_span():
    # Simple case - diagonal tensor with no off-diagonal elements
    evals = np.array([[1.0, 2.0, 3.0]])  # Shape (1, 3) as expected by function
    expected_span = 2.0  # 3.0 - 1.0
    assert _span(evals) == pytest.approx(expected_span)

    # Multiple tensors at once
    evals = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.0, 1.0],
    ])
    expected_spans = np.array([2.0, 2.0])  # [3.0-1.0, 1.0-(-1.0)]
    np.testing.assert_array_almost_equal(_span(evals), expected_spans)

    # Zero span case
    evals = np.array([[2.0, 2.0, 2.0]])
    assert _span(evals) == pytest.approx(0.0)

def test_skew():
    # Test skew calculation for a simple diagonal tensor
    evals = np.array([[1.0, 2.0, 3.0]])  # Shape (1, 3)
    # iso = (1 + 2 + 3)/3 = 2
    # median = 2
    # span = 3 - 1 = 2
    # skew = 3 * (2 - 2) / 2 = 0
    assert _skew(evals) == pytest.approx(0.0)

    # Test skew for multiple tensors
    evals = np.array([
        [1.0, 2.0, 3.0],  # skew = 0
        [0.0, 1.0, 4.0],  # iso = 5/3, median = 1, span = 4, skew = 3*(5/3 - 1)/4 = 0.5
    ])
    expected_skews = np.array([0.0, 0.5])
    np.testing.assert_array_almost_equal(_skew(evals), expected_skews)

    # Test zero span case - should return zero
    # since skew is undefined
    evals = np.array([[2.0, 2.0, 2.0]])
    assert _skew(evals) == pytest.approx(0.0)

def test_anisotropy():
    # Test regular anisotropy
    # First sort in Haeberlen convention
    evals = np.array([[1.0, 2.0, 4.0]])  # Shape (1, 3)
    sorted_evals = _evals_sort(evals, TensorConvention.Haeberlen)
    # aniso = σzz - (σxx + σyy)/2 = 4 - (1 + 2)/2 = 2.5
    assert _anisotropy(sorted_evals) == pytest.approx(2.5)

    # Test reduced anisotropy
    # reduced_aniso = aniso * 2/3 = 2.5 * 2/3
    assert _anisotropy(sorted_evals, reduced=True) == pytest.approx(2.5 * 2/3)

    # Test multiple tensors
    evals = np.array([
        [1.0, 2.0, 4.0],  # aniso = 2.5
        [10.0, 20.0, 40.0],  # aniso = 25
        [2, 1, -6], # aniso = -7.5
    ])
    sorted_evals = _evals_sort(evals, TensorConvention.Haeberlen)
    expected_aniso = np.array([2.5, 25.0, -7.5])
    np.testing.assert_array_almost_equal(_anisotropy(sorted_evals), expected_aniso)

def test_asymmetry():
    # Test asymmetry calculation
    # First sort in Haeberlen convention
    evals = np.array([[1.0, 2.0, 4.0]])  # Shape (1, 3)
    sorted_evals = _evals_sort(evals, TensorConvention.Haeberlen)
    # reduced_aniso = (4 - (1 + 2)/2) * 2/3 = 2.5 * 2/3
    # asymmetry = (2 - 1) / (2.5 * 2/3) = 0.6
    assert _asymmetry(sorted_evals) == pytest.approx(0.6)

    # Test symmetric case (zero asymmetry)
    evals = np.array([[1.0, 1.0, 2.0]])  # Shape (1, 3)
    sorted_evals = _evals_sort(evals, TensorConvention.Haeberlen)
    assert _asymmetry(sorted_evals) == pytest.approx(0.0)

    # Test multiple tensors
    evals = np.array([
        [1.0, 2.0, 4.0],  # asymmetry = 0.6
        [1.0, 1.0, 2.0],  # asymmetry = 0.0
    ])
    sorted_evals = _evals_sort(evals, TensorConvention.Haeberlen)
    expected_asymmetry = np.array([0.6, 0.0])
    np.testing.assert_array_almost_equal(_asymmetry(sorted_evals), expected_asymmetry)

    # Test zero anisotropy case - should return zero for asymmetry since it is undefined
    evals = np.array([[2.0, 2.0, 2.0]])  # Shape (1, 3)
    sorted_evals = _evals_sort(evals, TensorConvention.Haeberlen)
    asymmetry = _asymmetry(sorted_evals)
    assert asymmetry == pytest.approx(0.0)
