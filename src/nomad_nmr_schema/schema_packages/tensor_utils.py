# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Contains the NMRTensor class, simplifying the process of diagonalisation of an
NMR tensor as well as its representation in multiple conventions
"""

import warnings
from enum import Enum

import numpy as np

# Numerical tolerances for tensor operations
EIGENVALUE_EQUALITY_TOLERANCE: float = 1e-16
ISOTROPIC_ZERO_TOLERANCE: float = 1e-6

# Constants for tensor dimensions and validation
TENSOR_DIMENSION: int = 3
EIGENVALUE_TRIPLET_SIZE: int = 2

class TensorConvention(str, Enum):
    Haeberlen = "h"
    Increasing = "i"
    Decreasing = "d"
    NQR = "n"

    def sort_eigenvalues(
        self, evals: np.ndarray
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sort eigenvalues according to this convention.

        Args:
            evals: Array of shape (N, 3) containing N sets of eigenvalues

        Returns:
            Sorted eigenvalues array of same shape
        """
        evals = np.array(evals)
        # Special case: all eigenvalues equal (within tolerance)
        if np.all(np.abs(evals - evals[:, 0:1]) < EIGENVALUE_EQUALITY_TOLERANCE):
            return evals

        iso = np.average(evals, axis=1)

        if self in (self.Increasing, self.Decreasing):
            sorted_idx = np.argsort(evals, axis=1)
            if self == self.Decreasing:
                sorted_idx = sorted_idx[:, ::-1]

        elif self == self.Haeberlen:
            # Sort by deviation from isotropic value, with middle and first swapped
            deviations = np.abs(evals - iso[:, None])
            sorted_idx = np.argsort(deviations, axis=1)
            # Swap first two indices for Haeberlen convention
            temp = sorted_idx[:, 0].copy()
            sorted_idx[:, 0] = sorted_idx[:, 1]
            sorted_idx[:, 1] = temp

        elif self == self.NQR:
            # Sort by absolute values for NQR convention
            if np.any(np.abs(iso) > ISOTROPIC_ZERO_TOLERANCE):
                warnings.warn(
                    f"Isotropic values are not zero ({iso}), "
                    f"but NQR order is requested.\n"
                    "If you're dealing with an EFG tensor, "
                    "then check it carefully since these should be traceless.\n"
                    "Sorting by absolute values.\n"
                )
            sorted_idx = np.argsort(np.abs(evals), axis=1)

        return evals[np.arange(evals.shape[0])[:, None], sorted_idx]

    @classmethod
    def from_input(cls, input_str: str) -> 'TensorConvention':
        normalized = input_str.lower().strip()
        conversion_map = {
            'h': cls.Haeberlen,
            'haeberlen': cls.Haeberlen,
            'i': cls.Increasing,
            'increasing': cls.Increasing,
            'd': cls.Decreasing,
            'decreasing': cls.Decreasing,
            'n': cls.NQR,
            'nqr': cls.NQR
        }

        try:
            return conversion_map[normalized]
        except KeyError:
            raise ValueError(f"Invalid convention: {input_str}")

class NMRTensor:
    """NMRTensor class for handling NMR tensor operations and representations.

    Provides methods for:
    - Tensor diagonalization and eigenvalue sorting
    - Multiple convention representations (Haeberlen, NQR etc.)
    - Tensor property calculations (anisotropy, asymmetry etc.)

    Attributes:
        ORDER_INCREASING: Sort eigenvalues in increasing order
        ORDER_DECREASING: Sort eigenvalues in decreasing order
        ORDER_HAEBERLEN: Sort using Haeberlen convention
        ORDER_NQR: Sort using NQR convention
    """

    def __init__(self,
                 data: np.ndarray | tuple[np.ndarray, np.ndarray],
                 order: str | TensorConvention = TensorConvention.Increasing):
        """Initialize the NMRTensor.

        Args:
            data: Either a 3x3 matrix containing the tensor, or a tuple of
                [eigenvalues, eigenvectors] for the symmetric part alone.
            order: Order to use for eigenvalues/eigenvectors. Can be 'i'
                (increasing), 'd' (decreasing), 'h' (Haeberlen) or 'n' (NQR).
                Default is increasing order.

        Raises:
            ValueError: If data has invalid dimensions or format
        """
        self._order = None
        self._data = None
        self._symm = None
        self._evals = None
        self._evecs = None
        self._process_data(data)
        # The following will also sort the eigenvalues and eigenvectors
        self.order = order

        # Initialize other attributes
        self._anisotropy = None
        self._redaniso = None
        self._asymmetry = None
        self._span = None
        self._skew = None
        self._trace = None
        self._degeneracy = None

        self._incr_evals = _evals_sort([self._evals], 'i')[0]
        self._haeb_evals = _haeb_sort([self._evals])[0]

    def _process_data(self, data: np.ndarray | tuple[np.ndarray, np.ndarray]) -> None:
        """Process input data into tensor format.

        Args:
            data: Input tensor data or eigenvalue/eigenvector tuple

        Raises:
            ValueError: If data format or dimensions are invalid
        """
        if not isinstance(data, np.ndarray | tuple | list):
            raise ValueError("Data must be a numpy array, tuple, or list")

        if len(data) == TENSOR_DIMENSION:
            self._data = np.array(data, dtype=float)
            if self._data.shape != (TENSOR_DIMENSION, TENSOR_DIMENSION):
                dims = f"{TENSOR_DIMENSION}, {TENSOR_DIMENSION}"
                raise ValueError(f"Matrix data must have shape ({dims})")
            self._symm = (self._data + self._data.T) / 2.0
            evals, evecs = np.linalg.eigh(self._symm)
        elif len(data) == EIGENVALUE_TRIPLET_SIZE:
            evals, evecs = data
            evals = np.array(evals, dtype=float)
            evecs = np.array(evecs, dtype=float)
            if evals.shape != (3,) or evecs.shape != (3, 3):
                raise ValueError(
                    "Eigenvalues must have shape (3,) and eigenvectors shape (3, 3)"
                )
            self._symm = np.linalg.multi_dot([evecs, np.diag(evals), evecs.T])
            self._data = self._symm
        else:
            raise ValueError(
                "Data must be a 3x3 matrix or a pair of [eigenvalues, eigenvectors]"
            )

        rtol = EIGENVALUE_EQUALITY_TOLERANCE
        if not np.allclose(evecs @ evecs.T, np.eye(3), rtol=rtol):
            raise ValueError("Eigenvectors must form an orthogonal matrix")

        self._evals = evals
        self._evecs = evecs

    def _order_tensor(self, order):
        # Sort eigenvalues and eigenvectors as specified
        if self._order is None or self._order != order:
            self._evals, sort_i = _evals_sort([self._evals], order, True)
            self._evals = self._evals[0]
            self._evecs = self._evecs[:, sort_i[0]]
            # Last eigenvector must be the cross product of the first two
            self._evecs[:, 2] = np.cross(self._evecs[:, 0], self._evecs[:, 1])

        # For any property that depends on the eigenvalue order, reset it
        self._anisotropy = None
        self._redaniso = None
        self._asymmetry = None
        self._quat = None

    @property
    def order(self):
        return self._order
    # method to update the order of the tensor
    @order.setter
    def order(self, value):
        self._order_tensor(value)
        self._order = value

    @property
    def data(self) -> np.ndarray:
        """Get the raw tensor data.

        Returns:
            The 3x3 tensor matrix
        """
        return self._data.copy()

    @property
    def eigenvalues(self) -> np.ndarray:
        """Get the eigenvalues in current ordering convention.

        Returns:
            Array of 3 eigenvalues
        """
        return self._evals.copy()

    @property
    def eigenvectors(self) -> np.ndarray:
        """Get the eigenvectors in current ordering convention.

        Returns:
            3x3 array of eigenvectors as columns
        """
        return self._evecs.copy()

    @property
    def trace(self):
        if self._trace is None:
            self._trace = np.trace(self._data)
        return self._trace

    @property
    def isotropy(self):
        return self.trace / 3.0

    @property
    def anisotropy(self):
        if self._anisotropy is None:
            self._anisotropy = _anisotropy(self._haeb_evals[None, :])[0]
        return self._anisotropy

    @property
    def reduced_anisotropy(self):
        if self._redaniso is None:
            self._redaniso = _anisotropy(self._haeb_evals[None, :], True)[0]
        return self._redaniso

    @property
    def asymmetry(self):
        if self._asymmetry is None:
            self._asymmetry = _asymmetry(self._haeb_evals[None, :])[0]
        return self._asymmetry

    @property
    def span(self):
        if self._span is None:
            self._span = _span(self._incr_evals[None, :])[0]
        return self._span

    @property
    def skew(self):
        if self._skew is None:
            self._skew = _skew(self._incr_evals[None, :])[0]
        return self._skew

def _evals_sort(
    evals: np.ndarray,
    convention: str | TensorConvention = "i",
    return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sort a list of eigenvalue triplets by various conventions

    Args:
        evals: Array of eigenvalue triplets to sort
        convention: The sorting convention to use
        return_indices: Whether to return the sorting indices

    Returns:
        Sorted eigenvalues, and optionally the sorting indices
    """
    if isinstance(convention, str):
        convention = TensorConvention.from_input(convention)

    sorted_evals = convention.sort_eigenvalues(evals)

    if not return_indices:
        return sorted_evals
    else:
        # Calculate indices by comparing original with sorted
        indices = np.array([np.where(row == orig_row[:, None])[1]
                          for row, orig_row in zip(sorted_evals, evals)])
        return sorted_evals, indices


def _haeb_sort(evals, return_indices=False):
    return _evals_sort(evals, "h", return_indices)


def _anisotropy(haeb_evals, reduced=False):
    """Calculate anisotropy given eigenvalues sorted with Haeberlen
    convention"""

    f = 2.0 / 3.0 if reduced else 1.0

    return (haeb_evals[:, 2] - (haeb_evals[:, 0] + haeb_evals[:, 1]) / 2.0) * f


def _asymmetry(haeb_evals):
    """Calculate asymmetry
    Note that when the anisotropy is zero, the asymmetry is not defined
    and we set it to zero.
    """

    aniso = _anisotropy(haeb_evals, reduced=True)
    # Fix the anisotropy zero values
    aniso = np.where(aniso == 0, np.inf, aniso)

    return (haeb_evals[:, 1] - haeb_evals[:, 0]) / aniso


def _span(evals):
    """Calculate span

    .. math::
        \\Omega = \\sigma_{33} - \\sigma_{11}

    where :math:`\\sigma_{33}` is the largest, and :math:`\\sigma_{11}` is the
    smallest eigenvalue.

    """

    return np.amax(evals, axis=-1) - np.amin(evals, axis=-1)


def _skew(evals):
    """Calculate skew

    .. math::
        \\kappa = 3 (\\sigma_{iso} - \\sigma_{22}) / \\Omega

    where :math:`\\Omega` is the span of the tensor.

    When the span is zero, the skew is not defined and we set it to
    zero.

    Note that for chemical shift tensors (:math:`\\delta`), the sign is reversed.
    """

    span = _span(evals)
    span = np.where(span == 0, np.inf, span)
    return 3 * (np.average(evals, axis=1) - np.median(evals, axis=1)) / span
