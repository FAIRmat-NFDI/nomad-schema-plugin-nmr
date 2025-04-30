from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad.datamodel.metainfo.basesections import Entity
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.outputs import Outputs as BaseOutputs
from nomad_simulations.schema_packages.physical_property import PhysicalProperty

m_package = SchemaPackage()


def resolve_name_from_entity_ref(entities: list[Entity], logger: 'BoundLogger') -> str:
    """
    Resolves the `name` of the atom-resolved `PhysicalProperty` from the `entity_ref`
    by assigning a label corresponding to the `AtomsState.chemical_symbol` and a number
    corresponding to the position in the list of `AtomsState`.

    Args:
        entities (list[Entity]): The list of entities to resolve the name from.
        logger ('BoundLogger'): The logger to log messages.

    Returns:
        (str): The resolved name of the atom-resolved `PhysicalProperty`.
    """
    name = ''
    for entity in entities:
        atoms_state = entity
        # Check if `entity_ref` exists and it is an AtomsState
        if not atoms_state or not isinstance(atoms_state, AtomsState):
            logger.error(
                'Could not find `entity_ref` referencing an `AtomsState` section.'
            )
            return ''
        # Check if the parent of `entity_ref` exists
        cell = atoms_state.m_parent
        if not cell:
            logger.warning(
                'The parent of the `AtomsState` in `entity_ref` does not exist.'
            )
            return ''

        index = ''  # ! implement here if needed
        name += f'{atoms_state.chemical_symbol}{index}'
    return name


def extract_eigenvalues(
    tensor: np.ndarray, logger: 'BoundLogger', convention: str = 's'
) -> tuple[float, float, float] | None:
    """
    Extract the eigenvalues of a 3x3 tensor and sort them based on the specified
    sorting method.

    Conventions:
    - 's' (Standard): Sort eigenvalues such that |val_zz| >= |val_yy| >= |val_xx|.
    - 'h' (Haeberlen): Sort eigenvalues such that
      |val_zz - val_iso| >= |val_xx - val_iso| >= |val_yy - val_iso|. Here,
      val_iso is defined as the average of the three eigenvalues:
      val_iso = np.trace(tensor) / 3.0

    Args:
        tensor (np.ndarray): The 3x3 tensor for which eigenvalues are to be extracted.
        logger ('BoundLogger'): The logger to log messages.
        convention (str, optional): Sorting convention. Defaults to 's'.
            - 's': Standard sorting by absolute values.
            - 'h': Haeberlen convention sorting.

    Returns:
        (Optional[tuple[float, float, float]]): The sorted eigenvalues.
    """
    try:
        # Compute eigenvalues and eigenvectors, discard eigenvectors
        eigenvalues, _ = np.linalg.eigh(tensor)

        # Sort eigenvalues in ascending order
        if convention == 's':
            # Standard sorting by absolute values
            sorted_eigenvalues = sorted(eigenvalues, key=abs)  # Sort by absolute value
            # Returned sorted eigenvalues in increasing order
            return sorted_eigenvalues[0], sorted_eigenvalues[1], sorted_eigenvalues[2]
        elif convention == 'h':
            # Haeberlen convention sorting
            iso = np.trace(tensor) / 3.0  # Calculate isotropic value
            sorted_eigenvalues = sorted(eigenvalues, key=lambda x: abs(x - iso))
            # Return sorted eigenvalues with xx and yy swapped - Haeberlen convention
            return sorted_eigenvalues[1], sorted_eigenvalues[0], sorted_eigenvalues[2]
        else:
            logger.error(f"Invalid convention '{convention}' specified.")
            return None

    except Exception as e:
        logger.warning(f'Could not extract eigenvalues of the tensor: {e}')
        return None


class MagneticShielding(PhysicalProperty):
    """
    Nuclear response of a material to shield the effects of an applied external field.
    This is a tensor 3x3 related with the induced magnetic field as:

        B_induced = - magnetic_shielding * B_external

    See, e.g, https://pubs.acs.org/doi/10.1021/cr300108a.

    This property will appear as a list under `Outputs` where each of the elements
    correspond to an atom in the unit cell.
    The specific atom is known by defining the reference to the specific `AtomsState`
    under `ModelSystem.cell.atoms_state` using `entity_ref`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='dimensionless',
        description="""
        Value of the magnetic shielding tensor per atom.
        """,
    )
    ms_isotropic = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The isotropic component of the `MagneticShielding` tensor. The isotropic
        magnetic shielding is defined as the average of the three principal components
        of the magnetic shielding tensor:

            ms_isotropic = (sigma_xx + sigma_yy + sigma_zz) / 3

        where sigma_xx, sigma_yy, and sigma_zz are the eigenvalues of the magnetic
        shielding tensor, sorted based on Haeberlen convention (see
        `extract_eigenvalues()` function).

        Alternatively, this is 1/3 of the trace of the magnetic shielding tensor (see
        `extract_isotropic_part()` function below). Both formulas evaluate to the same
        numerical value.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    ms_anisotropy = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The magnetic shielding anisotropy is defined as:

            ms_anisotropy = sigma_zz - (sigma_xx + sigma_yy) / 2.0

        where sigma_xx, sigma_yy, and sigma_zz are the eigenvalues of the magnetic
        shielding tensor, sorted based on the Haeberlen convention (see
        `extract_eigenvalues()` function).

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    ms_reduced_anisotropy = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The reduced anisotropy is defined as:

            ms_reduced_anisotropy = sigma_zz - ms_isotropic

        where sigma_zz is the eigenvalue of the magnetic shielding tensor with the
        largest deviation from the isotropic value as per Haeberlen convention (see
        `extract_eigenvalues()` function) and ms_isotropic is the isotropic component of
        the tensor, defined as:

            ms_isotropic = (sigma_xx + sigma_yy + sigma_zz) / 3.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    ms_asymmetry = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The magnetic shielding asymmetry is defined as:

            ms_asymmetry = (sigma_yy - sigma_xx) / ms_reduced_anisotropy

        where sigma_xx, sigma_yy, and sigma_zz are the eigenvalues of the magnetic
        shielding tensor, sorted based on the Haeberlen convention (see
        `extract_eigenvalues()` function) and ms_reduced_anisotropy is defined as:

            ms_reduced_anisotropy = sigma_zz - ms_isotropic

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    ms_span = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The span is defined as:

            ms_span = sigma_33 - sigma_11

        where sigma_11, sigma_22, and sigma_33 are the eigenvalues of the magnetic
        shielding tensor, sorted based on the standard convention (see
        `extract_eigenvalues()` function). The span quantifies the range of the
        spectrum being analysed.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )

    ms_skew = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The skew is defined as:

            ms_skew = 3 * (ms_isotropic - sigma_22) / ms_span

        where sigma_11, sigma_22, and sigma_33 are the eigenvalues of the magnetic
        shielding tensor, sorted based on the standard convention (see
        `extract_eigenvalues()` function) ms_isotropic is the isotropic component of the
        tensor, as calculated before.

        This parameter quantifies the asymmetry of the magnetic shielding tensor around
        its isotropic value.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        # ! this info is in the shape attribute of the Quantity
        self.rank = [3, 3]
        self.name = self.m_def.name

    def extract_isotropic_part(self, logger: 'BoundLogger') -> float | None:
        """
        Extract the isotropic component of the magnetic shielding tensor. This is 1/3 of
        the trace of the magnetic shielding tensor `value`.

        Args:
            logger ('BoundLogger'): The logger to log messages.

        Returns:
            (Optional[float]): The isotropic component of the magnetic shielding tensor.
        """
        try:
            # Calculate the isotropic value
            isotropic = np.trace(np.array(self.value)) / 3.0
        except Exception:
            logger.warning('Could not extract the trace of the `value` tensor.')
            return None
        return isotropic

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )

        # isotropic value extraction
        isotropic = self.extract_isotropic_part(logger)
        if isotropic is not None:
            logger.info(f'Appending isotropic value for {self.name}')
            self.ms_isotropic = isotropic
        else:
            logger.warning(f'Isotropic value extraction failed for {self.name}')

        # Extract eigenvalues by Haeberlen convention and calculate magnetic shielding
        # anisotropy, reduced anisotropy, and asymmetry.
        eigenvalues_haeberlen = extract_eigenvalues(
            np.array(self.value), logger, convention='h'
        )
        if eigenvalues_haeberlen is not None:
            sigma_xx, sigma_yy, sigma_zz = eigenvalues_haeberlen
            logger.info(
                f'Eigenvalues for {self.name} (Haeberlen convention): '
                f'sigma_xx={sigma_xx}, sigma_yy={sigma_yy}, sigma_zz={sigma_zz}'
            )

            # Calculate ms_anisotropy
            self.ms_anisotropy = sigma_zz - (sigma_xx + sigma_yy) / 2.0
            logger.info(
                f'Magnetic shielding anisotropy for {self.name}: {self.ms_anisotropy}'
            )

            # Calculate ms_reduced_anisotropy
            self.ms_reduced_anisotropy = sigma_zz - self.ms_isotropic
            logger.info(
                f'Magnetic shielding reduced anisotropy for {self.name}: '
                f'{self.ms_reduced_anisotropy}'
            )

            # Calculate ms_asymmetry
            if self.ms_reduced_anisotropy != 0:
                self.ms_asymmetry = (sigma_yy - sigma_xx) / self.ms_reduced_anisotropy
                logger.info(
                    f'Magnetic shielding asymmetry for {self.name}: {self.ms_asymmetry}'
                )
            else:
                logger.warning(
                    f'Cannot calculate Magnetic shielding asymmetry for {self.name} as '
                    f'reduced anisotropy is zero.'
                )
        else:
            logger.warning(f'Failed to extract eigenvalues for {self.name}')
            return  # Exit early if eigenvalues extraction fails

        # Extract eigenvalues by standard convention to calculate span and skew.
        eigenvalues_standard = extract_eigenvalues(
            np.array(self.value), logger, convention='s'
        )
        if eigenvalues_standard is not None:
            sigma_11, sigma_22, sigma_33 = eigenvalues_standard
            logger.info(
                f'Eigenvalues for {self.name} (Standard convention): '
                f'sigma_11={sigma_11}, sigma_22={sigma_22}, sigma_33={sigma_33}'
            )

            # Calculate ms_span
            self.ms_span = sigma_33 - sigma_11
            logger.info(f'Magnetic Shielding Span for {self.name}: {self.ms_span}')

            # Calculate ms_skew
            if self.ms_span != 0:
                self.ms_skew = 3 * (self.ms_isotropic - sigma_22) / self.ms_span
                logger.info(f'Magnetic Shielding skew for {self.name}: {self.ms_skew}')
            else:
                logger.warning(
                    f'Cannot calculate ms_skew for {self.name} as ms_span is zero.'
                )
        else:
            logger.warning(f'Failed to extract eigenvalues for {self.name}')


class ElectricFieldGradient(PhysicalProperty):
    """
    Interaction between the quadrupole moment of the nucleus and the electric field
    gradient (EFG) at the nucleus position generated by the surrounding charges.
    This property is relevant for Nuclear Magnetic Resonance (NMR). The eigenvalues
    of these tensors can be used to compute the `asymmetry_parameter`.

    See, e.g, https://pubs.acs.org/doi/10.1021/cr300108a.

    This class by default refers to the 'total' contribution to the EFG. This property
    will appear as a list under `Outputs` where each of the elements correspond to an
    atom in the unit cell.
    The specific atom is known by defining the reference to the specific `AtomsState`
    under `ModelSystem.cell.atoms_state` using `entity_ref`.
    """

    # The below code is commented out because we want to capture the local and non-local
    # contributions to the EFG in their own classes.
    # type = Quantity(
    #     type=MEnum("total", "local", "non_local"),
    #     description="""
    #     Type of contribution to the electric field gradient (EFG). The total EFG can
    #     be decomposed on the `local` and `non_local` contributions.
    #     """,
    # )

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='volt / meter ** 2',
        description="""
        The electric field gradient (EFG) tensor.
        """,
    )
    value_Vzz = Quantity(
        type=np.float64,
        unit='dimensionless',
        description='Largest eigenvalue of the EFG tensor, obtained by sorting based on'
        'standard convention (see `extract_eigenvalues()` function).',
    )

    # quadrupolar_coupling_constant = Quantity(
    #     type=np.float64,
    #     description="""
    #     Quadrupolar coupling constant for each atom in the unit cell.
    #     Once the eigenvalues of the EFG tensors are computed, it is computed as:

    #         quadrupolar_coupling_constant = efg_zz * e * Q / h

    #     where efg_zz is the largest eigenvalue of the EFG tensor,
    #     Q is the nuclear quadrupole moment, e is the elementary charge, and
    #     h is the Planck's constant.
    #     """,
    # )

    quadrupolar_asymmetry = Quantity(
        type=np.float64,
        description="""
        The quadrupolar asymmetry parameter for each atom in the unit cell. It is
        computed from the eigenvalues of the EFG tensor as:

            quadrupolar_asymmetry, eta_Q = (V_xx - V_yy) / V_zz

        where V_xx, V_yy and V_zz are the eigenvalues of the EFG tensor ordered
        such that |V_zz| >= |V_yy| >= |V_xx|.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions  !!! TODO
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )

        # Extract eigenvalues and calculate quadrupolar asymmetry parameter
        eigenvalues_standard = extract_eigenvalues(
            np.array(self.value), logger, convention='s'
        )
        if eigenvalues_standard is not None:
            value_Vxx, value_Vyy, self.value_Vzz = (
                eigenvalues_standard  # store only Vzz
            )
            logger.info(f'Eigenvalue for {self.name}: Vzz={self.value_Vzz}')
            if self.value_Vzz != 0:
                self.quadrupolar_asymmetry = (value_Vxx - value_Vyy) / self.value_Vzz
                logger.info(
                    f'Quadrupolar asymmetry parameter for {self.name}: '
                    f' {self.quadrupolar_asymmetry}'
                )
            else:
                logger.warning(
                    f'Cannot calculate quadrupolar asymmetry parameter for {self.name} '
                    f'as Vzz is zero.'
                )
        else:
            logger.warning(f'Failed to extract eigenvalues for {self.name}')


class ElectricFieldGradientLocal(PhysicalProperty):
    """
    Represents the local contribution to the electric field gradient (EFG) at the
    nucleus position. Some DFT codes may provide the EFG decomposed into
    theory-dependent contributions. The local contribution is one of these decomposed
    contributions, identified by a 'efg_local' tag in the magres data block of a
    .magres file.

    This property is relevant for Nuclear Magnetic Resonance (NMR) and will appear
    as a list under `Outputs` where each element corresponds to an atom in the unit
    cell.

    The specific atom is known by defining the reference to the specific `AtomsState`
    under `ModelSystem.cell.atoms_state` using `entity_ref`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='volt / meter ** 2',
        description="""
        Value of the local electric field gradient (EFG) tensor for each atom.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions  !!! TODO
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )


class ElectricFieldGradientNonlocal(PhysicalProperty):
    """
    Represents the non-local contribution to the electric field gradient (EFG) at the
    nucleus position. Some DFT codes may provide the EFG decomposed into
    theory-dependent contributions. The non-local contribution is one of these
    decomposed contributions, identified by a 'efg_nonlocal' tag in the magres data
    block of a .magres file.

    This property is relevant for Nuclear Magnetic Resonance (NMR) and will appear
    as a list under `Outputs` where each element corresponds to an atom in the unit
    cell.

    The specific atom is known by defining the reference to the specific `AtomsState`
    under `ModelSystem.cell.atoms_state` using `entity_ref`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='volt / meter ** 2',
        description="""
        Value of the non-local electric field gradient (EFG) tensor for each atom.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions  !!! TODO
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )


# class ElectricFieldGradients(BaseOutputs):
#     """
#     Represents the electric field gradients (EFG) data.

#     This class contains the total, local, and non-local components
#     of the electric field gradients.
#     Each component is represented as a subsection of the ElectricFieldGradient class
#     and can have multiple entries.

#     Attributes:
#         efg_total (SubSection): A list of total electric field gradient entries.
#         efg_local (SubSection): A list of local electric field gradient entries.
#         efg_nonlocal (SubSection): A list of non-local electric field gradient
#         entries.
#     """

#     efg_total = SubSection(sub_section=ElectricFieldGradient.m_def, repeats=True)
#     efg_local = SubSection(sub_section=ElectricFieldGradient.m_def, repeats=True)
#     efg_nonlocal = SubSection(sub_section=ElectricFieldGradient.m_def, repeats=True)


class IndirectSpinSpinCoupling(PhysicalProperty):
    """
    Indirect exchanges or interactions between 2 nuclear spins that arises from
    hyperfine interactions between the nuclei and local electrons. This parameter is
        identified by the 'isc' tag in the magres data block of a .magres file.

    The total indirect coupling can be decomposed into the following contributions,
    which can each be output by some DFT codes:
        - Fermi contact (tag 'isc_fc' in the magres data block)
        - Orbital paramagnetic (tag 'isc_orbital_p' in the magres data block)
        - Orbital diamagnetic (tag 'isc_orbital_d' in the magres data block)
        - Spin dipolar (tag 'isc_spin' in the magres data block)
    These contributions will be captured in their own classes.

    This property will appear as a list under `Outputs` where each of the elements
    correspond to an atom-atom coupling term. The specific pair of atoms defined for
    the coupling is known by referencing the specific `AtomsState`
    under `ModelSystem.cell.atoms_state` using `entity_ref_1` and `entity_ref_2`.
    """

    # TODO dipolar (or direct) coupling needs to be included at a higher level (app),
    # potentially calculated by Soprano python library to compute the overall spin-spin
    # coupling.

    # we hide `entity_ref` from `PhysicalProperty` to avoid confusion
    m_def = Section(a_eln={'hide': ['entity_ref']})

    # The below code is commented out because we want to capture the decomposed
    # contributions to the indirect spin-spin coupling in their own classes.
    # type = Quantity(
    #     type=MEnum(
    #         "total",
    #         "direct_dipolar",
    #         "fermi_contact",
    #         "orbital_diamagnetic",
    #         "orbital_paramagnetic",
    #         "spin_dipolar",
    #     ),
    #     description="""
    #     Type of contribution to the indirect spin-spin coupling. The total indirect
    #     spin-spin coupling is composed of:

    #         `total` = `direct_dipolar` + J_coupling

    #     Where the J_coupling is:
    #         J_coupling = `fermi_contact`
    #                     + `spin_dipolar`
    #                     + `orbital_diamagnetic`
    #                     + `orbital_paramagnetic`

    #     See https://pubs.acs.org/doi/full/10.1021/cr300108a.
    #     """,
    # )

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        The total indirect spin-spin coupling tensor output from DFT codes is called the
        reduced spin coupling tensor K_ij, where i and j are nuclei between which the
        coupling is computed. The K_ij tensor is obtained from the magnetic field
        induced at nucleus i due to the perturbative effect of the magnetic moment of
        nucleus j as:

            K_ij = B_induced_i / magnetic_moment_j

        where B_induced_i is the induced magnetic field at nucleus i.

        Where the indirect spin-spin coupling is:
            indirect_spin_spin_coupling = `fermi_contact`
                                         + `spin_dipolar`
                                         + `orbital_diamagnetic`
                                         + `orbital_paramagnetic`

        See, https://pubs.acs.org/doi/full/10.1021/cr300108a.
        """,
    )
    K_isotropic = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The isotropic component of the reduced spin coupling tensor. The isotropic
        value is defined as the average of the three principal components of the
        reduced spin coupling tensor:

            K_isotropic = (K_xx + K_yy + K_zz) / 3

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention (see `extract_eigenvalues()`
        function).

        Alternatively, this is 1/3 of the trace of the reduced spin coupling tensor (see
        `extract_isotropic_part()` function below). Both formulas evaluate to the same
        numerical value.
        """,
    )
    K_symmetric = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        The symmetric component of the reduced spin coupling tensor. This is defined as:

            K_symmetric = (K + K.T) / 2

        where K is the reduced spin coupling tensor, K.T is the transposed K tensor.
        """,
    )
    K_asymmetric = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        The asymmetric component of the reduced spin coupling tensor. This is defined
        as:

            K_asymmetric = (K - K.T) / 2

        where K is the reduced spin coupling tensor, K.T is the transposed K tensor.
        """,
    )
    K_anisotropy = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The reduced spin couling anisotropy is defined as:

            isc_anisotropy = K_zz - (K_xx + K_yy) / 2.0

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention (see `extract_eigenvalues()`
        function).
        """,
    )
    K_principal_component_asymmetry = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The principal component asymmetry is defined as:

            K_principal_component_asymmetry = (K_yy - K_xx) / (Kzz - K_isotropic)

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention (see `extract_eigenvalues()`
        function) and K_isotropic is the isotropic component of the tensor, as
        calculated before.
        """,
    )

    entity_ref_1 = Quantity(
        type=Entity,
        description="""
        Reference to the first entity that the coupling refers to. In this case, this
        is the first `AtomsState` in the pair of atoms that the coupling refers to.
        """,
    )

    entity_ref_2 = Quantity(
        type=Entity,
        description="""
        Reference to the second entity that the coupling refers to. In this case, this
        is the second `AtomsState` in the pair of atoms that the coupling refers to.
        """,
    )

    def extract_isotropic_part(self, logger: 'BoundLogger') -> float | None:
        """
        Extract the isotropic component of the reduced spin coupling tensor. This is 1/3
        of the trace of the reduced spin coupling tensor `value`.

        Args:
            logger ('BoundLogger'): The logger to log messages.

        Returns:
            (Optional[float]): The isotropic component of the reduced spin coupling
            tensor.
        """
        try:
            # Calculate the isotropic value
            isotropic = np.trace(np.array(self.value)) / 3.0
        except Exception:
            logger.warning('Could not extract the trace of the `value` tensor.')
            return None
        return isotropic

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref_1, self.entity_ref_2], logger=logger
        )

        # Calculate the isotropic component (K_isotropic)
        isotropic = self.extract_isotropic_part(logger)
        if isotropic is not None:
            logger.info(f'Appending isotropic value for {self.name}')
            self.K_isotropic = isotropic
        else:
            logger.warning(f'Isotropic value extraction failed for {self.name}')

        # Calculate the symmetric and asymmetric components
        try:
            tensor = np.array(self.value)
            self.K_symmetric = (tensor + tensor.T) / 2.0
            self.K_asymmetric = (tensor - tensor.T) / 2.0
            logger.info(f'K_symmetric for {self.name}: {self.K_symmetric}')
            logger.info(f'K_asymmetric for {self.name}: {self.K_asymmetric}')
        except Exception as e:
            logger.warning(
                f'Failed to calculate symmetric/asymmetric components for '
                f'{self.name}: {e}'
            )

        # Extract eigenvalues using the Haeberlen convention
        eigenvalues_haeberlen = extract_eigenvalues(
            np.array(self.value), logger, convention='h'
        )
        if eigenvalues_haeberlen is not None:
            K_xx, K_yy, K_zz = eigenvalues_haeberlen
            logger.info(
                f'Eigenvalues for {self.name} (Haeberlen convention): '
                f'K_xx={K_xx}, K_yy={K_yy}, K_zz={K_zz}'
            )

            # Calculate K_anisotropy
            self.K_anisotropy = K_zz - (K_xx + K_yy) / 2.0
            logger.info(f'K_anisotropy for {self.name}: {self.K_anisotropy}')

            # Calculate K_principal_component_asymmetry
            if (K_zz - self.K_isotropic) != 0:
                self.K_principal_component_asymmetry = (K_yy - K_xx) / (
                    K_zz - self.K_isotropic
                )
                logger.info(
                    f'K_principal_component_asymmetry for {self.name}: '
                    f'{self.K_principal_component_asymmetry}'
                )
            else:
                logger.warning(
                    f'Cannot calculate K_principal_component_asymmetry for {self.name} '
                    f'as (K_zz - K_isotropic) is zero.'
                )
        else:
            logger.warning(f'Failed to extract eigenvalues for {self.name}')


class IndirectSpinSpinCouplingFermiContact(PhysicalProperty):
    """
    Represents the Fermi contact contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_fc' tag in the magres data block
    of a .magres file.

    This property will appear as a list under `Outputs` where each element corresponds
    to an atom-atom coupling term. The specific pair of atoms is known by referencing
    the specific `AtomsState` under `ModelSystem.cell.atoms_state` using `entity_ref_1`
    and `entity_ref_2`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        Value of the Fermi contact contribution to the indirect spin-spin coupling
        tensor.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref_1, self.entity_ref_2], logger=logger
        )


class IndirectSpinSpinCouplingOrbitalDiamagnetic(PhysicalProperty):
    """
    Represents the orbital diamagnetic contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_orbital_d' tag in the magres data block
    of a .magres file.

    This property will appear as a list under `Outputs` where each element corresponds
    to an atom-atom coupling term. The specific pair of atoms is known by referencing
    the specific `AtomsState` under `ModelSystem.cell.atoms_state` using `entity_ref_1`
    and `entity_ref_2`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        Value of the orbital diamagnetic contribution to the indirect spin-spin coupling
        tensor.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref_1, self.entity_ref_2], logger=logger
        )


class IndirectSpinSpinCouplingOrbitalParamagnetic(PhysicalProperty):
    """
    Represents the orbital paramagnetic contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_orbital_p' tag in the magres data block
    of a .magres file.

    This property will appear as a list under `Outputs` where each element corresponds
    to an atom-atom coupling term. The specific pair of atoms is known by referencing
    the specific `AtomsState` under `ModelSystem.cell.atoms_state` using `entity_ref_1`
    and `entity_ref_2`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        Value of the orbital paramagnetic contribution to the indirect spin-spin
        coupling tensor.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref_1, self.entity_ref_2], logger=logger
        )


class IndirectSpinSpinCouplingSpinDipolar(PhysicalProperty):
    """
    Represents the spin dipolar contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_spin' tag in the magres data block
    of a .magres file.

    This property will appear as a list under `Outputs` where each element corresponds
    to an atom-atom coupling term. The specific pair of atoms is known by referencing
    the specific `AtomsState` under `ModelSystem.cell.atoms_state` using `entity_ref_1`
    and `entity_ref_2`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        Value of the spin dipolar contribution to the indirect spin-spin coupling
        tensor.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref_1, self.entity_ref_2], logger=logger
        )


class MagneticSusceptibility(PhysicalProperty):
    """
    Section containing the information of magnetic susceptibility tensor. Degree of
    magnetization of a material in the presence of a magnetic field.

    See, e.g, https://doi.org/10.1039/9781837673179-00061.

    This tensor is identified by the 'sus' tag in the magres data block of a .magres
    file.
    """

    m_def = Section(validate=False)

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        # unit='10 ** -6 * cm ** 3 / mol', # avoid adding a scaling factor in the units,
        # any scaling may have to be done when displaying data on GUI, if so preferred.
        unit='m ** 3 / mol',
        description="""
        Value of the macroscopic magnetic susceptibility tensor.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions
        self.name = self.m_def.name  # Explicitly setting the name attribute

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Outputs(BaseOutputs):
    """
    The outputs of the principal metadata for NMR.
    """

    magnetic_shieldings = SubSection(sub_section=MagneticShielding.m_def, repeats=True)
    electric_field_gradients = SubSection(
        sub_section=ElectricFieldGradient.m_def, repeats=True
    )
    electric_field_gradients_local = SubSection(
        sub_section=ElectricFieldGradientLocal.m_def, repeats=True
    )
    electric_field_gradients_nonlocal = SubSection(
        sub_section=ElectricFieldGradientNonlocal.m_def, repeats=True
    )
    indirect_spin_spin_couplings = SubSection(
        sub_section=IndirectSpinSpinCoupling.m_def, repeats=True
    )
    indirect_spin_spin_couplings_fermi_contact = SubSection(
        sub_section=IndirectSpinSpinCouplingFermiContact.m_def, repeats=True
    )
    indirect_spin_spin_couplings_orbital_p = SubSection(
        sub_section=IndirectSpinSpinCouplingOrbitalParamagnetic.m_def, repeats=True
    )
    indirect_spin_spin_couplings_orbital_d = SubSection(
        sub_section=IndirectSpinSpinCouplingOrbitalDiamagnetic.m_def, repeats=True
    )
    indirect_spin_spin_couplings_spin_dipolar = SubSection(
        sub_section=IndirectSpinSpinCouplingSpinDipolar.m_def, repeats=True
    )
    magnetic_susceptibilities = SubSection(
        sub_section=MagneticSusceptibility.m_def, repeats=True
    )


m_package.__init_metainfo__()
