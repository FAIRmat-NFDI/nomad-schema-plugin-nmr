from typing import TYPE_CHECKING

import numpy as np
from structlog.stdlib import BoundLogger

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section

from nomad.datamodel.metainfo.basesections import Entity
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.outputs import Outputs as BaseOutputs
from nomad_simulations.schema_packages.physical_property import PhysicalProperty

from .tensor_utils import NMRTensor, TensorConvention

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

        # index = ''  # ! implement here if needed
        index = atoms_state.index if hasattr(atoms_state, 'index') else ''
        name += f'{atoms_state.chemical_symbol}{index}'
    return name


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
    TODO: these should be ppm
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='dimensionless',
        description="""
        Value of the magnetic shielding tensor per atom.
        """,
    )
    isotropy = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The isotropy component of the `MagneticShielding` tensor. The isotropy
        magnetic shielding is defined as the average of the three principal components
        of the magnetic shielding tensor:

            isotropy = (sigma_xx + sigma_yy + sigma_zz) / 3

        where sigma_xx, sigma_yy, and sigma_zz are the eigenvalues of the magnetic
        shielding tensor, sorted based on Haeberlen convention.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    anisotropy = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The magnetic shielding anisotropy is defined as:

            anisotropy = sigma_zz - (sigma_xx + sigma_yy) / 2.0

        where sigma_xx, sigma_yy, and sigma_zz are the eigenvalues of the magnetic
        shielding tensor, sorted based on the Haeberlen convention function).

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    reduced_anisotropy = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The reduced anisotropy is defined as:

            reduced_anisotropy = sigma_zz - isotropy

        where sigma_zz is the eigenvalue of the magnetic shielding tensor with the
        largest deviation from the isotropy value as per Haeberlen convention.
        For clarity, isotropy is defined as:

            isotropy = (sigma_xx + sigma_yy + sigma_zz) / 3.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    asymmetry = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The magnetic shielding asymmetry is defined as:

            asymmetry = (sigma_yy - sigma_xx) / reduced_anisotropy

        where sigma_xx, sigma_yy, and sigma_zz are the eigenvalues of the magnetic
        shielding tensor, sorted based on the Haeberlen convention function).

            reduced_anisotropy = sigma_zz - isotropy

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )
    span = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The span is defined as:

            span = sigma_33 - sigma_11

        where sigma_11, sigma_22, and sigma_33 are the eigenvalues of the magnetic
        shielding tensor, sorted based on the standard convention function.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )

    skew = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The skew is defined as:

            skew = 3 * (isotropy - sigma_22) / span

        where sigma_11, sigma_22, and sigma_33 are the eigenvalues of the magnetic
        shielding tensor, sorted based on the standard convention function.

        This parameter quantifies the asymmetry of the magnetic shielding tensor around
        its isotropy value.

        See, e.g, https://doi.org/10.1039/C6CC02542K.
        """,
    )

    def __init__(
        self, m_def: 'Section' = None, m_context: 'Context' = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.name = self.m_def.name
        self.isotropy = None
        self.anisotropy = None
        self.reduced_anisotropy = None
        self.asymmetry = None
        self.span = None
        self.skew = None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )

        # Initialise the tensor with the Haeberlen convention
        tensor = NMRTensor(np.array(self.value), TensorConvention.Haeberlen)

        # Calculate properties
        self.isotropy = tensor.isotropy
        self.anisotropy = tensor.anisotropy
        self.reduced_anisotropy = tensor.reduced_anisotropy
        self.asymmetry = tensor.asymmetry

        # Log all properties
        props = {
            'isotropy': self.isotropy,
            'anisotropy': self.anisotropy,
            'reduced_anisotropy': self.reduced_anisotropy,
            'asymmetry': self.asymmetry,
        }
        for prop, value in props.items():
            logger.info(f'MS {prop} for {self.name}: {value}')

        # Span and skew
        self.span = tensor.span
        self.skew = tensor.skew
        logger.info(f'Magnetic Shielding Span for {self.name}: {self.span}')
        logger.info(f'Magnetic Shielding skew for {self.name}: {self.skew}')


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

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='au',
        description="""
        The electric field gradient (EFG) tensor.
        """,
    )
    Vzz = Quantity(
        type=np.float64,
        unit='au',
        description='Largest (absolute)eigenvalue of the EFG tensor',
    )

    asymmetry = Quantity(
        type=np.float64,
        unit='dimensionless',
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
        self.name = self.m_def.name

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )

        tensor = NMRTensor(np.array(self.value))
        # Using standard convention for EFG
        tensor.order = TensorConvention.NQR
        eigenvalues = tensor.eigenvalues

        # Store largest eigenvalue (Vzz)
        self.Vzz = eigenvalues[2]
        logger.info(f'Eigenvalue for {self.name}: Vzz={self.Vzz}')

        # Calculate quadrupolar asymmetry
        self.asymmetry = tensor.asymmetry
        logger.info(f'Asymmetry for {self.name}: {self.asymmetry}')


class BaseIndirectSpinSpinCoupling(PhysicalProperty):
    """
    Base class for all indirect spin-spin coupling classes. This represents the
    common structure
    and behavior of various types of indirect spin-spin couplings in NMR.

    This is used as a base for:
    - Total indirect coupling (IndirectSpinSpinCoupling)
    - Fermi contact contribution (IndirectSpinSpinCouplingFermiContact)
    - Orbital paramagnetic contribution (IndirectSpinSpinCouplingOrbitalParamagnetic)
    - Orbital diamagnetic contribution (IndirectSpinSpinCouplingOrbitalDiamagnetic)
    - Spin dipolar contribution (IndirectSpinSpinCouplingSpinDipolar)
    """

    # we hide `entity_ref` from `PhysicalProperty` to avoid confusion
    m_def = Section(a_eln={'hide': ['entity_ref']})

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='tesla ** 2 / joule',
        description="""
        Value of the indirect spin-spin coupling tensor or one of its contributions.
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

    isotropy = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The isotropic component of the reduced spin coupling tensor. The isotropic
        value is defined as the average of the three principal components of the
        reduced spin coupling tensor:

            isotropy = (K_xx + K_yy + K_zz) / 3

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention.
        """,
    )

    anisotropy = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The reduced spin couling anisotropy is defined as:

            isc_anisotropy = K_zz - (K_xx + K_yy) / 2.0

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention.
        """,
    )

    reduced_anisotropy = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The reduced anisotropy is defined as:

            reduced_anisotropy = K_zz - isotropy

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention and isotropy is the
        isotropic component of the tensor.
        """,
    )

    asymmetry = Quantity(
        type=np.float64,
        unit='dimensionless',
        description="""
        The principal component asymmetry is defined as:

            asymmetry = (K_yy - K_xx) / (K_zz - isotropy)

        where K_xx, K_yy, and K_zz are the eigenvalues of the reduced spin coupling
        tensor, sorted based on Haeberlen convention and isotropy is the
        isotropic component of the tensor.
        """,
    )

    span = Quantity(
        type=np.float64,
        unit='tesla ** 2 / joule',
        description="""
        The span is defined as:
            span = K_33 - K_11

        where K_11, K_22, and K_33 are the eigenvalues of the reduced spin coupling
        tensor, sorted based on the standard convention.
        The span quantifies the range of the spectrum being analysed.
        See, e.g, https://doi.org/10.1039/C6CC02542K.
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

        tensor = NMRTensor(np.array(self.value), order=TensorConvention.Haeberlen)
        logger.debug(f'Tensor values for {self.name}: {tensor.values}')

        # Calculate isotropic component
        self.isotropy = tensor.isotropy
        logger.info(f'Appending isotropy value for {self.name}: {self.isotropy}')

        # Calculate anisotropy
        self.anisotropy = tensor.anisotropy
        logger.info(f'anisotropy for {self.name}: {self.anisotropy}')

        # Calculate asymmetry
        self.asymmetry = tensor.asymmetry
        logger.info(f'asymmetry for {self.name}: {self.asymmetry}')


class IndirectSpinSpinCoupling(BaseIndirectSpinSpinCoupling):
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
    These contributions are captured in their own classes.
    """

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


class IndirectSpinSpinCouplingFermiContact(BaseIndirectSpinSpinCoupling):
    """
    Represents the Fermi contact contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_fc' tag in the magres data block
    of a .magres file.
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


class IndirectSpinSpinCouplingOrbitalDiamagnetic(BaseIndirectSpinSpinCoupling):
    """
    Represents the orbital diamagnetic contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_orbital_d' tag in the magres data block
    of a .magres file.
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


class IndirectSpinSpinCouplingOrbitalParamagnetic(BaseIndirectSpinSpinCoupling):
    """
    Represents the orbital paramagnetic contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_orbital_p' tag in the magres data block
    of a .magres file.
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


class IndirectSpinSpinCouplingSpinDipolar(BaseIndirectSpinSpinCoupling):
    """
    Represents the spin dipolar contribution to the indirect spin-spin coupling.
    This contribution is identified by the 'isc_spin' tag in the magres data block
    of a .magres file.
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

    value_vgv_approx = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='m ** 3 / mol',
        description="""
        Approximate magnetic susceptibility tensor (vGv approximation). This tensor
        is typically provided by DFT codes such as VASP and represents an approximate
        calculation of the magnetic susceptibility.
        """,
    )

    value_pgv_approx = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit='m ** 3 / mol',
        description="""
        Approximate magnetic susceptibility tensor (pGv approximation). This tensor
        is typically provided by DFT codes such as VASP and represents an approximate
        calculation of the magnetic susceptibility.
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

        # Log information about the tensors if they are present
        if hasattr(self, 'value') and self.value is not None:
            logger.info(f'Magnetic susceptibility tensor for {self.name}: {self.value}')

        if hasattr(self, 'value_vgv_approx') and self.value_vgv_approx is not None:
            logger.info(
                f'Approximate magnetic susceptibility tensor (vGv) for {self.name}: '
                f'{self.value_vgv_approx}'
            )

        if hasattr(self, 'value_pgv_approx') and self.value_pgv_approx is not None:
            logger.info(
                f'Approximate magnetic susceptibility tensor (pGv) for {self.name}: '
                f'{self.value_pgv_approx}'
            )


class Outputs(BaseOutputs):
    """
    The outputs of the principal metadata for NMR.
    """

    magnetic_shieldings = SubSection(sub_section=MagneticShielding.m_def, repeats=True)
    electric_field_gradients = SubSection(
        sub_section=ElectricFieldGradient.m_def, repeats=True
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
