# Expected value for Magnetic Susceptibility tensor
EXPECTED_MAGNETIC_SUSCEPTIBILITY_VALUE = [
    [-16.0397, 0.0000, 0.0000],
    [0.0000, -16.1160, 0.0000],
    [0.0000, 0.0000, -48.9908],
]

# Expected values for Magnetic Shielding tensor
EXPECTED_SHIELDING_VALUE = [
    [25.6559, -0.0000, -0.0000],
    [-0.0000, -74.2956, 0.0000],
    [0.0000, -0.0000, 167.7902],
]

# Add expected derived values from Magnetic Shielding tensor
EXPECTED_SHIELDING_DERIVED = {
    'isotropy': 39.7168,
    'anisotropy': 192.11,
    'reduced_anisotropy': 128.07,
    'asymmetry': 0.7804,
    'span': 242.09,
    'skew': 0.1742,
}

# Expected values for Electric Field Gradient tensor
EXPECTED_GRADIENT_VALUE = [
    [0.12793404309, 0.0514298737569, 0.20226839328],
    [0.0514298737569, -0.133531745662, 0.0414560149276],
    [0.20226839328, 0.0414560149276, 0.00559770257191],
]

# Add expected derived values from Electric Field Gradient tensor
EXPECTED_GRADIENT_DERIVED = {
    'Vzz': 0.2884,
    'asymmetry': 0.0182,
}

# Expected values for Indirect Spin-Spin Coupling tensor
EXPECTED_ISC_VALUE = [
    [30.6826678444, -1.62975719737, -5.48015039636],
    [-1.57659364137, 37.4487199089, -1.18603354744],
    [-5.44203254428, -1.20427495723, 33.5914667613],
]

# Add expected derived values from Indirect Spin-Spin Coupling tensor
EXPECTED_ISC_DERIVED = {
    'isotropy': 33.9076,
    'anisotropy': -11.6629,
    'reduced_anisotropy': -7.7753,
    'asymmetry': 0.0076,
    'span': 11.6925,
}

EXPECTED_HYPERFINE_DIPOLAR_VALUE = [
    [57.6604, -0.0020, -0.0000],
    [-0.0020, 59.4429, 0.0010],
    [-0.0000, 0.0010, -117.1033],
]

EXPECTED_HYPERFINE_FERMI_CONTACT_VALUE = [0.140601]

EXPECTED_UNPAIRED_SPINS_VALUE = [0]

EXPECTED_DELTA_G_VALUE = [
    [12351.0700, 0.0000, -0.0000],
    [0.0100, 4383.3800, 0.0100],
    [-0.0000, 0.0000, -143.3700],
]

EXPECTED_DELTA_G_PARATEC_VALUE = [
    [12639.98, 0.0, -0.0],
    [0.01, 4508.84, 0.01],
    [-0.0, 0.0, -128.9],
]
