import awkward as ak
import numpy as np

TAU_MASS = 1.777  # GeV
CM_ENERGY = 91.2 # GeV

def post_calibrate_tau_tau(tau_a: ak.Array, tau_b: ak.Array) -> ak.Array:
    # energy = (tau_a + tau_b).mass / 2
    energy = CM_ENERGY / 2
    mass = TAU_MASS
    p = (energy*energy - mass*mass)**0.5

    # reconstruct pt

    pt_a = p / np.cosh(tau_a.eta)
    pt_b = p / np.cosh(tau_b.eta)

    px_a = pt_a * np.cos(tau_a.phi)
    py_a = pt_a * np.sin(tau_a.phi)
    pz_a = pt_a * np.sinh(tau_a.eta)
    px_b = pt_b * np.cos(tau_b.phi)
    py_b = pt_b * np.sin(tau_b.phi)
    pz_b = pt_b * np.sinh(tau_b.eta)

    px_shift = (px_a + px_b)
    py_shift = (py_a + py_b)
    pz_shift = (pz_a + pz_b)

    px_a_corr = px_a - (px_shift/2)
    py_a_corr = py_a - (py_shift / 2)
    pz_a_corr = pz_a - (pz_shift/2)

    # use corrected direction of tau_a only
    norm_a_corr = np.sqrt(px_a_corr ** 2 + py_a_corr ** 2 + pz_a_corr ** 2)

    ux_a = px_a_corr / norm_a_corr
    uy_a = py_a_corr / norm_a_corr
    uz_a = pz_a_corr / norm_a_corr

    # rebuild with fixed |p|
    px_a_final = p * ux_a
    py_a_final = p * uy_a
    pz_a_final = p * uz_a

    # force tau_b to be exactly opposite
    px_b_final = -px_a_final
    py_b_final = -py_a_final
    pz_b_final = -pz_a_final

    tau_a_final = ak.zip(
        {
            "px": px_a_final,
            "py": py_a_final,
            "pz": pz_a_final,
            "energy": np.full_like(px_a_final, energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )

    tau_b_final = ak.zip(
        {
            "px": px_b_final,
            "py": py_b_final,
            "pz": pz_b_final,
            "energy": np.full_like(px_b_final, energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )
    return tau_a_final, tau_b_final