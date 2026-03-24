from __future__ import annotations

import awkward as ak
import numpy as np
import vector


PHOTON_DR_MAX = 0.3
FOUR_VECTOR_FEATURES = ["energy", "pt", "eta", "phi"]
vector.register_awkward()


def build_momentum4d(px, py, pz, energy):
    return ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        },
        with_name="Momentum4D",
    )


def compute_pt_eta_phi(px: ak.Array, py: ak.Array, pz: ak.Array) -> tuple[ak.Array, ak.Array, ak.Array]:
    p4 = build_momentum4d(px, py, pz, np.sqrt(px * px + py * py + pz * pz))
    eta = ak.where(np.isfinite(p4.eta), p4.eta, 0)
    return p4.pt, eta, p4.phi


def p4_to_features(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    px = np.asarray(px, dtype=np.float32)
    py = np.asarray(py, dtype=np.float32)
    pz = np.asarray(pz, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32)
    p4 = vector.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        }
    )
    energy_values = np.asarray(p4.E, dtype=np.float32)
    pt_values = np.asarray(p4.pt, dtype=np.float32)
    eta_values = np.asarray(p4.eta, dtype=np.float32)
    phi_values = np.asarray(p4.phi, dtype=np.float32)
    eta_values = np.where(np.isfinite(eta_values), eta_values, 0.0)
    return np.stack([energy_values, pt_values, eta_values, phi_values], axis=1).astype(np.float32)


def sum_masked_p4(events: ak.Array, mask: ak.Array):
    part_p4 = build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"],
        events["Part_fourMomentum_fCoordinates_fY"],
        events["Part_fourMomentum_fCoordinates_fZ"],
        events["Part_fourMomentum_fCoordinates_fT"],
    )
    return ak.sum(part_p4[mask], axis=1)


def map_hemisphere_to_tau_sign(
    first_values: np.ndarray,
    second_values: np.ndarray,
    first_charge: np.ndarray,
    second_charge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_is_tau_minus = (first_charge < 0) & (second_charge > 0)
    second_is_tau_minus = (second_charge < 0) & (first_charge > 0)
    tau_minus_mask = first_is_tau_minus | second_is_tau_minus
    tau_plus_mask = tau_minus_mask

    tau_minus = np.where(first_is_tau_minus[:, None], first_values, 0.0) + np.where(second_is_tau_minus[:, None], second_values, 0.0)
    tau_plus = np.where(first_is_tau_minus[:, None], second_values, 0.0) + np.where(second_is_tau_minus[:, None], first_values, 0.0)
    return tau_minus.astype(np.float32), tau_plus.astype(np.float32), tau_minus_mask, tau_plus_mask


def build_visible_tau_assumptions(events: ak.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    charge = events["Part_charge"]
    hemisphere = events["Part_hemisphere"]
    pdg_id = abs(events["Part_pdgId"])
    part_p4 = build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"],
        events["Part_fourMomentum_fCoordinates_fY"],
        events["Part_fourMomentum_fCoordinates_fZ"],
        events["Part_fourMomentum_fCoordinates_fT"],
    )
    eta = ak.where(np.isfinite(part_p4.eta), part_p4.eta, 0)
    phi = part_p4.phi

    hemisphere_masks = {
        "a": hemisphere == 1,
        "b": hemisphere == -1,
    }

    prong_features = {}
    prong_charge_sums = {}
    rho_features = {}

    for hemisphere_name, hemisphere_mask in hemisphere_masks.items():
        prong_mask = hemisphere_mask & (charge != 0)
        photon_mask = hemisphere_mask & (charge == 0) & (pdg_id == 21)

        prong_p4 = sum_masked_p4(events, prong_mask)
        prong_features[hemisphere_name] = p4_to_features(
            ak.to_numpy(prong_p4.px, allow_missing=False).astype(np.float32),
            ak.to_numpy(prong_p4.py, allow_missing=False).astype(np.float32),
            ak.to_numpy(prong_p4.pz, allow_missing=False).astype(np.float32),
            ak.to_numpy(prong_p4.E, allow_missing=False).astype(np.float32),
        )
        prong_charge_sums[hemisphere_name] = ak.to_numpy(ak.sum(charge[prong_mask], axis=1), allow_missing=False).astype(np.float32)

        prong_eta = eta[prong_mask]
        prong_phi = phi[prong_mask]
        photon_eta = eta[photon_mask]
        photon_phi = phi[photon_mask]

        pairs = ak.cartesian(
            {
                "photon_eta": photon_eta,
                "photon_phi": photon_phi,
                "prong_eta": prong_eta,
                "prong_phi": prong_phi,
            },
            axis=1,
            nested=True,
        )
        delta_eta = pairs["photon_eta"] - pairs["prong_eta"]
        delta_phi = np.arctan2(
            np.sin(pairs["photon_phi"] - pairs["prong_phi"]),
            np.cos(pairs["photon_phi"] - pairs["prong_phi"]),
        )
        delta_r = np.sqrt(delta_eta * delta_eta + delta_phi * delta_phi)
        photon_near_prong = ak.fill_none(ak.any(delta_r < PHOTON_DR_MAX, axis=-1), False)

        photon_p4 = ak.sum(part_p4[photon_mask][photon_near_prong], axis=1)
        rho_p4 = prong_p4 + photon_p4
        rho_features[hemisphere_name] = p4_to_features(
            ak.to_numpy(rho_p4.px, allow_missing=False).astype(np.float32),
            ak.to_numpy(rho_p4.py, allow_missing=False).astype(np.float32),
            ak.to_numpy(rho_p4.pz, allow_missing=False).astype(np.float32),
            ak.to_numpy(rho_p4.E, allow_missing=False).astype(np.float32),
        )

    prong_tau_minus, prong_tau_plus, prong_tau_minus_mask, prong_tau_plus_mask = map_hemisphere_to_tau_sign(
        prong_features["a"],
        prong_features["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )
    rho_tau_minus, rho_tau_plus, rho_tau_minus_mask, rho_tau_plus_mask = map_hemisphere_to_tau_sign(
        rho_features["a"],
        rho_features["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )

    tau_vis_prong = np.stack([prong_tau_minus, prong_tau_plus], axis=1).astype(np.float32)
    tau_vis_prong_mask = np.stack([prong_tau_minus_mask, prong_tau_plus_mask], axis=1)
    tau_vis_rho = np.stack([rho_tau_minus, rho_tau_plus], axis=1).astype(np.float32)
    tau_vis_rho_mask = np.stack([rho_tau_minus_mask, rho_tau_plus_mask], axis=1)
    return tau_vis_prong, tau_vis_prong_mask, tau_vis_rho, tau_vis_rho_mask


def truth_feature(values: ak.Array | None) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float32)
    return ak.to_numpy(ak.fill_none(values, np.nan), allow_missing=False).astype(np.float32)


def build_tau_targets(
    events: ak.Array,
    tau_vis_prong: np.ndarray,
    tau_vis_prong_mask: np.ndarray,
    tau_vis_rho: np.ndarray,
    tau_vis_rho_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_events = len(events)
    x_invisible = np.zeros((n_events, 2, 4), dtype=np.float32)
    x_invisible_mask = np.zeros((n_events, 2), dtype=bool)
    num_invisible_raw = np.zeros(n_events, dtype=np.int64)
    num_invisible_valid = np.zeros(n_events, dtype=np.int64)
    tau_vis_target = np.zeros((n_events, 2, 4), dtype=np.float32)
    tau_vis_target_mask = np.zeros((n_events, 2), dtype=bool)

    required_fields = [
        "truth_tau_px",
        "truth_tau_py",
        "truth_tau_pz",
        "truth_tau_E",
        "truth_anti_tau_px",
        "truth_anti_tau_py",
        "truth_anti_tau_pz",
        "truth_anti_tau_E",
        "event_category",
    ]
    if not all(field in events.fields for field in required_fields):
        return x_invisible, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target, tau_vis_target_mask

    tau_px = truth_feature(events["truth_tau_px"])
    tau_py = truth_feature(events["truth_tau_py"])
    tau_pz = truth_feature(events["truth_tau_pz"])
    tau_E = truth_feature(events["truth_tau_E"])
    anti_tau_px = truth_feature(events["truth_anti_tau_px"])
    anti_tau_py = truth_feature(events["truth_anti_tau_py"])
    anti_tau_pz = truth_feature(events["truth_anti_tau_pz"])
    anti_tau_E = truth_feature(events["truth_anti_tau_E"])

    truth_tau = np.stack([tau_E, tau_px, tau_py, tau_pz], axis=1).astype(np.float32)
    truth_anti_tau = np.stack([anti_tau_E, anti_tau_px, anti_tau_py, anti_tau_pz], axis=1).astype(np.float32)

    event_category = ak.to_numpy(events["event_category"], allow_missing=False).astype(np.int64)
    tau_minus_category = event_category % 10
    tau_plus_category = event_category // 10

    tau_minus_use_rho = tau_minus_category == 2
    tau_plus_use_rho = tau_plus_category == 2

    tau_vis_target[:, 0, :] = np.where(tau_minus_use_rho[:, None], tau_vis_rho[:, 0, :], tau_vis_prong[:, 0, :])
    tau_vis_target[:, 1, :] = np.where(tau_plus_use_rho[:, None], tau_vis_rho[:, 1, :], tau_vis_prong[:, 1, :])
    tau_vis_target_mask[:, 0] = np.where(tau_minus_use_rho, tau_vis_rho_mask[:, 0], tau_vis_prong_mask[:, 0])
    tau_vis_target_mask[:, 1] = np.where(tau_plus_use_rho, tau_vis_rho_mask[:, 1], tau_vis_prong_mask[:, 1])

    target_tau_minus = truth_tau - tau_vis_target[:, 0, :]
    target_tau_plus = truth_anti_tau - tau_vis_target[:, 1, :]

    x_invisible[:, 0, :] = p4_to_features(
        target_tau_minus[:, 1],
        target_tau_minus[:, 2],
        target_tau_minus[:, 3],
        target_tau_minus[:, 0],
    )
    x_invisible[:, 1, :] = p4_to_features(
        target_tau_plus[:, 1],
        target_tau_plus[:, 2],
        target_tau_plus[:, 3],
        target_tau_plus[:, 0],
    )

    tau_truth_valid = np.isfinite(truth_tau).all(axis=1)
    anti_tau_truth_valid = np.isfinite(truth_anti_tau).all(axis=1)
    x_invisible_mask[:, 0] = tau_truth_valid & tau_vis_target_mask[:, 0] & np.isfinite(x_invisible[:, 0, :]).all(axis=1)
    x_invisible_mask[:, 1] = anti_tau_truth_valid & tau_vis_target_mask[:, 1] & np.isfinite(x_invisible[:, 1, :]).all(axis=1)
    x_invisible[~x_invisible_mask] = 0.0
    tau_vis_target[~tau_vis_target_mask] = 0.0

    num_invisible_raw[:] = 2
    num_invisible_valid[:] = x_invisible_mask.sum(axis=1).astype(np.int64)
    return x_invisible, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target, tau_vis_target_mask
