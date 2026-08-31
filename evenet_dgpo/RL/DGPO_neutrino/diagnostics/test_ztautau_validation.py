from __future__ import annotations

import numpy as np
import torch

from RL.DGPO_neutrino.diagnostics.ztautau_validation import (
    build_ztautau_validation_metrics,
    collect_ztautau_validation_arrays,
)


def _batch(batch_size: int) -> dict[str, torch.Tensor]:
    zeros = torch.zeros(batch_size)
    ones = torch.ones(batch_size)
    return {
        "x_invisible": torch.stack(
            (
                torch.stack((0.05 * ones, 0.02 * ones), dim=-1),
                torch.stack((-0.04 * ones, -0.03 * ones), dim=-1),
            ),
            dim=1,
        ),
        "x_invisible_mask": torch.ones(batch_size, 2),
        "lead_a_visible_px": ones,
        "lead_a_visible_py": zeros,
        "lead_a_visible_pz": zeros,
        "lead_b_visible_px": -ones,
        "lead_b_visible_py": zeros,
        "lead_b_visible_pz": zeros,
    }


def test_candidate_zero_drives_1d_panels_while_tarp_keeps_all_k() -> None:
    batch_size, candidates_per_event = 5, 3
    batch = _batch(batch_size)
    candidates = batch["x_invisible"].unsqueeze(0).repeat(candidates_per_event, 1, 1, 1)
    candidates[0, :, 0, 0] = 0.25
    candidates[1, :, 0, 0] = 0.75
    reference = batch["x_invisible"].unsqueeze(0)
    arrays = collect_ztautau_validation_arrays(
        candidates,
        reference,
        batch,
        torch.ones(batch_size),
    )
    assert arrays["_tarp_truth"].shape == (batch_size, 4)
    assert arrays["_tarp_candidates"].shape == (
        batch_size,
        candidates_per_event,
        4,
    )
    np.testing.assert_allclose(
        arrays["target/tau_a_delta_theta/current"],
        np.full(batch_size, 0.25),
    )


def test_tarp_and_targeted_physics_metrics_are_emitted() -> None:
    rng = np.random.default_rng(17)
    n_events, k = 120, 4
    truth = rng.normal(size=(n_events, 4))
    candidates = truth[:, None, :] + rng.normal(scale=0.4, size=(n_events, k, 4))
    arrays = {
        "_tarp_truth": truth,
        "_tarp_candidates": candidates,
        "_tarp_profile": np.linspace(0.0, 1.0, n_events),
        "target/tau_a_delta_theta/truth": truth[:, 0],
        "target/tau_a_delta_theta/current": candidates[:, 0, 0],
        "target/tau_a_delta_theta/ref": candidates[:, 1, 0],
    }
    metrics = build_ztautau_validation_metrics(
        arrays,
        val_k=k,
        tarp_config={
            "enabled": True,
            "n_bins": 2,
            "min_events": 20,
            "arms": ["full", "rank_copula"],
            "n_null_assignments": 19,
            "n_reference_trials": 1,
            "seed": 9,
            "alpha": 0.05,
            "pooled_panel": True,
        },
        metrics_config={"enabled": True, "bins": 20},
        include_images=False,
    )
    assert "val_tarp/tarp_binned_min_holm_pvalue" in metrics
    assert "val_tarp/pooled/tarp_min_holm_pvalue" in metrics
    assert "val_ztautau/jsd/current/target/tau_a_delta_theta" in metrics
    assert metrics["val_tarp/geometry/candidates"] == float(k)


def test_tarp_reports_insufficient_candidate_pool() -> None:
    metrics = build_ztautau_validation_metrics(
        {},
        val_k=1,
        tarp_config={"enabled": True},
        metrics_config={"enabled": True},
        include_images=False,
    )
    assert metrics["val_tarp/skipped_insufficient_k"] == 1.0


def test_nonzero_candidate_index_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "candidate_index must be 0"):
        build_ztautau_validation_metrics(
            {},
            val_k=2,
            tarp_config={"enabled": False},
            metrics_config={"enabled": True, "candidate_index": 1},
            include_images=False,
        )


def test_metrics_are_explicitly_opt_in() -> None:
    assert build_ztautau_validation_metrics(
        {},
        val_k=16,
        tarp_config={},
        metrics_config={},
        include_images=False,
    ) == {}
