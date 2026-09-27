"""Contract tests for the matched 10% H4 classifier-design experiment."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_classifier_design_ablation import (  # noqa: E402
    CONFIG,
    FOURIER_CONFIG,
    assert_contract,
)
from train_neutrino_backend import read_overlay_yaml  # noqa: E402


def test_fourier_probe_disables_extensions_and_rejects_reintroduced_shortcut():
    import copy
    import pytest

    cfg = read_overlay_yaml(FOURIER_CONFIG)
    assert_contract(cfg)
    assert cfg["logger"]["wandb"]["id"] != read_overlay_yaml(CONFIG)["logger"]["wandb"]["id"]
    for key in ("topology_direct_logit", "topology_conditioning", "visible_pair_rest_frame"):
        changed = copy.deepcopy(cfg)
        changed["dgpo"]["adaptive_omnifold"]["audit_fit"][key] = True
        with pytest.raises(ValueError):
            assert_contract(changed)


def test_regularized_unfreeze_rejects_wrong_training_settings() -> None:
    import copy
    import pytest

    cfg = read_overlay_yaml(ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_classifier_unfreeze_reg.yaml")
    assert_contract(cfg)
    assert cfg["logger"]["wandb"]["id"] == "h4clfur1"
    for key, wrong in (("train_backbone", False), ("head_dropout", 0.15),
                       ("weight_decay", 5e-4), ("backbone_learning_rate", 2e-4)):
        changed = copy.deepcopy(cfg)
        changed["dgpo"]["adaptive_omnifold"]["audit_fit"][key] = wrong
        with pytest.raises(ValueError):
            assert_contract(changed)


def test_lastblock_probe_rejects_extra_unfreezing_and_wrong_selection() -> None:
    import copy
    import pytest

    cfg = read_overlay_yaml(ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_classifier_lastblock_reg.yaml")
    assert_contract(cfg)
    assert cfg["logger"]["wandb"]["id"] == "h4clflb1"
    for key, wrong in (
        ("train_backbone", True), ("train_last_pet_block", False),
        ("train_grouped_sequential_embedding", True),
        ("train_invisible_projector", True), ("train_encoder", True),
        ("train_layernorm", True), ("checkpoint_selection_metric", "balanced_accuracy"),
    ):
        changed = copy.deepcopy(cfg)
        changed["dgpo"]["adaptive_omnifold"]["audit_fit"][key] = wrong
        with pytest.raises(ValueError):
            assert_contract(changed)


def test_direct_probe_uses_the_finetuned_policy_and_fixed_budget() -> None:
    direct = read_overlay_yaml(CONFIG)
    assert_contract(direct)
    audit = direct["dgpo"]["adaptive_omnifold"]["audit_fit"]
    assert audit["steps"] == audit["min_steps"] == 3000
    assert audit["repeats"] == 1
    assert direct["experiment"]["dataset_fraction"] == 0.10
    assert (
        direct["nersc"]["reproducibility"]["diffusion_pretrain_fraction"]
        == 0.10
    )
    assert direct["nersc"]["reproducibility"]["dgpo_training_fraction"] == 0.10
    training = direct["options"]["Training"]
    assert "dgpo_omnifold_10pct_old_method_hard" in training["model_checkpoint_load_path"]
    assert training["pretrain_model_load_path"] is None
    assert direct["dgpo"]["checkpoint_load_mode"] == "weights_only"
    assert direct["dgpo"]["auto_resume_from_last"] is False
    assert direct["experiment"]["classifier_only"] is True
    assert direct["experiment"]["policy_updates_per_round"] == 0
    assert direct["experiment"]["reward_fit_count"] == 0
    assert direct["experiment"]["classifier_fit_count"] == 1
    assert direct["experiment"]["cold_h4_audit_policy_steps"] == [0]
    assert direct["dgpo"]["adaptive_omnifold"]["baseline_probe_on_start"] is False
    assert (
        direct["dgpo"]["adaptive_omnifold"]["recalibration"][
            "bootstrap_on_start"
        ]
        is False
    )
    assert direct["dgpo"]["gradient_conflict"]["enabled"] is False
    assert audit["topology_direct_logit"] is True
    assert direct["experiment"]["distributed_world_size"] == 16
    assert direct["nersc"]["nodes"] * direct["nersc"]["gpus_per_node"] == 16
    assert direct["nersc"]["execution"]["workers"] == 16
    assert direct["nersc"]["execution"]["gpus_per_worker"] == 1
    assert direct["nersc"]["execution"]["command"].startswith(
        "shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py"
    )
    assert direct["platform"]["number_of_workers"] == 16
    assert direct["platform"]["resources_per_worker"]["GPU"] == 1
    assert audit["batch_size"] % 16 == 0
    assert audit["validation_batch_size"] % 16 == 0
    assert audit["progress_every_n_steps"] == 10
    assert audit["validation_interval_steps"] == 40
    wandb = direct["logger"]["wandb"]
    assert wandb["id"] == "h4clf02"
    assert wandb["id"] != "h4clfdir"
    assert "classifier only" in wandb["run_name"].lower()
    assert wandb["profile"] == "standard"
    assert wandb["classifier_loss_curves"] is True
    assert wandb["simplified"] is True
    assert "ClassifierBottleneckDiagnostics" in wandb["tags"]
    assert "ClassifierOnly" in wandb["tags"]
    assert "NoPolicyUpdate" in wandb["tags"]


def test_classifier_only_terminal_path_precedes_reward_bootstrap() -> None:
    trainer_path = (
        ROOT / "evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py"
    )
    source = trainer_path.read_text()
    classifier_branch = source.index("if classifier_only:")
    reward_bootstrap = source.index(
        "need_initial_omnifold_bootstrap = bool(", classifier_branch
    )
    terminal_source = source[classifier_branch:reward_bootstrap]
    assert classifier_branch < reward_bootstrap
    assert "fit_raw_policy_audit(" in terminal_source
    assert '"classifier_only/reward_fits": 0' in terminal_source
    assert '"classifier_only/policy_updates": 0' in terminal_source
    assert "_finish_wandb_run(wandb_active)" in terminal_source
    assert terminal_source.rstrip().endswith("return")


def test_gradient_diagnostics_separate_the_shortcut_from_context() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "evenet_dgpo"))
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import (
        _classifier_gradient_group,
        _classifier_gradient_norms,
        _classifier_logit_diagnostics,
        _classifier_scale_diagnostics,
    )

    expected = {
        "bank.topology_output.weight": "direct_topology_head",
        "bank.topology_encoder.1.weight": "topology_context",
        "bank.fusion.1.weight": "topology_context",
        "bank.output.weight": "context_output_head",
        "bank.decoder.blocks.0.ffn.0.weight": "decoder",
        "backbone.PET.adapters.0.weight": "adapter",
        "backbone.InvisibleInputProjector.weight": "input_projector",
    }
    assert {name: _classifier_gradient_group(name) for name in expected} == expected

    direct = torch.nn.Parameter(torch.zeros(2))
    context = torch.nn.Parameter(torch.zeros(1))
    direct.grad = torch.tensor([3.0, 4.0])
    context.grad = torch.tensor([12.0])
    norms = _classifier_gradient_norms(
        [
            ("bank.topology_output.weight", direct),
            ("bank.output.weight", context),
        ]
    )
    assert norms == {
        "direct_topology_head": 5.0,
        "context_output_head": 12.0,
    }

    direct.data.copy_(torch.tensor([3.0, 4.0]))
    direct.grad = torch.tensor([0.3, 0.4])
    scales = _classifier_scale_diagnostics(
        [("bank.topology_output.weight", direct)]
    )
    assert scales["parameter_rms_direct_topology_head"] == pytest.approx(
        (12.5) ** 0.5
    )
    assert scales["gradient_rms_direct_topology_head"] == pytest.approx(
        (0.125) ** 0.5
    )
    assert scales[
        "gradient_to_parameter_rms_ratio_direct_topology_head"
    ] == pytest.approx(0.1)

    logit_stats = _classifier_logit_diagnostics(
        torch.tensor([0.0, 20.0, 4.0, 0.0, 0.0, 4.0, 10.0, 2.0, -4.0, 10.0, 2.0])
    )
    assert logit_stats["logit_mean_positive"] == pytest.approx(2.0)
    assert logit_stats["logit_std_positive"] == pytest.approx(1.0)
    assert logit_stats["logit_mean_negative"] == pytest.approx(-2.0)
    assert logit_stats["logit_std_negative"] == pytest.approx(1.0)
    assert logit_stats["logit_class_mean_separation"] == pytest.approx(4.0)
    assert logit_stats["logit_rms"] == pytest.approx(5.0 ** 0.5)
