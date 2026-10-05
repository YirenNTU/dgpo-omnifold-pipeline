"""Feature-only policy ablation; reward architecture and FiLM stay intact."""
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.train_neutrino_backend import read_overlay_yaml
from evenet.network.body.angular_conditioning import VisibleAngularFourier
from evenet.network.body.visible_conditioning import VisibleConditioning, visible_conditioning_spec
from evenet.network.body.test_visible_conditioning import spec, inputs, open_heads


def test_pet_fourier_zero_with_identical_parameter_shapes():
    on = VisibleAngularFourier(['Part_eta', 'Part_phi'], 8)
    off = VisibleAngularFourier(['Part_eta', 'Part_phi'], 8, fourier_enabled=False)
    off.load_state_dict(on.state_dict(), strict=True)
    off.projection.weight.data.fill_(1.)
    raw, mask = torch.randn(2, 3, 2), torch.ones(2, 3, 1, dtype=torch.bool)
    assert on.features(raw, mask).count_nonzero() > 0
    assert off.features(raw, mask).count_nonzero() == 0
    assert off(raw, mask).count_nonzero() == 0


def test_film_keeps_numeric_and_tokens_but_zeros_fourier():
    branch = VisibleConditioning(**spec(kinematics=True), fourier_enabled=False)
    baseline = VisibleConditioning(**spec(kinematics=True))
    branch.load_state_dict(baseline.state_dict(), strict=True)
    open_heads(branch)
    raw, tokens, mask = inputs()
    captured = []
    hook = branch.encoder.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    out = branch(raw, tokens, mask, normalized_raw=raw)
    hook.remove()
    assert captured[0][..., 4:20].count_nonzero() == 0
    assert captured[0][..., 21:29].count_nonzero() == 0
    torch.testing.assert_close(captured[0][..., 20:21], torch.where(mask, raw[..., :1], 0.))
    assert captured[0][..., :4].count_nonzero() > 0
    sum(v.square().sum() for layer in out for v in layer).backward()
    assert branch.encoder[0].weight.grad.norm() > 0
    assert branch.spec['fourier_enabled'] is False


def test_overlay_and_classifier_isolation():
    root = Path(__file__).resolve().parents[1]
    base = read_overlay_yaml(root/'config/dgpo_tau_ratio_1110.yaml')
    off = read_overlay_yaml(root/'config/dgpo_tau_ratio_1110_no_fourier.yaml')
    assert off['network']['Body']['PET']['visible_angular_fourier']['fourier_enabled'] is False
    assert off['network']['VisibleConditioning']['diffusion_fourier_enabled'] is False
    assert off['dgpo']['reference_trust'] == base['dgpo']['reference_trust']
    assert off['dgpo']['tau_ratio']['baseline_directory'] == base['dgpo']['tau_ratio']['baseline_directory']
    assert off['dgpo']['tau_ratio']['output_root'] != base['dgpo']['tau_ratio']['output_root']
    cfg = SimpleNamespace(VisibleConditioning=dict(diffusion_enabled=True, classifier_enabled=True,
        diffusion_fourier_enabled=False))
    kwargs = dict(feature_names=['Part_eta','Part_phi'], token_dim=4, hidden_dim=8,
        num_layers=3, n_branches=2)
    assert visible_conditioning_spec(cfg, target='diffusion', **kwargs)['fourier_enabled'] is False
    assert 'fourier_enabled' not in visible_conditioning_spec(cfg, target='classifier', **kwargs)


def test_matched_on_off_configs_differ_only_in_features_and_output_metadata():
    root = Path(__file__).resolve().parents[1]
    on = read_overlay_yaml(root/'config/dgpo_tau_ratio_1110_fourier_on.yaml')
    off = read_overlay_yaml(root/'config/dgpo_tau_ratio_1110_no_fourier.yaml')
    for cfg, enabled in ((on, True), (off, False)):
        assert cfg['network']['Body']['PET']['visible_angular_fourier'].pop('fourier_enabled') is enabled
        assert cfg['network']['VisibleConditioning'].pop('diffusion_fourier_enabled') is enabled
        assert cfg['platform']['number_of_workers'] == 16
    for path in (
        ('options', 'Training', 'model_checkpoint_save_path'),
        ('dgpo', 'tau_ratio', 'output_root'),
        ('logger', 'local', 'save_dir'),
        ('logger', 'wandb', 'run_name'),
        ('logger', 'wandb', 'tags'),
        ('nersc', 'ray', 'results_dir'),
        ('nersc', 'execution', 'command'),
    ):
        a, b = on, off
        for key in path[:-1]:
            a, b = a[key], b[key]
        assert a.pop(path[-1]) != b.pop(path[-1])
    assert on == off
