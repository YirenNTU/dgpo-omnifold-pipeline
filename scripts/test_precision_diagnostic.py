import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_precision as exp
from test_h4_ddim_coverage import panel_fixture


def test_precision_scope_restores_after_error():
    before = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    for arm in exp.ARMS:
        with pytest.raises(RuntimeError):
            with exp.precision_scope(arm) as settings:
                assert settings['matmul'] == arm
                assert not settings['cudnn_allow_tf32']
                if arm == 'highest':
                    assert not settings['matmul_allow_tf32']
                raise RuntimeError('intentional')
        assert torch.get_float32_matmul_precision() == before
        assert torch.backends.cudnn.allow_tf32 == cudnn


def test_fp64_geometry_matches_canonical():
    from diagnose_h4_ratio_tail import unpack
    from diagnose_h4_ddim_coverage import analyze_arm
    panel = panel_fixture(32)
    fields = unpack(dict(test_condition=panel['condition'], packing_spec=panel['packing_spec']))
    samples = panel['truth'][:, None].repeat(1, 3, 1)
    samples[:, 1, 0] += .00001
    _, truth, canonical = analyze_arm(panel, samples)
    np.testing.assert_allclose(exp.geometry(fields, samples, torch.float64), canonical, atol=1e-12, rtol=0)
    np.testing.assert_allclose(exp.geometry(fields, panel['truth'], torch.float64)[:, 0], truth, atol=1e-12, rtol=0)
    assert exp.geometry(fields, samples, torch.float32).shape == canonical.shape


def test_real_sampler_interface_replays_inputs_without_truth_leakage():
    from evenet.utilities.joint_coverage import event_noise

    class Normalizer:
        def __call__(self, x, mask):
            return x

        def denormalize(self, x, mask, remove_padding):
            return x

    class Model(torch.nn.Module):
        invisible_input_dim = 2
        invisible_normalizer = Normalizer()

        def __init__(self):
            super().__init__()
            self.noises = {arm: [] for arm in exp.ARMS}

        def predict_diffusion_vector(self, noise_x, cond_x, time, mode, noise_mask):
            assert not torch.is_grad_enabled()
            assert not cond_x['x_invisible'].any()
            if (time == 1).all():
                self.noises[torch.get_float32_matmul_precision()].append(noise_x.clone())
            return noise_x*.1

    model = Model().eval()
    noise = event_noise(torch.arange(4), 2, 42)
    truth = torch.randn(4, 2, 2)
    before = noise.clone()
    values, settings = exp.evaluate_batch(model, {}, truth, noise,
        dict(K=2, ddim_steps=3, x0_mode='legacy'))
    for arm in exp.ARMS:
        assert torch.equal(torch.stack(model.noises[arm], dim=1), before)
        assert values[f'samples/{arm}'].shape == (4, 2, 4)
        assert settings[arm]['matmul'] == arm
    assert torch.equal(noise, before)
    assert torch.equal(values['samples/medium'], values['samples/highest'])


def test_complete_analysis_identical_arms_is_not_false_recovery(tmp_path):
    panel = panel_fixture(32)
    values = {}
    for arm in exp.ARMS:
        values[f'samples/{arm}'] = panel['truth'][:, None].repeat(1, 2, 1)
        for t in exp.TIMES:
            values[f'velocity/{arm}/{t}'] = torch.zeros(32, 2, 2)
            values[f'mse/{arm}/{t}'] = torch.ones(32)
    report = exp.analyze(panel, values, dict(bootstrap=20, seed=42))
    for row in report['comparisons'].values():
        assert row['target_coordinate_difference']['max'] == 0
        assert all(r['absolute_truth_gap_change'] == 0 for r in row['coverage'])
    exp.save_json(tmp_path/'report.json', report)
    assert len(json.loads((tmp_path/'report.json').read_text())['arms']) == 3


def test_cli_dry_run_needs_no_remote_files_or_ray(tmp_path):
    output = tmp_path/'output'
    result = subprocess.run([sys.executable, exp.__file__, '--checkpoint', '/missing/model.ckpt',
                             '--output', str(output), '--dry-run'], capture_output=True, text=True, check=True)
    assert 'workers: 16' in result.stdout
    assert 'policy_updates: 0' in result.stdout
    assert not output.exists()


def test_nonfinite_difference_rejected():
    with pytest.raises(ValueError, match='Nonfinite'):
        exp.difference(torch.zeros(1), torch.tensor([float('nan')]))


def test_script_entry_serializes_without_backend_closures(monkeypatch):
    import runpy
    try:
        from ray import cloudpickle
    except ImportError:
        # Same serializer implementation, available in the lightweight local env.
        from joblib.externals import cloudpickle
    # A non-importable script namespace forces by-value serialization, as with
    # the remote CLI. Testing the ordinary imported worker would miss the bug.
    namespace = runpy.run_path(exp.__file__)
    entry = namespace['ray_worker_entry']
    assert not entry.__closure__
    assert 'precision_scope' not in entry.__code__.co_names
    payload = cloudpickle.dumps(entry)
    recovered = cloudpickle.loads(payload)
    monkeypatch.setattr(exp, 'worker', lambda cfg: ('worker-imported', cfg))
    assert recovered({'workers': 16}) == ('worker-imported', {'workers': 16})
    assert len(payload) < 4096  # do not silently embed model/helper graphs
