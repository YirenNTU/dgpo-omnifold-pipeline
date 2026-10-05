import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_film_strength import FilmProbe, noise_for, summarize
from evenet.network.layers.transformer import GeneratorTransformerBlockModule


def fixture():
    torch.manual_seed(19)
    block = GeneratorTransformerBlockModule(8, 2, 0., True, .3, 0.).eval()
    head = SimpleNamespace(gen_transformer_blocks=[block])
    x, c = torch.randn(3, 4, 8), torch.randn(3, 4, 8)
    mask = torch.ones(3, 4, 1)
    selected = torch.zeros_like(mask, dtype=torch.bool); selected[:, -2:] = True
    parts = tuple(torch.randn(3, 8)*.2 for _ in range(4))
    def run(parts_=parts):
        return block(x, c, mask, modulation=parts_, modulation_mask=selected)[1]
    return head, block, run, parts


@pytest.mark.parametrize('arm,indices', [('no_scale', [0,2]), ('no_shift',[1,3]), ('no_film',[0,1,2,3]), ('no_block_0',[0,1,2,3])])
def test_intervention_matches_explicit_forward_and_restores(arm, indices):
    head, block, run, parts = fixture()
    with torch.inference_mode():
        original = run()
        expected_parts = tuple(torch.zeros_like(v) if i in indices else v for i,v in enumerate(parts))
        expected = run(expected_parts)
        with FilmProbe(head, arm):
            actual = run()
        assert torch.equal(actual, expected)
        assert not torch.equal(actual, original)
        assert torch.equal(run(), original)
        assert not block._forward_pre_hooks


def test_recording_noop_correct_scale_and_layerscale():
    head, block, run, parts = fixture()
    with torch.inference_mode():
        baseline = run()
        with FilmProbe(head, record=True) as probe:
            assert torch.equal(run(), baseline)
        assert torch.allclose(probe.values['block0/attn_scale'], parts[0].square().mean(-1)[:,None].expand(-1,2))
        assert torch.allclose(probe.values['block0/attn_post_layerscale'], .09*probe.values['block0/attn_pre_layerscale'], atol=1e-7)
        assert all(v.shape == (3,2) for v in probe.values.values())
        assert summarize(probe.values)


def test_hooks_removed_on_failure():
    head, block, run, parts = fixture()
    with pytest.raises(RuntimeError):
        with FilmProbe(head, record=True):
            raise RuntimeError('test')
    assert not block._forward_pre_hooks and not block.norm1._forward_hooks


def test_noise_is_batch_independent():
    whole = noise_for(range(7), 42)
    assert torch.equal(whole[3:], noise_for(range(3,7),42))


def test_dry_run_no_sources_or_output(tmp_path):
    p = subprocess.run([sys.executable, str(Path(__file__).with_name('diagnose_film_strength.py')),
        '--checkpoint','/missing/model.ckpt','--panel','/missing/panel.pt',
        '--output',str(tmp_path/'output'),'--dry-run'], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert 'policy_updates: 0' in p.stdout
    assert not (tmp_path/'output').exists()


def test_sixteen_rank_merge_restores_panel_order_and_rejects_duplicates():
    from diagnose_film_strength import merge_parts
    expected = torch.arange(66).reshape(33,2).float()
    parts = [{'positions': torch.arange(rank,33,16), 'values': {'metric': expected[rank::16]}} for rank in range(16)]
    assert torch.equal(merge_parts(parts,33)['metric'],expected)
    parts[0]['positions'][0] = 1
    with pytest.raises(ValueError, match='duplicate or missing'):
        merge_parts(parts,33)


@pytest.mark.parametrize('gain', [0., .25, .5, 1.])
def test_gain_sweep_matches_explicit_scale_keeps_shift_and_restores(gain):
    head, block, run, parts = fixture()
    weights = {k: v.clone() for k,v in block.state_dict().items()}
    with torch.inference_mode():
        original = run()
        expected = run(tuple(v*gain if i in (0,2) else v for i,v in enumerate(parts)))
        with FilmProbe(head, scale_gain=gain):
            actual = run()
        assert torch.equal(actual, expected)
        assert torch.equal(run(),original)
        if gain == 1:
            assert torch.equal(actual,original)
        if gain == 0:
            with FilmProbe(head,'no_scale'):
                assert torch.equal(actual,run())
    assert all(torch.equal(weights[k],v) for k,v in block.state_dict().items())


def test_paired_stats_known_difference_and_identity():
    from diagnose_film_strength import paired_mse_stats
    base = torch.full((32,2),2.)
    stats = paired_mse_stats({'mse/baseline':base, 'mse/gain_0':base-.5, 'mse/gain_1':base},
                            [('gain_0','baseline',0.),('gain_1','baseline',1.)],42)
    assert stats['gain_0']['delta_ci95'] == [-.5,-.5]
    assert stats['gain_0']['relative_mse_change'] == -.25
    assert stats['gain_1']['delta_ci95'] == [0.,0.]


def test_scale_sweep_dry_run(tmp_path):
    import yaml
    p = subprocess.run([sys.executable,str(Path(__file__).with_name('diagnose_film_strength.py')),
        '--checkpoint','/missing/named.ckpt','--output',str(tmp_path/'out'), '--scale-sweep','--dry-run'],
        text=True,capture_output=True)
    assert p.returncode == 0,p.stderr
    cfg = yaml.safe_load(p.stdout)['diagnostic']
    assert cfg['workers']==16 and cfg['gains']==[0.,.25,.5,1.]
    assert cfg['schema']=='film-scale-sweep-v1'
    assert not (tmp_path/'out').exists()
