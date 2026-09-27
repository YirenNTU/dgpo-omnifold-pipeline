import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_output_scaling as launcher
from test_h4_output_scaling import runtime_guard
from RL.DGPO_neutrino.omnifold_ztautau.ratio_health import report
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


def test_equal_distributions():
    x = torch.linspace(-.9,.9,100)[:,None].repeat(1,3)
    result = report(torch.zeros(100),torch.zeros(100),x,x)
    assert result['test_bce'] == pytest.approx(.69314718056)
    assert result['raw/ess_fraction'] == pytest.approx(1.)
    assert result['raw/log_mean_ratio'] == pytest.approx(0.)
    assert result['raw/top1pct_mass'] == pytest.approx(.01)
    assert result['raw/joint_delta_phi_opening_jsd'] == pytest.approx(0., abs=1e-14)


def test_exact_ratio_improves_joint():
    a = torch.tensor([[-.5]*3]*75+[[.5]*3]*25)
    b = torch.tensor([[-.5]*3]*25+[[.5]*3]*75)
    g = torch.tensor([3.]*25+[1/3]*75).log()
    t = torch.tensor([3.]*75+[1/3]*25).log()
    result = report(t,g,a,b)
    assert abs(result['raw/log_mean_ratio']) < 1e-7
    assert result['raw/joint_delta_phi_opening_jsd'] < 1e-12
    assert result['raw/joint_delta_phi_opening_jsd_change'] < 0


def test_extreme_finite_logits_stay_finite():
    x = torch.zeros(100,3)
    g = torch.zeros(100); g[0] = 1000
    result = report(torch.zeros(100),g,x,x)
    assert result['raw/ess_fraction'] == pytest.approx(.01)
    assert result['raw/top1pct_mass'] == pytest.approx(1.)
    assert result['raw/log_mean_ratio'] > 990


def test_contract_and_retry():
    cfg = launcher.validated_config('ratio-health')
    runtime_guard(cfg)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'],n_train=20000,n_validation=4000)
    fit.validate()
    assert fit.validation_min_delta == 1e-3
    assert fit.min_steps == 1000 and fit.steps == 3000
    assert fit.restore_best and fit.checkpoint_selection_metric == 'loss'
    assert cfg['dgpo']['adaptive_omnifold']['audit_fit']['validation_patience_epochs'] == 10
    assert launcher.validated_config('standardized-long')['dgpo']['adaptive_omnifold']['audit_fit']['validation_min_delta'] == 1e-4
    assert fit.ratio_audit_export_dir.endswith('ratio-health/ratio_audit')
    assert cfg['platform']['number_of_workers'] == 16
    assert cfg['dgpo']['adaptive_omnifold']['audit_fit']['disjoint_final_audit'] is True
    assert cfg['logger']['wandb']['id'] == 'h4ratio1'
    retry = launcher.retry_config(cfg,'ratio-health','retry1')
    assert retry['dgpo']['adaptive_omnifold']['audit_fit']['ratio_audit_export_dir'].endswith('ratio-health-retry1/ratio_audit')


def test_nonfinite_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        report(torch.zeros(2),torch.tensor([0.,float('nan')]),torch.zeros(2,3),torch.zeros(2,3))


def test_live_logger_accepts_health_metrics():
    source = (launcher.ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text()
    block=source.split('def _log_omnifold_fit_progress(',1)[1].split('_log.info(',1)[0]
    assert '"ratio_health/"' in block


def _export_worker(rank, root):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+root+'/rendezvous',rank=rank,world_size=2)
    try:
        with pytest.MonkeyPatch.context() as patch:
            test_restored_model_export_and_independent_split(Path(root)/'shared',patch)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_export(tmp_path):
    torch.multiprocessing.spawn(_export_worker,args=(str(tmp_path),),nprocs=2,join=True)


def test_restored_model_export_and_independent_split(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio as er, ratio_fit as rf
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.packing_spec = SimpleNamespace(to_dict=lambda: {'test':True})
        def forward(self,c,z):
            return z.sum(-1)*self.weight
    def fitted(model,*args,**kwargs):
        with torch.no_grad(): model.weight.fill_(.3)
        return rf.RatioFitDiagnostics(.4,.8,10,steps_completed=10,best_step=7)
    monkeypatch.setattr(rf,'fit_density_ratio',fitted)
    monkeypatch.setattr(er,'periodic_tau_pair_features',lambda c,z,s: z[...,:3].tanh())
    destination = tmp_path/'audit'
    cfg = rf.RatioFitConfig(ratio_audit_export_dir=str(destination))
    g = torch.Generator().manual_seed(23)
    c = torch.randn(100,4,generator=g)
    t = torch.randn(100,4,generator=g)
    x = torch.randn(100,1,4,generator=g)
    rows=[]
    kwargs=dict(model_factory=Tiny,data_condition=c,data_sample=t,gen_condition=c,
                gen_sample=x,gen_weight=torch.ones(100,1),fit_config=cfg,seed=42,
                identity_split_seed=42,progress_callback=rows.append)
    er.fit_independent_evenet_audit(**kwargs)
    assert (destination/'COMPLETE').exists()
    bundle=torch.load(destination/'best_classifier_and_test.pt',weights_only=True)
    test=bundle['split_indices']['test']
    assert bundle['fit_diagnostics']['best_step']==7
    assert torch.allclose(bundle['gen_logits'],.3*x[test].sum(-1).reshape(-1))
    assert torch.equal(bundle['model_state']['weight'],torch.tensor(.3))
    assert not set(test.tolist()) & set(bundle['split_indices']['fit'].tolist())
    assert 'ratio_health/test_bce' in rows[-1]
    with pytest.raises(ValueError,match='already exists'):
        er.fit_independent_evenet_audit(**kwargs)
    kwargs['fit_config']=rf.RatioFitConfig(ratio_audit_export_dir=str(tmp_path/'other'))
    with pytest.raises(ValueError,match='independent test'):
        er.fit_independent_evenet_audit(**kwargs,reuse_early_stop_for_audit=True)
