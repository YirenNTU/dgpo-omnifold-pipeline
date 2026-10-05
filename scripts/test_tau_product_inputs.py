"""Only local synthetic tests; remote experiments remain user-submitted."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch

from scripts.test_tau_explicit_inputs import fixture
from scripts.test_tau_conditioning import model_config
from scripts.tau_explicit_inputs import prepare_arrays, analyzer_products, transform_panel_inputs, validate_spec
from scripts.train_conditional_spin_ratio import build_classifier
from scripts import run_tau_product_inputs as launcher


def test_outer_product_order_sign_and_candidate_only_panel_parity():
    x=np.array([[.2,-.3,.4,-.5,.6,-.7]])
    np.testing.assert_allclose(analyzer_products(x),[[-.10,.12,-.14,.15,-.18,.21,-.20,.24,-.28]],rtol=1e-7)
    data,va,vb,gt,gg,delta=fixture()
    out,spec=prepare_arrays(data,va,vb,gt,gg,'products')
    geom,_=prepare_arrays(data,va,vb,gt,gg,'geometry')
    for key,value in out.items():
        without=np.concatenate((value[:,:-30],value[:,-21:]),1) if key.startswith('candidate_') else value
        np.testing.assert_array_equal(without,geom[key])
    np.testing.assert_array_equal(out['candidate_truth'][:,-30:-21],analyzer_products(gt))
    np.testing.assert_array_equal(out['candidate_generated'][:,-30:-21],analyzer_products(gg))
    c,f=transform_panel_inputs(data['condition'],data['candidate_truth'],va,vb,delta,spec)
    np.testing.assert_array_equal(c,out['condition']);np.testing.assert_array_equal(f,out['candidate_truth'])
    # Arbitrarily changing truth geometry must never alter a generated input.
    changed,_=prepare_arrays(data,va,vb,gt[:,::-1],gg,'products')
    np.testing.assert_array_equal(changed['candidate_generated'],out['candidate_generated'])
    bad=copy.deepcopy(spec);bad['product_fields'].reverse()
    with pytest.raises(ValueError):validate_spec(bad)


def test_initial_parameters_rng_and_function_match_geometry_exactly():
    data,va,vb,gt,gg,_=fixture()
    outputs=[];models=[];rng=[]
    for arm in ['geometry','products']:
        out,spec=prepare_arrays(data,va,vb,gt,gg,arm)
        cfg=dict(model_config('film'),condition_dim=15,candidate_dim=out['candidate_truth'].shape[1],
            explicit_input=spec,condition_width=256,condition_hidden=256,ratio_bound=30)
        torch.manual_seed(42);m=build_classifier(cfg).eval();rng.append(torch.get_rng_state())
        outputs.append(m(torch.from_numpy(out['condition']),torch.from_numpy(out['candidate_truth'])))
        models.append(m)
    assert torch.equal(rng[0],rng[1])
    torch.testing.assert_close(outputs[0],outputs[1],atol=1e-7,rtol=1e-6)
    for key,value in models[0].state_dict().items():
        new=models[1].state_dict()[key]
        if key=='spin_encoder.0.weight':new=torch.cat((new[:,:-30],new[:,-21:]),1)
        torch.testing.assert_close(value,new,rtol=0,atol=0)


def test_config_and_prepare_does_not_start_ray_or_training():
    config=launcher.ROOT/'config/conditional_tau_product_inputs_10pct.yaml'
    settings=launcher.read_settings(config)
    assert settings['workers']==16 and settings['batch_size']==1024 and settings['ratio_bound']==30
    assert settings['geometry_run']=='n3jjqcn7'
    data,va,vb,gt,gg,_=fixture();out,spec=prepare_arrays(data,va,vb,gt,gg,'products')
    settings.update(explicit_input=spec,input_inventory={},explicit_parameter_counts={},
        input_dimensions={'condition':15,'candidate':44},parameter_counts={},
        fresh_runtime='runtime',fresh_generator_checkpoint='raw1110')
    base=dict(model_config('film'),ratio_objective='bce',lr=2e-4,epochs=250)
    cfg=launcher.make_config(settings,base,Path('/base'),Path('/output'),'products')
    assert cfg['additional_candidate_inputs']==9 and cfg['Cij_loss'] is False
    assert cfg['no_control_refits'] and cfg['condition_width']==256 and cfg['ratio_bound']==30
    assert cfg['lr']==base['lr'] and cfg['epochs']==base['epochs']
    with patch('sys.argv',['script',str(config),'prepare']), \
         patch.object(launcher,'prepare',return_value=(None,)*4),patch.object(launcher,'run_arm') as run:
        launcher.main()
    run.assert_not_called()


def test_geometry_control_replay_and_mismatched_inputs_rejected(tmp_path):
    data,va,vb,gt,gg,_=fixture()
    geometry,spec=prepare_arrays(data,va,vb,gt,gg,'geometry')
    settings=launcher.read_settings(launcher.ROOT/'config/conditional_tau_product_inputs_10pct.yaml')
    settings.update(geometry_control=str(tmp_path),parameter_counts={},
        fresh_runtime='runtime',fresh_generator_checkpoint='raw1110')
    base={key:None for key in launcher.PROTOCOL_KEYS}
    base.update(model_config('film'),seed=42,ratio_objective='bce',lr=2e-4,epochs=250,
        diagnostic_every=1,conditioning_diagnostics=True,skip_train_mmd_diagnostic=True)
    local=dict(settings,explicit_input=spec,input_inventory={},explicit_parameter_counts={},
        input_dimensions={'condition':15,'candidate':35})
    cfg=launcher.explicit.make_config(local,base,tmp_path,tmp_path,'geometry')
    (tmp_path/'COMPLETE').write_text('done\n')
    (tmp_path/'wandb.json').write_text(json.dumps({'id':'n3jjqcn7'}))
    (tmp_path/'manifest.json').write_text(json.dumps(cfg))
    np.savez(tmp_path/'prepared.npz',**geometry)
    model=build_classifier(cfg)
    torch.save(dict(cfg,state_dict=model.state_dict()),tmp_path/'best.pt')
    with patch.object(launcher.explicit,'load_saved_scores',return_value=(None,None,{})), \
         patch.object(launcher.explicit.bounded,'verify_panel_report'):
        arrays,manifest=launcher.verify_geometry(settings,base,tmp_path,data,va,vb,gt,gg)
        assert manifest['explicit_input']==spec
        np.testing.assert_array_equal(arrays['candidate_truth'],geometry['candidate_truth'])
        broken=dict(geometry,condition=geometry['condition']+1)
        np.savez(tmp_path/'prepared.npz',**broken)
        with pytest.raises(ValueError,match='paired data/splits'):
            launcher.verify_geometry(settings,base,tmp_path,data,va,vb,gt,gg)
        np.savez(tmp_path/'prepared.npz',**geometry)
        cfg['lr']=.4
        (tmp_path/'manifest.json').write_text(json.dumps(cfg))
        with pytest.raises(ValueError,match='protocol differs: lr'):
            launcher.verify_geometry(settings,base,tmp_path,data,va,vb,gt,gg)


def test_product_report_wandb_tables_and_mass_tv(tmp_path):
    from types import SimpleNamespace
    report=dict(arms={name:{'absolute_component_error':[.1]*9}
        for name in ('bounded','geometry','condition256','unweighted')},
        comparisons={'bounded_minus_'+name:{'absolute_component_error_change_ci95':[[-.1]*9,[.1]*9]}
        for name in ('geometry','condition256')})
    groups={'groups':[dict(arm=name,group='category_11_pt_0',base_mass=1.,weighted_mass=1.,error=.3)
        for name in report['arms']]}
    (tmp_path/'fixed_K64_report.json').write_text(json.dumps(report))
    (tmp_path/'fixed_K64_groups.json').write_text(json.dumps(groups))
    logs=[];run=SimpleNamespace(summary={},log=logs.append)
    with patch.object(launcher.explicit,'finish'),patch('wandb.Table',side_effect=lambda **kwargs:kwargs):
        launcher.finish({'output':str(tmp_path)},None,None,None,None,run,{'geometry_run':'n3jjqcn7'})
    assert run.summary['K64/bounded/group_mass_tv']==0
    assert run.summary['additional_candidate_inputs']==9
    assert len(logs[-1]['K64/product_component_changes']['data'])==18
