import copy
import json
from pathlib import Path
import tempfile

import pytest
import torch

from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy.truth_pretrain import atomic_checkpoint,atomic_json


def setup_control(directory):
    settings={'initialization_seed':17,'policy_seed':17,'monitor_seed':23,'panel_seed':93,
              'classifier_seed':93,'audit_features':['both'],'minimum_fit_steps':8000,
              'max_fit_steps':32000,'milestones':[25,1000]}
    spec={'basis':'fourier','modulation':'film','layers':3,'nonlinear':True,'normalization':True,'width':32}
    old={**settings,'conditioning':{'fourier':spec}}
    new={**settings,'conditioning':{'shift':{**spec,'modulation':'shift'}},
         'external_control':{'directory':str(directory),'arm':'fourier','changed_factor':'modulation'},
         'bootstrap_repeats':100}
    report={'state':'completed','valid':True,'config':{'policy_lr':1e-4},'reward':'unchanged','velocity_coefficient':1.}
    atomic_json(directory/'plan.json',old);atomic_json(directory/'report.json',report)
    base={'positive':torch.linspace(-1,1,16),'negative':torch.linspace(0,2,16)}
    scores={'baseline__both_step0':base}
    scores.update({f'fourier__both_step{s}':base for s in settings['milestones']})
    atomic_checkpoint(directory/'test_scores.pt',scores)
    return new,report,scores


def test_control_contract_rejects_extra_changes_and_clock_mismatch():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='control_contract_') as tmp:
        p,r,s=setup_control(Path(tmp));lab.validate_external_control(p)
        for key in ('policy_seed','minimum_fit_steps'):
            bad=copy.deepcopy(p);bad[key]+=1
            with pytest.raises(ValueError):lab.validate_external_control(bad)
        bad=copy.deepcopy(p);bad['conditioning']['shift']['layers']=1
        with pytest.raises(ValueError):lab.validate_external_control(bad)


def test_baseline_replay_must_be_exact_and_reference_unchanged():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='baseline_contract_') as tmp:
        p,r,s=setup_control(Path(tmp));lab.check_external_baseline(p,r,s)
        bad=copy.deepcopy(s);bad['baseline__both_step0']['negative'][0]+=.01
        with pytest.raises(ValueError):lab.check_external_baseline(p,r,bad)
        bad=copy.deepcopy(r);bad['velocity_coefficient']=.5
        with pytest.raises(ValueError):lab.check_external_baseline(p,bad,s)


def test_activation_ablation_preserves_norm_and_modulation():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='activation_contract_') as tmp:
        p,r,s=setup_control(Path(tmp))
        p['conditioning']['shift']['modulation']='film'
        p['conditioning']['shift']['nonlinear']=False
        p['external_control']['changed_factor']='nonlinear'
        lab.validate_external_control(p)
        p['conditioning']['shift']['normalization']=False
        with pytest.raises(ValueError):lab.validate_external_control(p)


def test_norm_ablation_requires_unchanged_activations_and_depth():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='normalization_contract_') as tmp:
        p,r,s=setup_control(Path(tmp))
        p['conditioning']['shift']['modulation']='film'
        p['conditioning']['shift']['normalization']=False
        p['external_control']['changed_factor']='normalization'
        lab.validate_external_control(p)
        p['conditioning']['shift']['nonlinear']=False
        with pytest.raises(ValueError):lab.validate_external_control(p)


def test_depth_ablation_requires_unchanged_norm_and_modulation():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='depth_contract_') as tmp:
        p,r,s=setup_control(Path(tmp))
        p['conditioning']['shift']['modulation']='film'
        p['conditioning']['shift']['layers']=1
        p['external_control']['changed_factor']='layers'
        lab.validate_external_control(p)
        p['conditioning']['shift']['normalization']=False
        with pytest.raises(ValueError):lab.validate_external_control(p)


def test_saved_control_compares_only_after_complete_panel_pairing():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='paired_control_') as tmp:
        directory=Path(tmp);p,r,old=setup_control(directory)
        output=directory/'new';output.mkdir()
        panel={split:{'c':torch.zeros(16,1),'positive':torch.ones(16,3),'negative':torch.zeros(16,3)} for split in ('train','validation','test')}
        scores={'baseline__both_step0':old['baseline__both_step0']}
        for step in p['milestones']:
            atomic_checkpoint(directory/f'fourier_step{step}_panels.pt',panel)
            atomic_checkpoint(output/f'shift_step{step}_panels.pt',panel)
            scores[f'shift__both_step{step}']=old[f'fourier__both_step{step}']
        result=lab.compare_external_control(p,output,r,scores)
        assert result['all_panels_paired'] and result['baseline_predictions_exact']
        assert len(result['contrasts'])==2
        for v in result['contrasts'].values():assert v['bce']['delta']==0.
        panel['test']['positive'][0,0]=4.
        atomic_checkpoint(output/'shift_step1000_panels.pt',panel)
        with pytest.raises(ValueError):lab.compare_external_control(p,output,r,scores)
