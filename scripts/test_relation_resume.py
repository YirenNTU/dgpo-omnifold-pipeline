from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_neutrino_backend import read_yaml,read_overlay_yaml,deep_update,latest_training_checkpoint
ROOT=Path(__file__).resolve().parents[1]


def test_resume_preserves_architecture_and_extends_100_epochs():
    base=read_yaml(ROOT/'config/train_diffusion_nersc.yaml')
    old=deep_update(base,read_overlay_yaml(ROOT/'config/train_diffusion_relation_relations.yaml'))
    new=deep_update(read_yaml(ROOT/'config/train_diffusion_nersc.yaml'),read_overlay_yaml(ROOT/'config/train_diffusion_relation_resume.yaml'))
    assert new['network']==old['network'] and new['platform']==old['platform']
    t=new['options']['Training']
    assert t['epochs']-(t['resume_expected_epoch']+1)==100
    assert t['total_epochs']==114 and t['learning_rate_warm_up_factor']==5
    assert t['pretrain_model_load_path'] is None and not t['strict_relation_source']
    coverage = dict(t['JointCoverage'])
    assert coverage.pop('final_completed_epoch') == 114
    assert coverage==old['options']['Training']['JointCoverage']
    assert t['EarlyStopping']['patience']>t['epochs']
    assert new['platform']['number_of_workers']==16


def test_source_must_be_completed_50_not_best_earlier(tmp_path):
    payload=dict(state_dict={'w':torch.ones(1)},optimizer_states=[{'x':1}],lr_schedulers=[{'last_epoch':650}],epoch=12,global_step=169)
    torch.save(payload,tmp_path/'best.ckpt')
    with pytest.raises(ValueError,match='Expected completed epoch 50'):
        latest_training_checkpoint(tmp_path,expected_epoch=49)
    payload.update(epoch=49,global_step=650);torch.save(payload,tmp_path/'last.ckpt')
    assert latest_training_checkpoint(tmp_path,expected_epoch=49).name=='last.ckpt'
