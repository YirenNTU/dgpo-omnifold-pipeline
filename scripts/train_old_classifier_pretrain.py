#!/usr/bin/env python3
"""Refit the weak classifier used by successful c4a91e07 on 10% pretrain."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_neutrino_backend import read_overlay_yaml
from train_h4_pair_token import validated_config as pair_config
from train_h4_classifier_representation import CLEAN, validate_manifest
SOURCE='/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt'
OUTPUT='/pscratch/sd/y/yiren/Ztautau/old_classifier_pretrain'
CONFIG=ROOT/'config/dgpo_old_classifier_pretrain.yaml'


def validated_config():
    cfg=read_overlay_yaml(CONFIG);parent=pair_config()
    expected=deepcopy(parent['dgpo']);a=expected['adaptive_omnifold']
    for b in (a['audit_fit'],a['recalibration']):
        b.update(periodic_pair_features=False,topology_pair_token=False,topology_max_harmonic=1,
            train_last_pet_block=False,train_grouped_sequential_embedding=True,train_invisible_projector=True)
    a['audit_fit'].update(adapter_learning_rate=None,decoder_learning_rate=None,weight_decay=.0005,
        diagnostic_snapshot_dir=OUTPUT+'/diagnostic_snapshots',ratio_audit_export_dir=OUTPUT+'/ratio_audit')
    if cfg['dgpo']!=expected:raise ValueError('Old classifier recipe or fixed measurement protocol changed')
    if cfg['platform']!=parent['platform']:raise ValueError('Requires clean validation truth and16GPUs')
    expected_options=deepcopy(parent['options'])
    expected_options['Training'].update(model_checkpoint_load_path=SOURCE,model_checkpoint_save_path=OUTPUT+'/checkpoints')
    if cfg['options']!=expected_options:raise ValueError('Incorrect raw pretrain source or training options')
    expected_reward=deepcopy(parent['reward_config']);expected_reward['omnifold']['backbone_checkpoint']=SOURCE
    if cfg['reward_config']!=expected_reward:raise ValueError('Classifier backbone must use matching10pct pretrain')
    e=cfg['experiment']
    if (e['protocol']!='old-classifier-pretrain-v1' or not e['classifier_only']
        or e['classifier_fit_count']!=1 or e['reward_fit_count']!=0 or e['policy_updates_per_round']!=0
        or e['source_policy_step'] is not None or e['source_wandb_run'] is not None
        or cfg['logger']['wandb']['id']!='oldpre01' or cfg['logger']['wandb']['resume']!='never'
        or cfg['nersc']['ray']['results_dir']!=OUTPUT+'/ray_results'
        or cfg['nersc']['reproducibility']['source_checkpoint']!=SOURCE):
        raise ValueError('Incorrect classifier-only provenance or output isolation')
    return cfg


def preflight(cfg):
    if not Path(SOURCE).is_file() or not any(Path(CLEAN).glob('*.parquet')):
        raise ValueError('Missing 10pct pretrain checkpoint or clean validation truth')
    validate_manifest(Path(CLEAN).parent/'filter_manifest.json')
    out=Path(cfg['options']['Training']['model_checkpoint_save_path']).parent
    if out.exists():raise ValueError('Output exists; use a new --run-suffix')
    import torch
    checkpoint=torch.load(SOURCE,map_location='cpu',weights_only=False)
    if 'state_dict' not in checkpoint:raise ValueError('Source checkpoint has no state_dict')
    step=checkpoint.get('global_step')
    print(f'Raw pretrained source global_step={step}',flush=True)
    cfg['experiment']['loaded_pretrain_global_step']=step
    cfg['nersc']['reproducibility']['loaded_checkpoint_global_step']=step
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-only',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--run-suffix');args=p.parse_args();cfg=validated_config()
    if args.run_suffix:
        if not re.fullmatch(r'[a-z0-9-]{1,24}',args.run_suffix):p.error('Invalid run suffix')
        out=OUTPUT+'-'+args.run_suffix
        cfg['logger']['wandb']['id']+='-'+args.run_suffix
        cfg['logger']['local']['save_dir']=out+'/logs'
        cfg['nersc']['ray']['results_dir']=out+'/ray_results'
        cfg['options']['Training']['model_checkpoint_save_path']=out+'/checkpoints'
        cfg['dgpo']['adaptive_omnifold']['audit_fit']['diagnostic_snapshot_dir']=out+'/diagnostic_snapshots'
        cfg['dgpo']['adaptive_omnifold']['audit_fit']['ratio_audit_export_dir']=out+'/ratio_audit'
    print('Contract OK: successful weak classifier recipe,10pct raw pretrain,clean validation truth,16GPUs,no policy updates.',flush=True)
    if args.validate_only:return 0
    out=preflight(cfg)
    if args.check_only:return 0
    out.mkdir(parents=True,exist_ok=False)
    resolved=out/'resolved_ratio_experiment.yaml';resolved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return subprocess.run([sys.executable,str(ROOT/'scripts/train_neutrino_backend.py'),
        '--backend','dgpo-evenet','--base-config',str(ROOT/'config/train_diffusion_nersc.yaml'),
        '--overlay-config',str(resolved),'--','--ray-dir',cfg['nersc']['ray']['results_dir']],cwd=ROOT,check=False).returncode


if __name__=='__main__':raise SystemExit(main())
