"""Pin production last.ckpt, fresh-fit a classifier, then exit before actor training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'evenet_dgpo')]


def configure_classifier_mode(cfg):
    """The legacy experiment.classifier_only flag selects H4, NOT tau."""
    if cfg.get('reward_config', {}).get('type') != 'conditional_tau':
        raise ValueError('Tau classifier experiment requires conditional_tau reward')
    if cfg['dgpo'].get('adaptive_omnifold', {}).get('enabled', False):
        raise ValueError('Tau classifier experiment must not enable adaptive OmniFold')
    cfg.setdefault('experiment', {}).update(classifier_only=False, tau_classifier_only=True, actor_updates=0)
    cfg['dgpo']['tau_ratio']['classifier_only'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--checkpoint', type=Path, help='Reuse an already pinned checkpoint; otherwise pin current production last.ckpt')
    args = p.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if settings['workers'] != 16 or settings['batch_size'] != 1024:
        raise ValueError('Expected 16 GPUs, 1024 paired events per GPU')
    depths = settings.get('classifier_depths', [3])
    if not depths or len(set(depths)) != len(depths) or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError('Classifier depths must be unique positive integers')
    source = Path(settings['source_root'])
    source_runtime = Path(settings.get('source_runtime', source/'runtime.yaml'))
    cfg = yaml.safe_load(source_runtime.read_text())
    import torch
    from evenet.dataset.filtered_data import validate_filtered_dataset
    for key, expected in [('data_parquet_dir',416701), ('data_parquet_val_dir',119002)]:
        if validate_filtered_dataset(cfg['platform'][key])['rows'] != expected:
            raise ValueError('Filtered data population changed: '+key)
    if cfg['options']['Training']['EMA']['enable']:
        raise ValueError('Raw weights required')
    checkpoint = (args.checkpoint or Path(settings.get('checkpoint', source/'checkpoints/last.ckpt'))).resolve(strict=True)
    root = Path(settings['output_root']); root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='fresh-', dir=root))
    pinned = output/'source.ckpt'
    try: os.link(checkpoint, pinned)
    except OSError: shutil.copy2(checkpoint, pinned)
    saved = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
    if not saved.get('dgpo_omnifold_reward_stack') or int(saved.get('global_step',0)) <= 0:
        raise ValueError('Requires a full-state current DGPO checkpoint with installed reward')
    meta = dict(source_runtime=str(source_runtime), source_checkpoint=str(checkpoint), pinned_checkpoint=str(pinned),
                policy_step=int(saved['global_step']), epoch=int(saved['epoch']))
    if settings.get('expected_policy_step') is not None and meta['policy_step'] != settings['expected_policy_step']:
        raise ValueError('Pinned policy step differs from experiment contract')
    del saved
    dg = cfg['dgpo']
    dg.update(checkpoint_load_mode='resume', auto_resume_from_last=False,
              auto_resume_fallback_checkpoint_path=None, auto_resume_best_source_checkpoint_dir=None)
    dg['tau_ratio'].update(classifier_only=True, reward_cij_probe=False, workers=16,
                           output=str(output/'diagnostics'), generation_batch_size=1024,
                           classifier_depths=depths)
    dg['tau_ratio']['classifier_attention_ablation'] = bool(settings.get('classifier_attention_ablation', False))
    dg['tau_ratio']['fit']['batch_size'] = 1024
    cfg['platform'].update(number_of_workers=16, use_gpu=True)
    cfg['options']['Training'].update(model_checkpoint_load_path=str(pinned),
        pretrain_model_load_path=None, model_checkpoint_save_path=str(output/'unused_checkpoints'))
    cfg['logger']['local']['save_dir'] = str(output/'logs')
    cfg['logger']['wandb'].update(id=None, resume='never', fresh_run=True,
        run_name=settings['wandb_name'], group='Current DGPO classifier alignment')
    cfg['nersc']['ray']['results_dir'] = str(output/'ray_results')
    configure_classifier_mode(cfg)
    cfg['experiment'].update(source_policy_step=meta['policy_step'],
        purpose='Fresh current-policy ratio; inherited vs fresh head on identical candidates; no actor updates')
    runtime = output/'runtime.yaml'; runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output/'source.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('READY:', json.dumps(dict(**meta, runtime=str(runtime), workers=16, actor_updates=0)), flush=True)
    if not args.prepare_only:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(REPO), str(REPO/'scripts'), str(REPO/'evenet_dgpo'), env.get('PYTHONPATH','')))
        subprocess.run([sys.executable, '-u', str(REPO/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'),
                        str(runtime), '--ray-dir', str(output/'ray_results')], cwd=REPO, env=env, check=True)


if __name__ == '__main__': main()
