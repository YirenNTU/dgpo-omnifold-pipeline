"""User-launched full DGPO resume with startup attention refit, then every 5 epochs."""
import argparse
import os
import shutil
from pathlib import Path
import subprocess
import sys
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'evenet_dgpo')]


def validate_continuation(settings, cfg, state, resumed):
    if settings.get('require_existing_resume'):
        if not resumed:
            raise ValueError('Existing run last.ckpt required; do not fall back to step1780')
        if not state.get('dgpo_omnifold_reward_stack', {}).get('head', {}).get('cross_attention', False):
            raise ValueError('Resume requires the trained attention classifier')
    if settings.get('require_original_diffusion'):
        enabled = cfg.get('network', {}).get('VisibleConditioning', {}).get('diffusion_token_readout', {}).get('enabled', False)
        keys = state.get('state_dict', {})
        if enabled or settings.get('add_diffusion_attention') or any(
                k.removeprefix('model.').startswith('TruthGeneration.visible_conditioning.token_readout.') for k in keys):
            raise ValueError('Expected original diffusion without the extra token-readout branch')


def configure(cfg, settings, checkpoint, output):
    from scripts.train_tau_classifier_current_dgpo import configure_classifier_mode
    configure_classifier_mode(cfg)  # Validate tau/non-adaptive backend, then enable actor updates.
    cfg['experiment'].update(classifier_only=False, tau_classifier_only=False,
        protocol='attention-reward-dgpo', source_policy_step=1780,
        first_head='fresh best-val attention refit before first actor update')
    cfg['experiment'].pop('actor_updates', None)
    dg = cfg['dgpo']
    dg.update(checkpoint_load_mode='resume', auto_resume_from_last=False)
    dg['tau_ratio'].update(classifier_only=False, classifier_attention_ablation=False,
        production_cross_attention=True, startup_refit_policy_step=1780,
        refit_every_epochs=5, refit_relative_to_install=True, reward_cij_probe=True,
        output=str(output/'tau_diagnostics'), output_root=str(output), workers=16)
    if 'validation_every_epochs' in settings:
        cadence = settings['validation_every_epochs']
        if type(cadence) is not int or cadence < 1:
            raise ValueError('validation_every_epochs must be a positive integer')
        dg['tau_ratio']['validation_every_epochs'] = cadence
        dg['tau_ratio']['validation_relative_to_install'] = bool(settings.get('validation_relative_to_install', False))
        if dg['tau_ratio']['validation_relative_to_install'] and cadence != dg['tau_ratio']['refit_every_epochs']:
            raise ValueError('Install-aligned validation requires the same cadence as refit')
    if dg['reference_trust']['objective'] != 'velocity_mse' or dg['reference_trust']['coefficient'] != 1 or not dg['reference_trust']['enabled']:
        raise ValueError('Preserve enabled coefficient-1 velocity MSE reference')
    if dg['tau_ratio']['fit']['batch_size'] != 1024 or cfg['platform']['number_of_workers'] != 16:
        raise ValueError('Expected 16 GPUs and 1024 classifier paired events per GPU')
    if cfg['options']['Training']['EMA']['enable']:
        raise ValueError('Raw actor required')
    cfg['options']['Training'].update(model_checkpoint_load_path=str(checkpoint),
        pretrain_model_load_path=None, model_checkpoint_save_path=str(output/'checkpoints'))
    cfg['logger']['local']['save_dir'] = str(output/'logs')
    cfg['logger']['wandb'].update(id=None, resume='never', fresh_run=True,
        run_name=settings['wandb_name'], group='Attention tau reward DGPO')
    cfg['nersc']['ray']['results_dir'] = str(output/'ray_results')
    if settings.get('add_diffusion_attention', False):
        visible = cfg['network']['VisibleConditioning']
        if visible.get('diffusion_token_readout', {}).get('enabled', False):
            raise ValueError('Initial source runtime already has diffusion token attention')
        visible['diffusion_token_readout'] = dict(enabled=True, mode='residual', block='last', width=64, heads=4)
        dg['add_diffusion_attention'] = True
        cfg['experiment'].update(protocol='diffusion-attention-dgpo',
            intervention='last-block zero-output cross-attention residual; retain global FiLM',
            first_head='inherit trained attention classifier; preserve refit clock')
        cfg['experiment'].pop('source_policy_step', None)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    output = Path(settings['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    source_runtime = Path(settings['source_runtime'])
    source_checkpoint = Path(settings['checkpoint'])
    if settings.get('add_diffusion_attention', False):
        pinned = output/'source.ckpt'
        runtime_pin = output/'source_runtime.yaml'
        if not pinned.exists():
            resolved = source_checkpoint.resolve(strict=True)
            try: os.link(resolved, pinned)
            except OSError: shutil.copy2(resolved, pinned)
        if not runtime_pin.exists():
            shutil.copy2(source_runtime, runtime_pin)
        source_checkpoint, source_runtime = pinned, runtime_pin
    cfg = yaml.safe_load(source_runtime.read_text())
    from evenet.dataset.filtered_data import validate_filtered_dataset
    for key, count in [('data_parquet_dir',416701), ('data_parquet_val_dir',119002)]:
        if validate_filtered_dataset(cfg['platform'][key])['rows'] != count:
            raise ValueError('Filtered population changed')
    last = output/'checkpoints/last.ckpt'
    resumed = last.exists() or last.is_symlink()
    checkpoint = (last if resumed else source_checkpoint).resolve(strict=True)
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    validate_continuation(settings, cfg, state, resumed)
    if not state.get('dgpo_omnifold_reward_stack') or (not resumed and not settings.get('add_diffusion_attention') and state['global_step'] != 1780):
        raise ValueError('Requires full reward state and initial step1780')
    if settings.get('add_diffusion_attention') and not state['dgpo_omnifold_reward_stack']['head'].get('cross_attention', False):
        raise ValueError('Diffusion attention experiment must inherit the trained attention classifier')
    step = state['global_step']
    del state
    cfg = configure(cfg, settings, checkpoint, output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = output/'runtime.yaml'
    runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
    first_action = ('Inherit reward; add zero-output diffusion attention, preserve old Adam moments.'
                    if settings.get('add_diffusion_attention') else 'Refit attention classifier before first actor update.')
    if resumed:
        first_action = 'Resume saved actor, attention classifier, optimizer, reference and refit clock.'
    print(f'READY: full resume step={step} checkpoint={checkpoint}\n'
          f'{first_action}\n'
          f'Refit every 5 completed epochs after install; subsequent launches resume saved head.\n'
          f'Runtime: {runtime}', flush=True)
    if not args.prepare_only:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(REPO),str(REPO/'scripts'),str(REPO/'evenet_dgpo'),env.get('PYTHONPATH','')])
        subprocess.run([sys.executable,'-u',str(REPO/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'),
                        str(runtime),'--ray-dir',str(output/'ray_results')],cwd=REPO,env=env,check=True)


if __name__ == '__main__': main()
