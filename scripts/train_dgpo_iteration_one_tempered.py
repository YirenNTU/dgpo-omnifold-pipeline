#!/usr/bin/env python3
"""First-start a tempered DGPO run, then resume that overlay's last.ckpt.

Fresh clock from the verified f6b4ec46 step-260 live policy (not step 320, not
122b9d84, not 9073ef9e). Inner loss is L_DGPO(alpha=0.75) + lambda ||v-v_round||^2
with beta_kl=0. Default overlay is lambda=0.5 (9073ef9e). After that run rolled
back to step 0, use --config ...t075_trust02.yaml (lambda=0.2) in a new output.
The dual-classifier overlay is the guarded exception: it pinned-inherits the
d17994d9 epoch=-1 dual-classifier OmniFold/raw-monitor stack (policy still at
the matched 10% diffusion pretrain), then restarts clocks/optimizer under
round-scoped rollback, no hard trust boundary, and raw patience 4 before step
100 then 8.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_dgpo_iteration_one import ROOT, state_digest
from train_neutrino_backend import deep_update

BASE = ROOT / 'config/train_diffusion_nersc.yaml'
EXPECTED_WORKERS = 16
EXPECTED_GPUS_PER_WORKER = 1

DEFAULT = ROOT / 'config/dgpo_omnifold_ztautau_10pct_iteration1_t075_trust05.yaml'
TRUST02 = ROOT / 'config/dgpo_omnifold_ztautau_10pct_iteration1_t075_trust02.yaml'
DUAL_CLASSIFIER = ROOT / 'config/dgpo_omnifold_ztautau_10pct_dual_classifier_t075_trust02_step260.yaml'
CONTROL = ROOT / 'config/dgpo_omnifold_ztautau_10pct_iteration1_only.yaml'
TRUST_OUTPUT_TAG = {0.5: 't075_trust05', 0.2: 't075_trust02'}
DUAL_OUTPUT_TAG = 'dual_classifier_roundrollback_nohardtrust_ess30_t075_trust02_pretrain'
DUAL_WANDB_ID = 'b7f316a4'
DUAL_INHERIT_TAG = 'dual_classifier_inheritalpha_ess30_t075_trust02_pretrain'
DUAL_INHERIT_SNAPSHOT = 'dgpo-epoch=-1-next_ep=0-step=0.ckpt'
TRUST02_SNAPSHOT = 'dgpo-epoch=-1-next_ep=0-step=0.ckpt'
PARENT_STEP260 = 'dgpo-epoch=25-next_ep=26-step=260.ckpt'
DIFFUSION_PRETRAIN = (
    'diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt'
)
PINNED_STACK_KEYS = (
    'state_dict', 'dgpo_adaptive_omnifold_state', 'dgpo_ref_state_dict',
    'dgpo_round_ref_state_dict', 'dgpo_round_ref_sha256',
    'dgpo_omnifold_reward_stack', 'dgpo_omnifold_reward_metadata',
)
RECOVERY_KEYS = (
    'dgpo_checkpoint_version', 'dgpo_next_epoch', 'dgpo_optimizer_state_dict',
    'dgpo_round_ref_state_dict', 'dgpo_adaptive_omnifold_state',
    'dgpo_omnifold_reward_stack', 'state_dict',
)
FORBIDDEN_OUTPUT_MARKERS = (
    'dgpo_omnifold_10pct_iteration1_only_step320_seed42',
    'dgpo_10pct_iteration1_122b9d84',
    't075_trust05_step320',
    DUAL_INHERIT_TAG,
)


def _load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as exc:
        if 'mmap can only be used' not in str(exc):
            raise
        return torch.load(path, map_location='cpu', weights_only=False)


def _is_dual_classifier(config):
    recal = config['dgpo']['adaptive_omnifold']['recalibration']
    return bool(recal.get('reward_classifier'))


def assert_knobs(config):
    dgpo = config['dgpo']
    trigger = dgpo['adaptive_omnifold']['trigger']
    recal = dgpo['adaptive_omnifold']['recalibration']
    dual_classifier = _is_dual_classifier(config)
    if dual_classifier:
        if (
            recal.get('iteration_one_only') is not False
            or (recal['min_iterations'], recal['max_iterations']) != (1, 10)
            or recal.get('residual_closure_schedule')
            != [{'start_step': 0, 'max_auc': 0.52}]
        ):
            raise ValueError(
                'Dual-classifier reward requires 1..10 residuals and AUC 0.52 closure'
            )
    elif (
        recal.get('iteration_one_only') is not True
        or (recal['min_iterations'], recal['max_iterations']) != (1, 1)
    ):
        raise ValueError('Not an iteration-one-only configuration')
    if float(recal['tempering']) != 0.75:
        raise ValueError('Expected tempering 0.75')
    trust = float(dgpo['reference_trust']['coefficient'])
    if trust not in TRUST_OUTPUT_TAG:
        raise ValueError('Expected trust coefficient 0.5 or 0.2')
    output = str(config['options']['Training']['model_checkpoint_save_path'])
    tag = DUAL_OUTPUT_TAG if dual_classifier else TRUST_OUTPUT_TAG[trust]
    if tag not in output:
        raise ValueError(f'trust {trust} must write into a {tag} output directory')
    for other, other_tag in TRUST_OUTPUT_TAG.items():
        if other != trust and other_tag in output:
            raise ValueError(f'trust {trust} must not write into {other_tag}')
    if float(dgpo['beta_kl']) != 0.0:
        raise ValueError('beta_kl must stay 0')
    if float(trigger['raw_improvement_min_delta']) != 0.005:
        raise ValueError('Expected raw_improvement_min_delta 0.005')
    if trigger.get('rollback_to_best_on_plateau') is not True:
        raise ValueError('Expected rollback to the configured-scope raw best')
    if not dgpo['auto_resume_from_last']:
        raise ValueError('Later launches auto-resume this last.ckpt')
    if dgpo.get('auto_resume_fallback_checkpoint_path'):
        raise ValueError('Do not fall back to 122b9d84 or another parent last.ckpt')
    provenance = config['nersc']['reproducibility']
    load_path = str(config['options']['Training']['model_checkpoint_load_path'])
    if dual_classifier:
        if int(provenance.get('source_dgpo_global_step', -1)) != 0:
            raise ValueError(
                'Dual-classifier inherit marks source_dgpo_global_step=0'
            )
        if (
            int(provenance.get('source_classifier_epoch', 0)) != -1
            or int(provenance.get('source_classifier_global_step', -1)) != 0
            or int(provenance.get('source_classifier_next_epoch', -1)) != 0
        ):
            raise ValueError(
                'Dual-classifier inherit must pin d17994d9 epoch=-1/step=0 classifiers'
            )
        if (
            not load_path.endswith(DUAL_INHERIT_SNAPSHOT)
            or DUAL_INHERIT_TAG not in load_path
        ):
            raise ValueError(
                'Dual-classifier inherit must load d17994d9 '
                'dgpo-epoch=-1-next_ep=0-step=0.ckpt'
            )
        if load_path.endswith('last.ckpt') or '/last.ckpt' in load_path:
            raise ValueError(
                'Do not inherit d17994d9 last.ckpt (drifted DGPO policy)'
            )
    else:
        if int(provenance['source_dgpo_global_step']) != 260:
            raise ValueError('This overlay must start from the verified f6b4ec46 step 260')
        if 'step=320' in load_path:
            raise ValueError('Do not load a step=320 source')
    if dual_classifier:
        expected_reward = {
            'head_dropout': 0.15,
            'decoder_hidden_dim': 128,
            'decoder_layers': 1,
            'decoder_heads': 4,
            'periodic_pair_features': False,
            'topology_fourier_embedding': False,
            'topology_conditioning': False,
            'visible_pair_rest_frame': False,
        }
        if trust != 0.2:
            raise ValueError('Dual-classifier comparison keeps trust coefficient 0.2')
        boundary = dgpo['reference_trust']['adaptive_boundary']
        if boundary.get('enabled') is not False:
            raise ValueError(
                'Next-round dual-classifier experiment disables the hard trust boundary'
            )
        if trigger.get('best_scope') != 'round':
            raise ValueError(
                'Next-round dual-classifier rollback must reset its best every reward round'
            )
        if recal.get('reward_classifier') != expected_reward:
            raise ValueError('Dual-classifier reward must use the exact legacy/bulk architecture')
        audit = dgpo['adaptive_omnifold']['audit_fit']
        expected_monitor = {
            'head_dropout': 0.15,
            'decoder_hidden_dim': 128,
            'decoder_layers': 2,
            'decoder_heads': 4,
            'periodic_pair_features': True,
            'topology_fourier_embedding': True,
            'topology_conditioning': True,
            'visible_pair_rest_frame': True,
        }
        if any(audit.get(key) != value for key, value in expected_monitor.items()):
            raise ValueError('Dual-classifier monitor must keep the strong Fourier/rest architecture')
        if float(recal.get('log_ratio_clip', 0.0)) != 2.5:
            raise ValueError('Dual-classifier reward requires log_ratio_clip=2.5')
        if float(recal.get('minimum_ess_fraction', 0.0)) != 0.3:
            raise ValueError('Dual-classifier reward requires minimum_ess_fraction=0.3')
        if recal.get('adaptive_tempering') != {
            'enabled': True,
            'target_ess_fraction': 0.3,
            'minimum': 0.1,
            'grid_steps': 14,
            'inherit_previous': True,
        }:
            raise ValueError(
                'Dual-classifier reward requires ESS>=0.30 alpha search with inherit_previous'
            )
        if recal.get('ess_aware_checkpoint_selection') != {
            'enabled': True,
            'max_checkpoints': 16,
            'first_residual_only': True,
        }:
            raise ValueError(
                'Dual-classifier reward requires first-residual ESS-aware checkpoint selection'
            )
        if dgpo['checkpoint_load_mode'] != 'resume' or not dgpo.get('pinned_classifier_restart'):
            raise ValueError(
                'Dual-classifier first start must pinned-inherit the d17994d9 classifier stack'
            )
        if recal.get('bootstrap_on_start') is not False:
            raise ValueError('Do not cold-fit OmniFold; reuse the d17994d9 epoch=-1 stack')
        if DIFFUSION_PRETRAIN not in str(
            config['reward_config']['omnifold']['backbone_checkpoint']
        ):
            raise ValueError(
                'Dual-classifier backbone must remain diffusion_pretrain_10pct last.ckpt'
            )
        wb = config['logger']['wandb']
        if wb.get('fresh_run') is not False or wb.get('resume') != 'allow':
            raise ValueError(
                'Dual-classifier resume overlay must keep wandb resume=allow and fresh_run=false'
            )
        if wb.get('id') != DUAL_WANDB_ID:
            raise ValueError(
                'Next-round dual-classifier overlay must use its independent W&B id'
            )
        if (
            trigger.get('required_consecutive_checks') != 4
            or trigger.get('patience_schedule')
            != [
                {'start_step': 0, 'required_consecutive_checks': 4},
                {'start_step': 100, 'required_consecutive_checks': 8},
            ]
        ):
            raise ValueError(
                'Next-round dual-classifier overlay requires raw patience 4 then 8'
            )
    elif abs(trust - 0.5) < 1e-12:
        if dgpo['checkpoint_load_mode'] != 'weights_only':
            raise ValueError('First start is weights-only; later launches auto-resume this last.ckpt')
        if not load_path.endswith(PARENT_STEP260):
            raise ValueError('Expected the pinned step=260.ckpt filename')
    else:
        if dgpo['checkpoint_load_mode'] != 'resume' or not dgpo.get('pinned_classifier_restart'):
            raise ValueError('trust 0.2 first-start must pinned-inherit 9073ef9e classifiers')
        if recal.get('bootstrap_on_start') is not False:
            raise ValueError('Do not cold-fit OmniFold; reuse 9073ef9e round-1 stack')
        if not load_path.endswith(TRUST02_SNAPSHOT) or 't075_trust05_step260' not in load_path:
            raise ValueError('Expected 9073ef9e dgpo-epoch=-1-next_ep=0-step=0.ckpt')
        if 'last.ckpt' in load_path:
            raise ValueError('Do not inherit 9073ef9e last.ckpt (step 30 drifted policy)')


def assert_nersc_16gpu(config):
    """Fail closed unless this is the same 16x1 GPU Ray Train layout as 122b9d84."""
    execution = config.get('nersc', {}).get('execution', {})
    platform = config['platform']
    resources = dict(platform.get('resources_per_worker') or {})
    if (
        execution.get('mode') != 'ray_train_ddp'
        or int(execution.get('workers', 0)) != EXPECTED_WORKERS
        or int(execution.get('gpus_per_worker', 0)) != EXPECTED_GPUS_PER_WORKER
    ):
        raise ValueError('Expected nersc.execution ray_train_ddp with 16 workers x 1 GPU')
    if int(platform.get('number_of_workers', 0)) != EXPECTED_WORKERS:
        raise ValueError('Expected platform.number_of_workers=16')
    if int(resources.get('GPU', 0)) != EXPECTED_GPUS_PER_WORKER:
        raise ValueError('Expected 1 GPU per Ray Train worker')
    if platform.get('use_gpu') is False:
        raise ValueError('16-GPU DDP requires platform.use_gpu')
    adaptive = config['dgpo']['adaptive_omnifold']
    for label, fit in (
        ('audit', adaptive['audit_fit']),
        ('reward', adaptive['recalibration']['fit']),
    ):
        global_bs = int(fit['batch_size'])
        micro = int(fit['train_microbatch_size_per_rank'])
        if micro * EXPECTED_WORKERS != global_bs:
            raise ValueError(
                f'{label} batch_size {global_bs} is not {EXPECTED_WORKERS} x '
                f'train_microbatch_size_per_rank {micro}'
            )
    conflict = config['dgpo'].get('gradient_conflict') or {}
    events = int(conflict.get('events_per_block', 0) or 0)
    if events % EXPECTED_WORKERS:
        raise ValueError(f'gradient_conflict.events_per_block {events} is not divisible by 16')


def assert_live_ray_16gpu(*, expected_gpus=EXPECTED_WORKERS):
    """Refuse the trainer's silent local-Ray fallback when the 16-GPU cluster is missing."""
    addr = os.environ.get('RAY_ADDRESS')
    if not addr:
        raise RuntimeError(
            'RAY_ADDRESS is unset. Allocate 4 Perlmutter GPU nodes '
            '(NERSC/salloc_4node_16gpu_ray.sh) and source the Ray cluster first.'
        )
    import ray
    ray.init(address=addr, ignore_reinit_error=True)
    gpus = float(ray.cluster_resources().get('GPU', 0) or 0)
    if gpus < expected_gpus:
        raise RuntimeError(
            f'Ray cluster has {gpus} GPUs; need {expected_gpus}. '
            'Do not launch: dgpo_trainer would wait then continue on fewer GPUs.'
        )


def _effective_patience(trigger, *, global_step):
    patience = int(trigger['required_consecutive_checks'])
    for row in trigger.get('patience_schedule') or []:
        if int(global_step) >= int(row['start_step']):
            patience = int(row['required_consecutive_checks'])
    return patience


def verify(config):
    assert_knobs(config)
    assert_nersc_16gpu(config)
    training = config['options']['Training']
    if any(training['EMA'].get(k, False) for k in ('replace_model_after_load', 'use_for_generation')):
        raise ValueError('Expected live policy, not EMA')
    source = Path(training['model_checkpoint_load_path'])
    output = Path(training['model_checkpoint_save_path'])
    if output.resolve() == source.parent.resolve():
        raise ValueError('Output must not overwrite the source checkpoint directory')
    if any(marker in str(output) for marker in FORBIDDEN_OUTPUT_MARKERS):
        raise ValueError(f'Refusing to write into a 122b9d84/control directory: {output}')
    last = output / 'last.ckpt'
    for label, value in (
        ('classifier backbone', config['reward_config']['omnifold']['backbone_checkpoint']),
        ('training pool', config['platform']['data_parquet_dir']),
        ('validation pool', config['platform']['data_parquet_val_dir']),
    ):
        if not Path(value).exists():
            raise FileNotFoundError(f'{label}: {value}')
    if last.is_file():
        payload = _load_ckpt(last)
        missing = [key for key in RECOVERY_KEYS if key not in payload]
        if missing:
            raise ValueError(f'Incomplete resume checkpoint {last}: {missing}')
        trigger = config['dgpo']['adaptive_omnifold']['trigger']
        schedule = trigger.get('patience_schedule') or []
        if schedule:
            patience = _effective_patience(
                trigger,
                global_step=int(payload.get('global_step', 0)),
            )
            schedule_note = (
                f'live overlay patience schedule → effective '
                f'{patience} (not stored in checkpoint)'
            )
        else:
            patience = int(trigger['required_consecutive_checks'])
            schedule_note = f'live overlay fixed patience={patience}'
        print(
            f'Resume {last}\nCompleted DGPO steps: {payload.get("global_step")}\n'
            f'Next epoch: {payload.get("dgpo_next_epoch")}; '
            f'within-epoch progress: {payload.get("dgpo_epoch_step", 0)}\n'
            f'{schedule_note}\n'
            f'Output: {output}',
            flush=True,
        )
        return 'resume'
    if output.exists() and any(output.glob('*.ckpt')):
        raise FileExistsError(
            f'Output has checkpoints but no last.ckpt: {output}; '
            'refusing a mixed first-start'
        )
    provenance = config['nersc']['reproducibility']
    payload = _load_ckpt(source)
    trust = float(config['dgpo']['reference_trust']['coefficient'])
    dual_classifier = _is_dual_classifier(config)
    if dual_classifier or abs(trust - 0.2) < 1e-12:
        expected = (
            provenance.get('source_classifier_epoch', -1),
            provenance.get('source_classifier_global_step', 0),
            provenance.get('source_classifier_next_epoch', 0),
        )
        missing = [key for key in PINNED_STACK_KEYS if key not in payload]
        if missing:
            label = 'd17994d9' if dual_classifier else '9073ef9e'
            raise ValueError(f'{label} snapshot missing classifier stack: {missing}')
    else:
        expected = (
            provenance.get('source_epoch'),
            provenance['source_dgpo_global_step'],
            provenance.get('source_next_epoch'),
        )
    actual = (
        payload.get('epoch'),
        payload.get('global_step'),
        payload.get('dgpo_next_epoch'),
    )
    if actual != expected:
        raise ValueError(f'Source checkpoint clock mismatch: {actual} != {expected}')
    digest = state_digest(payload['state_dict'])
    expected_digest = provenance.get('source_policy_sha256')
    if expected_digest not in (None, ''):
        if digest != expected_digest:
            raise ValueError(f'Source live state_dict SHA256 mismatch: {digest}')
    elif not dual_classifier and abs(trust - 0.5) < 1e-12:
        raise ValueError('Expected pinned source_policy_sha256 for the step-260 start')
    inherit = (
        ' pinned OmniFold inherit;'
        if dual_classifier or abs(trust - 0.2) < 1e-12
        else ''
    )
    print(
        f'Verified live policy: {source}\nSource step: {actual[1]}\nSHA256: {digest}\n'
        f'New DGPO clock: step=0 epoch=0;{inherit} L = L_DGPO(0.75) + '
        f'{trust}||v-v_round||^2.\n'
        f'NERSC layout: {EXPECTED_WORKERS} workers x {EXPECTED_GPUS_PER_WORKER} GPU, '
        f'policy batch {config["platform"]["batch_size"]}/worker.\n'
        f'Output: {output}',
        flush=True,
    )
    return 'first_start'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.config.resolve() == CONTROL.resolve():
        raise ValueError('This launcher is not the 122b9d84 control overlay')
    overlay = yaml.safe_load(args.config.read_text())
    merged = deep_update(yaml.safe_load(BASE.read_text()), overlay)
    verify(overlay)
    assert_nersc_16gpu(merged)
    if args.check_only:
        return
    if not os.environ.get('WANDB_API_KEY'):
        print('WARNING: WANDB_API_KEY is unset; W&B logging will fail after bootstrap.', flush=True)
    assert_live_ray_16gpu()
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT / 'evenet_dgpo'), str(ROOT), env.get('PYTHONPATH', '')]
    )
    subprocess.run(
        [
            sys.executable, str(ROOT / 'scripts/train_neutrino_backend.py'),
            '--backend', 'dgpo-evenet',
            '--base-config', str(BASE),
            '--overlay-config', str(args.config.resolve()), '--', '--ray-dir',
            merged['nersc']['ray']['results_dir'],
        ],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == '__main__':
    main()
