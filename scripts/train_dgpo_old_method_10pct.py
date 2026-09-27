#!/usr/bin/env python3
"""Launch the 10% recreation of the original 5% hard DGPO method (nc4shnpg).

Default overlay cold-starts from the matched 10% diffusion pretrain last.ckpt
with soft trust coefficient 1.0, no hard boundary, fixed tempering 0.75,
weighted-gap retrain (every 5 epochs, consecutive 1, age 20), residuals
2..5 + acceptance BA<0.52, head_dropout 0.25, and fresh classifiers on every
refit (no inherit).

The parity phi-only Fourier H4 fork resumes the parent last.ckpt into a new
output/W&B directory. It restores the proven nonlinear H4 path, trains the
same PEFT scope in OmniFold and staleness, and warm-starts only compatible raw
monitors and same-fold iteration-1 classifiers with fresh optimizers.

The legacy fine-tuned-body ablation starts from the 10% diffusion fine-tune and
uses that checkpoint only to initialize the classifier Body. TruthGeneration
is discarded; the Body stays frozen except for GroupedSequentialEmbedding,
InvisibleInputProjector, and internal PET adapters.

  shifter python3 scripts/train_dgpo_old_method_10pct.py --check-only
  shifter python3 scripts/train_dgpo_old_method_10pct.py \\
    --config config/dgpo_omnifold_ztautau_10pct_old_method_hard_dropout015.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from train_neutrino_backend import build_runtime_config, read_overlay_yaml, read_yaml

DEFAULT = ROOT / 'config/dgpo_omnifold_ztautau_10pct_old_method_hard_ablation.yaml'
H4_PARITY = ROOT / 'config/dgpo_omnifold_ztautau_10pct_old_method_hard_dropout015.yaml'
FINETUNED_BODY = (
    ROOT / 'config/dgpo_omnifold_ztautau_10pct_legacy_finetuned_body_ablation.yaml'
)
CONTROL_5PCT = ROOT / 'config/dgpo_omnifold_ztautau.yaml'
BASE = ROOT / 'config/train_diffusion_nersc.yaml'
EXPECTED_WORKERS = 16
EXPECTED_GPUS_PER_WORKER = 1
OUTPUT_TAG = 'old_method_hard_nc4shnpg'
H4_PARITY_OUTPUT_TAG = 'fourier_phi_h4_parity_warm_from_c4a91e07'
FINETUNED_BODY_OUTPUT_TAG = 'legacy_finetuned_10pct_body'
WANDB_ID = 'c4a91e07'
H4_PARITY_WANDB_ID = 'p4r1tyh4'
FINETUNED_BODY_WANDB_ID = 'ft10body'
H4_PARITY_REFIT_ONCE_ID = 'fourier_phi_h4_parity_warm_from_c4a91e07_v2'
PARENT_LAST = (
    'dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42'
    '/checkpoints/last.ckpt'
)
DIFFUSION_PRETRAIN = 'diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt'
STIC_TRAIN = 'omnifold_attention_10pct_stic_filtered_test1/train'
HELD_OUT_VAL = 'diffusion_val_20pct_seed42/val'
RECOVERY_KEYS = (
    'dgpo_checkpoint_version',
    'dgpo_next_epoch',
    'dgpo_optimizer_state_dict',
    'dgpo_round_ref_state_dict',
    'dgpo_adaptive_omnifold_state',
    'dgpo_omnifold_reward_stack',
    'state_dict',
)
FORBIDDEN_OUTPUT_MARKERS = (
    'dual_classifier',
    'raw_plateau',
    'iteration1',
    'velocity_mse_trust10',
    'visible_rest',
)


def _load_ckpt(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as exc:
        if 'mmap can only be used' not in str(exc):
            raise
        payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'DGPO checkpoint is not a mapping: {path}')
    return payload


def assert_nersc_16gpu(config: dict) -> None:
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
        ('audit_fit', adaptive['audit_fit']),
        ('recalibration.fit', adaptive['recalibration']['fit']),
    ):
        if int(fit['batch_size']) != 16384:
            raise ValueError(f'{label}.batch_size must stay 16384 for 16-GPU global batches')
        if int(fit['train_microbatch_size_per_rank']) != 1024:
            raise ValueError(f'{label} microbatch must stay 1024/rank')


def _is_h4_parity_fork(config: dict) -> bool:
    save = str(config['options']['Training']['model_checkpoint_save_path'])
    return H4_PARITY_OUTPUT_TAG in save


def _is_finetuned_body_ablation(config: dict) -> bool:
    save = str(config['options']['Training']['model_checkpoint_save_path'])
    return FINETUNED_BODY_OUTPUT_TAG in save


def assert_knobs(config: dict) -> None:
    dgpo = config['dgpo']
    adaptive = dgpo['adaptive_omnifold']
    trigger = adaptive['trigger']
    recal = adaptive['recalibration']
    trust = dgpo['reference_trust']
    boundary = trust.get('adaptive_boundary') or {}
    training = config['options']['Training']
    load_path = str(training['model_checkpoint_load_path'])
    save_path = str(training['model_checkpoint_save_path'])
    h4_parity = _is_h4_parity_fork(config)
    finetuned_body = _is_finetuned_body_ablation(config)
    expected_dropout = 0.15 if h4_parity else 0.25
    expected_wandb = (
        H4_PARITY_WANDB_ID
        if h4_parity
        else FINETUNED_BODY_WANDB_ID
        if finetuned_body
        else WANDB_ID
    )

    if adaptive.get('monitor_mode') != 'weighted_and_raw':
        raise ValueError('Old method requires monitor_mode=weighted_and_raw')
    if adaptive.get('single_pool_train_validation'):
        raise ValueError(
            'weighted-gap + acceptance audit cannot use single_pool_train_validation'
        )
    if float(trust.get('coefficient', 0.0)) != 1.0 or trust.get('enabled') is not True:
        raise ValueError('Soft trust must stay enabled with coefficient 1.0')
    if str(trust.get('objective', 'velocity_mse')) != 'velocity_mse':
        raise ValueError('Soft trust objective must stay velocity_mse')
    if boundary.get('enabled'):
        raise ValueError('Hard trust adaptive_boundary must stay disabled')
    if float(recal.get('tempering', 0.0)) != 0.75:
        raise ValueError('Expected fixed tempering 0.75')
    if (recal.get('adaptive_tempering') or {}).get('enabled'):
        raise ValueError('ESS / adaptive tempering must stay off')
    if recal.get('log_ratio_clip') not in (None,):
        raise ValueError('log_ratio_clip must stay null')
    if (recal.get('ess_aware_checkpoint_selection') or {}).get('enabled'):
        raise ValueError('ess_aware_checkpoint_selection must stay off')
    expected_warm_iterations = [1] if h4_parity else []
    if list(recal.get('warm_start_iterations') or []) != expected_warm_iterations:
        raise ValueError(
            f'Expected warm_start_iterations={expected_warm_iterations}'
        )
    if bool(recal.get('warm_start_from_iteration_one')) != h4_parity:
        raise ValueError(
            'H4 parity fork alone must warm-start later residuals from '
            'same-round iteration 1'
        )
    if recal.get('crossfit_partition', 'auto') not in ('auto', 'identity'):
        raise ValueError("crossfit_partition must be 'auto' or 'identity'")
    if dgpo.get('pinned_classifier_restart'):
        raise ValueError('pinned_classifier_restart must stay false')
    if bool(trigger.get('warm_start_classifier')) != h4_parity:
        raise ValueError(
            'H4 parity fork requires a warm-start raw monitor; control stays fresh'
        )
    if trigger.get('raw_audit_enabled') is not True:
        raise ValueError('Enable raw_audit for diagnostic staleness/raw_auc logging')
    if trigger.get('rollback_to_best_on_plateau'):
        raise ValueError('Old weighted-gap controller does not rollback on plateau')
    expected_staleness_epochs = 1 if h4_parity else 5
    if int(adaptive.get('staleness_every_n_epochs') or 0) != expected_staleness_epochs:
        raise ValueError(
            f'Expected staleness_every_n_epochs={expected_staleness_epochs}'
        )
    if float(trigger.get('retrain_auc_margin', 0.0)) != 0.005:
        raise ValueError('Expected retrain_auc_margin=0.005')
    expected_consecutive = 2 if h4_parity else 1
    if int(trigger.get('required_consecutive_epochs') or 0) != expected_consecutive:
        raise ValueError(
            f'Expected required_consecutive_epochs={expected_consecutive}'
        )
    if int(trigger.get('max_reward_age_epochs') or 0) != 20:
        raise ValueError('Expected max_reward_age_epochs=20')
    expected_cooldown = 5 if h4_parity else 0
    if int(trigger.get('retrain_cooldown_epochs') or 0) != expected_cooldown:
        raise ValueError(
            f'Expected retrain_cooldown_epochs={expected_cooldown}'
        )
    if (recal.get('min_iterations'), recal.get('max_iterations')) != (2, 5):
        raise ValueError('Expected residuals 2..5')
    if recal.get('acceptance_audit_enabled') is not True:
        raise ValueError('acceptance_audit_enabled must stay true')
    if float(recal.get('acceptance_max_balanced_accuracy', 0.0)) != 0.52:
        raise ValueError('Expected acceptance_max_balanced_accuracy=0.52')
    if float(recal.get('residual_min_auc_gain', 0.0)) != 0.01:
        raise ValueError('Expected residual_min_auc_gain=0.01')
    if int(trigger.get('probe_max_events') or 0) != 250000:
        raise ValueError('Expected probe_max_events=250000 for sensitive audit')
    if int(recal.get('score_pool_events') or 0) != 250000:
        raise ValueError('Expected score_pool_events=250000 held-out closure panel')
    if recal.get('iteration_one_only'):
        raise ValueError('iteration_one_only must stay false')
    if recal.get('decoder_hidden_dim') != 128 or recal.get('decoder_layers') != 1:
        raise ValueError('Legacy decoder must stay 128x1')
    if recal.get('train_grouped_sequential_embedding') is not True:
        raise ValueError('GroupedSequentialEmbedding must be trainable')
    if recal.get('train_invisible_projector') is not True:
        raise ValueError('InvisibleInputProjector must be trainable')
    if float(recal.get('head_dropout', 0.0)) != expected_dropout:
        raise ValueError(f'Expected classifier head_dropout {expected_dropout}')
    if float(adaptive['audit_fit'].get('head_dropout', 0.0)) != expected_dropout:
        raise ValueError(f'Expected audit head_dropout {expected_dropout}')
    fit = recal['fit']
    if fit.get('require_saturation') is True:
        explicit_min_steps = fit.get('min_steps')
        if explicit_min_steps is not None and int(explicit_min_steps) < 1:
            raise ValueError('require_saturation needs recalibration.fit.min_steps>=1')
        if explicit_min_steps is None and int(fit.get('min_epochs') or 0) < 1:
            raise ValueError(
                'require_saturation needs min_epochs>=1 when min_steps is omitted'
            )
    audit = adaptive['audit_fit']
    shared = (
        'anomaly_detection_steps',
        'batch_size',
        'train_microbatch_size_per_rank',
        'drop_last_batch',
        'learning_rate',
        'backbone_learning_rate',
        'weight_decay',
        'gradient_clip_norm',
        'sampling',
        'safety_max_epochs',
        'min_epochs',
        'enforce_min_epochs',
        'validation_interval_epochs',
        'validation_patience_epochs',
        'validation_min_delta',
        'validation_batch_size',
        'progress_every_n_steps',
        'restore_best',
        'topology_warmup_steps',
        'topology_body_unfreeze_step',
        'topology_warmup_learning_rate',
    )
    for key in shared:
        if audit.get(key) != fit.get(key):
            raise ValueError(
                f'audit_fit.{key}={audit.get(key)!r} must match '
                f'recalibration.fit.{key}={fit.get(key)!r}'
            )
    # min_steps: 1 + min_epochs>1 is a footgun unless enforce_min_epochs is on.
    for label, block in (('recalibration.fit', fit), ('audit_fit', audit)):
        min_epochs = float(block.get('min_epochs') or 0)
        min_steps = block.get('min_steps')
        if (
            min_epochs > 1
            and min_steps is not None
            and int(min_steps) == 1
            and block.get('enforce_min_epochs') is not True
        ):
            raise ValueError(
                f'{label}: min_steps=1 with min_epochs={min_epochs} arms patience '
                'immediately; omit min_steps or set enforce_min_epochs=true'
            )
    for key in (
        'head_dropout',
        'decoder_hidden_dim',
        'decoder_layers',
        'decoder_heads',
        'periodic_pair_features',
        'topology_fourier_embedding',
        'topology_direct_logit',
        'topology_context_residual_scale',
        'topology_conditioning',
        'visible_pair_rest_frame',
        'topology_max_harmonic',
        'topology_include_theta_pair',
        'topology_hidden_dim',
        'topology_embedding_dim',
        'topology_fusion_hidden_dim',
        'topology_dropout',
        'train_layernorm',
        'train_encoder',
        'train_grouped_sequential_embedding',
        'train_invisible_projector',
        'train_backbone',
        'asymmetric_attention',
    ):
        if key not in audit and key not in recal:
            continue
        if audit.get(key) != recal.get(key):
            raise ValueError(
                f'audit_fit.{key}={audit.get(key)!r} must match '
                f'recalibration.{key}={recal.get(key)!r}'
            )
    for flag in (
        'topology_conditioning',
        'visible_pair_rest_frame',
    ):
        if recal.get(flag) or adaptive['audit_fit'].get(flag):
            raise ValueError(f'{flag} must stay false (no AdaLN / rest-frame stack)')
    if h4_parity:
        if config['logger']['wandb'].get('classifier_loss_curves') is not False:
            raise ValueError(
                'H4 parity fork uses fixed omnifold_live curves, not '
                'per-fit classifier metric families'
            )
        if int(trigger.get('probe_seed', -1)) != int(recal.get('seed', -2)):
            raise ValueError(
                'H4 parity fork requires matching staleness/recalibration '
                'candidate-generation seeds'
            )
        if recal.get('periodic_pair_features') is not True:
            raise ValueError('H4 parity fork requires periodic_pair_features')
        if recal.get('topology_fourier_embedding') is not True:
            raise ValueError('H4 parity fork requires topology_fourier_embedding')
        if recal.get('topology_direct_logit'):
            raise ValueError('H4 parity fork removes the failed direct linear shortcut')
        if float(recal.get('topology_context_residual_scale', -1)) != 1.0:
            raise ValueError('H4 parity fork requires full nonlinear context fusion')
        if int(recal.get('topology_max_harmonic') or 0) != 4:
            raise ValueError('H4 parity fork must use topology_max_harmonic=4')
        if recal.get('topology_include_theta_pair'):
            raise ValueError('Phi-only fork must keep topology_include_theta_pair false')
        if int(recal['fit'].get('min_steps') or 0) != 1:
            raise ValueError('H4 parity fit uses the explicit per-fold floor')
        if int(recal['fit'].get('min_steps_per_fold') or 0) != 1000:
            raise ValueError('Cold H4 OmniFold needs 1000 updates per fold')
        if float(recal['fit'].get('warm_start_min_epochs_per_fold') or 0) != 10:
            raise ValueError('Warm H4 OmniFold needs 10 actual fold epochs')
        if float(recal['fit'].get('validation_patience_epochs') or 0) != 15.0:
            raise ValueError('H4 parity fork needs 15 patience epochs')
        if float(recal['fit'].get('validation_min_delta') or 0) != 0.0001:
            raise ValueError('Phi fork must use validation_min_delta=0.0001')
        if recal.get('train_layernorm'):
            raise ValueError('H4 parity fork must keep LayerNorm frozen')
        if recal.get('train_encoder'):
            raise ValueError('H4 parity fork must keep GlobalEmbedding frozen')
        if recal.get('train_backbone'):
            raise ValueError('Phi fork must keep train_backbone false (PET attention frozen)')
        if adaptive['audit_fit'].get('train_layernorm'):
            raise ValueError('audit_fit must keep LayerNorm frozen with OmniFold')
        if adaptive['audit_fit'].get('train_encoder'):
            raise ValueError('audit_fit must keep GlobalEmbedding frozen with OmniFold')
        if adaptive['audit_fit'].get('train_backbone'):
            raise ValueError('audit_fit must keep train_backbone false')
        for block_name, block in (
            ('recalibration.fit', recal['fit']),
            ('audit_fit', adaptive['audit_fit']),
        ):
            if int(block.get('topology_warmup_steps') or 0) != 0:
                raise ValueError(f'{block_name} must not stage topology training')
            if int(block.get('topology_body_unfreeze_step') or 0) != 0:
                raise ValueError(f'{block_name} must unfreeze its PEFT scope immediately')
        for key, expected in (
            ('topology_hidden_dim', 64),
            ('topology_embedding_dim', 32),
            ('topology_fusion_hidden_dim', 64),
        ):
            if int(recal.get(key) or 0) != expected:
                raise ValueError(f'{key} must stay {expected} for the phi Fourier fork')
        if float(recal.get('topology_dropout', -1)) != 0.15:
            raise ValueError('context topology_dropout must stay 0.15')
        audit_readiness = adaptive['audit_fit'].get('training_readiness')
        if audit_readiness != {
            'cold_start_min_epochs': 100,
            'warm_start_min_epochs': 5,
        }:
            raise ValueError('Raw monitor requires cold100/warm5 epoch readiness')
        if int(adaptive['audit_fit'].get('min_steps') or 0) != 300:
            raise ValueError('Every staleness audit needs at least 300 updates')
        if not adaptive.get('fixed_audit_panel') or not adaptive.get(
            'cache_event_inputs'
        ):
            raise ValueError('Warm raw monitoring requires a fixed cached panel')
    else:
        if recal.get('periodic_pair_features') or adaptive['audit_fit'].get(
            'periodic_pair_features'
        ):
            raise ValueError(
                'periodic_pair_features must stay false for the parent legacy hard method'
            )
        if recal.get('topology_fourier_embedding') or adaptive['audit_fit'].get(
            'topology_fourier_embedding'
        ):
            raise ValueError(
                'topology_fourier_embedding must stay false for the parent legacy hard method'
            )
        for key in (
            'train_layernorm',
            'train_encoder',
            'train_backbone',
        ):
            if recal.get(key) or adaptive['audit_fit'].get(key):
                raise ValueError(f'parent control must keep {key}=false')
    if float(dgpo.get('beta_kl', 1.0)) != 0.0:
        raise ValueError('beta_kl must stay 0')
    if not dgpo.get('auto_resume_from_last'):
        raise ValueError('Later launches auto-resume this last.ckpt')
    if dgpo.get('auto_resume_fallback_checkpoint_path'):
        raise ValueError('Do not pin a fallback checkpoint for this cold-start ablation')
    classifier_backbone = str(
        config['reward_config']['omnifold']['backbone_checkpoint']
    )
    if finetuned_body:
        if DIFFUSION_PRETRAIN not in classifier_backbone:
            raise ValueError(
                'Fine-tuned-body ablation must initialize the classifier from '
                'the 10% diffusion fine-tune last.ckpt'
            )
        if recal.get('body_only_checkpoint') is not True:
            raise ValueError(
                'Fine-tuned-body ablation must discard checkpoint task heads'
            )
    else:
        if DIFFUSION_PRETRAIN not in classifier_backbone:
            raise ValueError(
                'OmniFold backbone must stay diffusion_pretrain_10pct last.ckpt'
            )
        if recal.get('body_only_checkpoint'):
            raise ValueError(
                'Only the fine-tuned-body ablation may use body_only_checkpoint'
            )
    if OUTPUT_TAG not in save_path:
        raise ValueError(f'Output must stay under a {OUTPUT_TAG} directory')
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        if marker in save_path:
            raise ValueError(f'Do not write into a {marker} experiment directory')
    if h4_parity:
        if dgpo.get('checkpoint_load_mode') != 'resume':
            raise ValueError('H4 parity fork first start must resume the parent last.ckpt')
        if recal.get('bootstrap_on_start'):
            raise ValueError('H4 parity fork must not re-bootstrap; parent stack is restored')
        if recal.get('refit_once_on_resume') is not True:
            raise ValueError('H4 parity fork must refit OmniFold once at start')
        if recal.get('refit_once_fail_closed'):
            raise ValueError(
                'H4 parity fork must not fail closed on startup refit; '
                'late-parent chance AUC correctly rejects and should keep incumbent'
            )
        if str(recal.get('refit_once_id') or '') != H4_PARITY_REFIT_ONCE_ID:
            raise ValueError(
                f'H4 parity fork requires refit_once_id={H4_PARITY_REFIT_ONCE_ID}'
            )
        if PARENT_LAST not in load_path:
            raise ValueError('H4 parity fork must load the parent c4a91e07 last.ckpt')
        if H4_PARITY_OUTPUT_TAG not in save_path:
            raise ValueError(f'H4 parity output must include {H4_PARITY_OUTPUT_TAG}')
        if Path(load_path).resolve().parent == Path(save_path).resolve():
            raise ValueError('H4 parity fork must not write into the parent checkpoint dir')
        if 'nohardtrust_seed42/checkpoints' in save_path and H4_PARITY_OUTPUT_TAG not in save_path:
            raise ValueError('Refusing to overwrite the parent nohardtrust output tree')
    elif finetuned_body:
        if recal.get('refit_once_on_resume'):
            raise ValueError(
                'Fine-tuned-body ablation bootstraps a fresh reward stack'
            )
        if dgpo.get('checkpoint_load_mode') != 'weights_only':
            raise ValueError(
                'Fine-tuned-body ablation must branch with weights_only'
            )
        if DIFFUSION_PRETRAIN not in load_path:
            raise ValueError(
                'Fine-tuned-body ablation must start from the 10% diffusion '
                'fine-tune last.ckpt'
            )
        if recal.get('bootstrap_on_start') is not True:
            raise ValueError(
                'Fine-tuned-body ablation must fit a fresh OmniFold stack'
            )
        if Path(load_path).resolve().parent == Path(save_path).resolve():
            raise ValueError(
                'Fine-tuned-body ablation must not overwrite its source'
            )
    else:
        if recal.get('refit_once_on_resume'):
            raise ValueError('Parent control must not refit_once_on_resume; resume restores the saved stack')
        if dgpo.get('checkpoint_load_mode') != 'weights_only':
            raise ValueError('First start is weights_only; later launches auto-resume last.ckpt')
        if DIFFUSION_PRETRAIN not in load_path:
            raise ValueError('Cold start must load diffusion_pretrain_10pct last.ckpt')
        if recal.get('bootstrap_on_start') is not True:
            raise ValueError('Cold start must bootstrap the initial OmniFold stack')
        if H4_PARITY_OUTPUT_TAG in save_path:
            raise ValueError('Parent control overlay must not use the H4 parity output tree')
    training_opts = config['options']['Training']
    if float(training_opts.get('learning_rate', 0.0)) != 0.0001:
        raise ValueError('Policy learning_rate must stay 1e-4 like nc4shnpg')
    platform = config['platform']
    if STIC_TRAIN not in str(platform['data_parquet_dir']):
        raise ValueError('Train pool must be the STIC-filtered 10% train shard')
    if HELD_OUT_VAL not in str(platform['data_parquet_val_dir']):
        raise ValueError('Score pool must use the held-out diffusion_val_20pct panel')
    wb = config['logger']['wandb']
    if wb.get('fresh_run') is not False or wb.get('resume') != 'allow':
        raise ValueError('W&B must use resume=allow and fresh_run=false for preempt continuity')
    if wb.get('id') != expected_wandb:
        raise ValueError(f'Expected fixed W&B id {expected_wandb}')
    if not dgpo.get('ztautau_metrics', {}).get('log_images', False):
        raise ValueError('Enable ztautau_metrics.log_images for physics monitoring')
    assert_nersc_16gpu(config)


def assert_live_ray_16gpu(*, expected_gpus: int = EXPECTED_WORKERS) -> None:
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


def resolve_launch_mode(config: dict, *, check_filesystem: bool) -> str:
    """Resolve a safe first/fork/body-ablation start or a complete resume."""
    training = config['options']['Training']
    output = Path(training['model_checkpoint_save_path'])
    last = output / 'last.ckpt'
    h4_parity = _is_h4_parity_fork(config)
    finetuned_body = _is_finetuned_body_ablation(config)
    if not check_filesystem:
        print(
            'Filesystem checks skipped; NERSC will auto-resume last.ckpt when present.',
            flush=True,
        )
        return 'unchecked'
    if last.is_file():
        payload = _load_ckpt(last)
        missing = [key for key in RECOVERY_KEYS if key not in payload]
        if missing:
            raise ValueError(
                f'Incomplete resume checkpoint {last}: missing {missing}. '
                'Remove it only if bootstrap never finished; otherwise fix the snapshot.'
            )
        print(
            f'Resume {last}\n'
            f'Completed DGPO steps: {payload.get("global_step")}\n'
            f'Next epoch: {payload.get("dgpo_next_epoch")}; '
            f'within-epoch progress: {payload.get("dgpo_epoch_step", 0)}\n'
            f'Reward round: {payload.get("dgpo_reward_round_id")}\n'
            f'Output: {output}',
            flush=True,
        )
        return 'resume'
    if output.exists() and any(output.glob('*.ckpt')):
        raise FileExistsError(
            f'Output has checkpoints but no last.ckpt: {output}; '
            'refusing a mixed first-start'
        )
    if finetuned_body:
        source = Path(training['model_checkpoint_load_path'])
        if not source.is_file():
            raise FileNotFoundError(
                f'Fine-tuned-body ablation needs the 10% diffusion fine-tune '
                f'last.ckpt at {source}'
            )
        payload = _load_ckpt(source)
        if 'state_dict' not in payload:
            raise ValueError(
                f'Fine-tuned-body source has no policy state_dict: {source}'
            )
        print(
            f'Body-ablation start: copy policy and classifier Body weights from '
            f'{source}\nReset DGPO/reward state and fit fresh legacy OmniFold; '
            f'write new tree: {output}',
            flush=True,
        )
        return 'body_ablation_start'
    if h4_parity:
        source = Path(training['model_checkpoint_load_path'])
        if not source.is_file():
            raise FileNotFoundError(
                f'H4 parity fork needs the parent last.ckpt at {source}'
            )
        payload = _load_ckpt(source)
        missing = [key for key in RECOVERY_KEYS if key not in payload]
        if missing:
            raise ValueError(
                f'Incomplete parent checkpoint {source}: missing {missing}'
            )
        print(
            f'Fork-start: resume full state from parent {source}\n'
            f'Completed DGPO steps: {payload.get("global_step")}\n'
            f'Next epoch: {payload.get("dgpo_next_epoch")}; '
            f'within-epoch progress: {payload.get("dgpo_epoch_step", 0)}\n'
            f'Reward round: {payload.get("dgpo_reward_round_id")}\n'
            f'Write new tree (no parent overwrite): {output}',
            flush=True,
        )
        return 'fork_start'
    print(
        f'First start: cold weights from diffusion_pretrain_10pct, then OmniFold '
        f'bootstrap into {output} (saves recoverable last.ckpt before policy steps).',
        flush=True,
    )
    return 'first_start'


def verify_paths(config: dict, *, check_filesystem: bool) -> None:
    training = config['options']['Training']
    required = [
        training['model_checkpoint_load_path'],
        config['reward_config']['omnifold']['backbone_checkpoint'],
        config['platform']['data_parquet_dir'],
        config['platform']['data_parquet_val_dir'],
    ]
    if not check_filesystem:
        return
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    output = Path(training['model_checkpoint_save_path'])
    if output.resolve() == Path(training['model_checkpoint_load_path']).resolve().parent:
        raise ValueError('Output must not overwrite the diffusion pretrain directory')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument(
        '--skip-filesystem-checks',
        action='store_true',
        help='CPU-only knob validation without NERSC paths.',
    )
    args = parser.parse_args(argv)
    overlay = args.config.resolve()
    config = read_overlay_yaml(overlay)
    assert_knobs(config)
    check_fs = not args.skip_filesystem_checks
    verify_paths(config, check_filesystem=check_fs)
    mode = resolve_launch_mode(config, check_filesystem=check_fs)
    runtime = build_runtime_config(
        base_config=BASE, overlay_config=overlay, backend='dgpo-evenet'
    )
    if check_fs:
        for section in read_yaml(runtime).values():
            if isinstance(section, dict) and isinstance(section.get('default'), str):
                if not Path(section['default']).is_file():
                    raise FileNotFoundError(
                        f"Missing config dependency: {section['default']}"
                    )
    print(f'Preflight passed: {overlay.name} ({mode})', flush=True)
    if _is_h4_parity_fork(config):
        print(
            'Phi-Fourier H4 parity fork: resume parent last.ckpt into a new '
            'output/W&B tree; proven nonlinear fusion and matched trainable '
            'scope from step 1; cold100/warm5 raw monitor epochs, '
            'cold1000/warm10 iteration-1 fits, 15-epoch patience, and '
            'five-epoch rejected-refit cooldown; no parent overwrite.',
            flush=True,
        )
    elif _is_finetuned_body_ablation(config):
        print(
            'Legacy classifier ablation: initialize the frozen Body, '
            'GroupedSequentialEmbedding, and InvisibleInputProjector from the '
            '10% diffusion fine-tune; discard TruthGeneration, then train '
            'the original PEFT/head scope and bootstrap a fresh reward stack.',
            flush=True,
        )
    else:
        print(
            'Old-method 10% knobs: soft trust=1.0, no hard boundary, fixed alpha=0.75, '
            'weighted-gap every 5 epochs, residuals 2..5 + BA<0.52, no inherit; '
            'probe/score panels 250k; auto-resume last.ckpt.',
            flush=True,
        )
    if args.check_only:
        return
    if not os.environ.get('WANDB_API_KEY'):
        print(
            'WARNING: WANDB_API_KEY is unset; W&B logging will fail after bootstrap.',
            flush=True,
        )
    if check_fs:
        assert_live_ray_16gpu()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/train_neutrino_backend.py'),
            '--backend',
            'dgpo-evenet',
            '--base-config',
            str(BASE),
            '--overlay-config',
            str(overlay),
            '--',
            '--ray-dir',
            config['nersc']['ray']['results_dir'],
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == '__main__':
    main()
