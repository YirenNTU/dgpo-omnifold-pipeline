"""Frozen conditional-tau ratio at the native DGPO (K,B,2,2) interface.

The head returns its training-time score: bounded log ratio by default, or an
explicitly freshly fitted unbounded BCE logit. Never unwrap old bounded weights.
The feature trunk is an independent raw1110 model, never the actor.
No truth, Cij, source identity, or previous-round score enters a candidate score.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from RL.DGPO_neutrino.rewards import BaseReward, apply_event_valid_to_rewards
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, pack_event_inputs


KIND = 'conditional_tau_bound30'
from RL.DGPO_neutrino.tau_process_labels import (
    POLICY_CONDITIONING_CONTRACT, SUPERVISED_POLICY_CONDITIONING_CONTRACT,
    canonical_policy_batch,
)

@contextmanager
def feature_precision():
    """Match the saved features without changing the actor's numerical setup."""
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old)


def validate_head(saved):
    if saved.get('general_spin_angles'):
        from .tau_combined_reward import validate_combined_head
        validate_combined_head(saved)
    if saved.get('topology_branch'):
        raise ValueError('Topology branch is a classifier-only experiment, not an installable DGPO reward')
    if saved.get('physics_branch'):
        raise ValueError('Physics branch is a classifier-only experiment, not an installable DGPO reward')
    required = dict(head_kind='film', head_depth=3, condition_width=256,
                    condition_hidden=256, relative_dim=6)
    for key, value in required.items():
        if saved.get(key) != value:
            raise ValueError(f'Expected baseline tau head {key}={value}')
    bound = saved.get('ratio_bound', 'missing')
    if isinstance(bound, bool) or bound not in (30, None):
        raise ValueError('Tau reward requires explicit ratio_bound 30 or None')
    if bound is None and (saved.get('fitting_current_policy') is not True
            or type(saved.get('source_policy_step')) is not int
            or saved['source_policy_step'] < 0):
        raise ValueError('Unbounded reward requires a freshly BCE-fitted policy denominator; do not unwrap old bounded weights')
    if saved.get('explicit_input') or saved.get('ratio_objective', 'bce') != 'bce':
        raise ValueError('Only the saved paired-BCE baseline is supported')
    if 'relative_preprocessing' not in saved or 'state_dict' not in saved:
        raise ValueError('Incomplete classifier checkpoint')
    if 'classification' in saved['packing_spec']['shapes']:
        from scripts.conditional_tau_preprocessing import layout
        _, width = layout(saved['packing_spec'])
        if (saved.get('policy_conditioning_contract') != SUPERVISED_POLICY_CONDITIONING_CONTRACT
                or saved.get('process_label_conditioning') is not True
                or saved['packing_spec']['shapes']['classification'] not in ([], [1])
                or type(saved.get('num_process_classes')) is not int or saved['num_process_classes'] < 1
                or saved['condition_dim'] != width+16+saved['num_process_classes']):
            raise ValueError('Process-conditioned head requires supervised labels plus decay/process one-hot inputs')
    if saved.get('cross_attention', False) and saved.get('attention_heads', 4) != 4:
        raise ValueError('Tau attention reward requires four heads')


def condition_inputs(packed, category, mean, scale, spec):
    from scripts.conditional_tau_preprocessing import apply_masked_feature
    category = np.asarray(category).reshape(-1)
    if not np.isfinite(category).all() or not np.equal(category, category.astype(int)).all():
        raise ValueError('Invalid visible decay category')
    category = category.astype(int)
    if not np.isin(category//10, [1, 2, 3, 4]).all() or not np.isin(category % 10, [1, 2, 3, 4]).all():
        raise ValueError('Missing/invalid visible decay category; no inferred fallback')
    onehot = np.eye(16, dtype=np.float32)[(category//10-1)*4 + category % 10-1]
    raw = np.concatenate((np.asarray(packed, dtype=np.float32), onehot), axis=1)
    if 'classification' in spec['shapes']:
        from scripts.conditional_tau_preprocessing import layout
        spans, _ = layout(spec)
        label = np.asarray(packed)[:, spans['classification'][0]].reshape(-1)
        num_classes = len(mean)-np.asarray(packed).shape[1]-16
        if (len(label) != len(raw) or not np.isfinite(label).all()
                or not np.equal(label, label.astype(int)).all()
                or num_classes < 1 or (label < 0).any() or (label >= num_classes).any()):
            raise ValueError('Invalid packed supervised process labels')
        raw = np.concatenate((raw, np.eye(num_classes, dtype=np.float32)[label.astype(int)]), axis=1)
    return apply_masked_feature(raw, np.asarray(mean), np.asarray(scale), spec)


def candidate_coordinates(visible_a, visible_b, deltas, stats):
    from scripts.diagnose_ztautau_cij import tau_from_deltas
    from scripts.train_conditional_spin_ratio import tau_features
    from scripts.tau_relative_inputs import relative_angles
    a = tau_from_deltas(visible_a, deltas[:, 0])
    b = tau_from_deltas(visible_b, deltas[:, 1])
    relative = (relative_angles(a, b, visible_a, visible_b)-np.asarray(stats['mean']))/np.asarray(stats['scale'])
    return np.concatenate((relative, tau_features(a, b)), axis=1).astype('float32')


class ConditionalTauReward(BaseReward):
    def __init__(self, bundle, backbone, device, microbatch=256):
        from scripts.train_conditional_spin_ratio import build_classifier
        if bundle.get('kind') != KIND or bundle.get('schema_version') != 1:
            raise ValueError('Unknown conditional tau reward bundle')
        validate_head(bundle['head'])
        self.bundle = copy.deepcopy(bundle)
        self.installed_head = copy.deepcopy(bundle['head'])
        self.device, self.microbatch = torch.device(device), int(microbatch)
        if self.microbatch < 1:
            raise ValueError('Feature microbatch must be positive')
        self.spec = EventPackingSpec.from_dict(bundle['head']['packing_spec'])
        self.policy_conditioning_contract = bundle.get('policy_conditioning_contract', POLICY_CONDITIONING_CONTRACT)
        self.num_process_classes = int(bundle['head'].get('num_process_classes',19))
        has_labels = 'classification' in self.spec.shapes
        if has_labels != (self.policy_conditioning_contract == SUPERVISED_POLICY_CONDITIONING_CONTRACT):
            raise ValueError('Reward packing and policy process-label contract differ; refit the classifier')
        self.backbone = backbone.to(device).eval().requires_grad_(False)
        self.head = build_classifier(bundle['head']).to(device)
        self.head.load_state_dict(bundle['head']['state_dict'], strict=True)
        self.head.eval().requires_grad_(False)
        self.round_id = 0
        self.denominator_step = 0
        self.last_refit_epoch = -1
        self.last_evaluation_epoch = -2
        self.adaptive_oracle_state = None

    @property
    def name(self):
        return 'conditional_tau'

    def checkpoint_metadata(self):
        metadata = dict(kind=KIND, source_run=self.bundle['source_run'],
                    policy_conditioning_contract=self.policy_conditioning_contract,
                    source_checkpoint=self.bundle['source_checkpoint'],
                    reward_round_id=self.round_id, denominator_step=self.denominator_step,
                    ratio_bound=self.installed_head['ratio_bound'], feature_trunk=self.bundle.get('feature_trunk', 'frozen_raw1110'),
                    reward='unbounded_log_ratio' if self.installed_head['ratio_bound'] is None else 'bounded_log_ratio')
        diagnostic = getattr(self, '_tau_diagnostic_ensemble', None)
        if diagnostic is not None:
            metadata.update(diagnostic_teacher=diagnostic['selection'],
                diagnostic_teacher_members=4 if diagnostic['selection'] == 'ensemble' else 1,
                diagnostic_generator_checkpoint=diagnostic['artifact']['generator_checkpoint'],
                diagnostic_aggregation='mean_training_time_bounded_log_ratio'
                    if diagnostic['selection'] == 'ensemble' else 'member0_training_time_bounded_log_ratio',
                diagnostic_reference_recentered=False)
        return metadata

    def stack_payload(self):
        diagnostic = None
        if getattr(self, '_tau_diagnostic_ensemble', None) is not None:
            from RL.DGPO_neutrino.tau_classifier_ensemble_probe import diagnostic_stack_payload
            diagnostic = diagnostic_stack_payload(self)
        saved = dict(kind=KIND, schema_version=1, source_run=self.bundle['source_run'],
                    policy_conditioning_contract=self.policy_conditioning_contract,
                    source_checkpoint=self.bundle['source_checkpoint'],
                    condition_mean=self.bundle['condition_mean'],
                    condition_scale=self.bundle['condition_scale'],
                    normalization_file=self.bundle['normalization_file'],
                    head=diagnostic['head'] if diagnostic is not None else {**self.installed_head, 'state_dict':
                          {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()}},
                    round_id=self.round_id, denominator_step=self.denominator_step,
                    last_refit_epoch=self.last_refit_epoch,
                    last_evaluation_epoch=self.last_evaluation_epoch)
        if getattr(self, 'adaptive_oracle_state', None) is not None:
            saved['adaptive_oracle_state'] = copy.deepcopy(self.adaptive_oracle_state)
        if diagnostic is not None:
            saved['diagnostic_ensemble'] = diagnostic['diagnostic_ensemble']
        return saved

    def load_stack_payload(self, saved):
        if saved.get('kind') != KIND or saved.get('schema_version') != 1:
            raise ValueError('Cannot resume H4/cascade state as a tau-ratio reward')
        if saved.get('policy_conditioning_contract') != self.policy_conditioning_contract:
            raise ValueError('Cannot resume pre-fix or label0 tau DGPO with a different classification contract; start a fresh experiment')
        for key in ('source_run', 'source_checkpoint'):
            if saved[key] != self.bundle[key]:
                raise ValueError('Resume tau reward source differs: '+key)
        for key in ('condition_mean', 'condition_scale'):
            # Full checkpoints are mapped to each worker's CUDA device, whereas
            # the immutable reward bundle is loaded on CPU. Compare values on
            # CPU without changing either source or relaxing exact equality.
            if key not in saved or not torch.equal(
                    torch.as_tensor(saved[key], device='cpu'),
                    torch.as_tensor(self.bundle[key], device='cpu')):
                raise ValueError('Resume condition normalization differs: '+key)
        if saved.get('normalization_file') != self.bundle['normalization_file']:
            raise ValueError('Resume pinned normalization file changed')
        validate_head(saved['head'])
        if saved['head'].get('general_spin_angles') != self.bundle['head'].get('general_spin_angles'):
            raise ValueError('Resume combined feature schema changed')
        for key in ('packing_spec', 'relative_preprocessing', 'condition_dim', 'candidate_dim'):
            if 'diagnostic_ensemble' in saved:
                from RL.DGPO_neutrino.tau_classifier_ensemble_probe import _equal
                matches = _equal(saved['head'][key], self.bundle['head'][key])
            else:
                matches = saved['head'][key] == self.bundle['head'][key]
            if not matches:
                raise ValueError('Resume preprocessing/architecture changed: '+key)
        if 'diagnostic_ensemble' in saved:
            from RL.DGPO_neutrino.tau_classifier_ensemble_probe import restore_diagnostic_reward
            restore_diagnostic_reward(self, saved)
            return
        self._replace_head(saved['head'])
        self._clear_diagnostic_ensemble()
        for key in ('round_id', 'denominator_step', 'last_refit_epoch', 'last_evaluation_epoch'):
            setattr(self, key, int(saved[key]))
        self.installed_head = copy.deepcopy(saved['head'])
        self.adaptive_oracle_state = copy.deepcopy(saved.get('adaptive_oracle_state'))

    def install(self, head, *, policy_step, epoch):
        if head.get('general_spin_angles') != self.bundle['head'].get('general_spin_angles'):
            raise ValueError('Cannot replace a reward with a different combined feature schema')
        if head.get('general_spin_angles') and head.get('source_policy_step') != int(policy_step):
            raise ValueError('Combined head denominator step differs from install step')
        validate_head(head)
        if head['packing_spec'] != self.bundle['head']['packing_spec']:
            raise ValueError('Cannot install a classifier with different process-label inputs')
        self._replace_head(head)
        self._clear_diagnostic_ensemble()
        self.round_id += 1
        self.denominator_step = int(policy_step)
        self.last_refit_epoch = int(epoch)

    def _clear_diagnostic_ensemble(self):
        for key in ('_tau_diagnostic_ensemble', '_tau_diagnostic_ensemble_owner'):
            self.__dict__.pop(key, None)

    def _replace_head(self, saved):
        from scripts.train_conditional_spin_ratio import build_classifier
        from scripts.tau_fresh_negatives import isolated_rng
        # Build transactionally, including attention modules on resumed rewards.
        with isolated_rng(self.device, 42):
            head = build_classifier(saved).to(self.device)
            head.load_state_dict(saved['state_dict'], strict=True)
        self.head = head.eval().requires_grad_(False)
        self.installed_head = copy.deepcopy(saved)

    @torch.no_grad()
    def prepare_inputs(self, batch):
        """Candidate-independent, observable-only inputs; scoped to one batch."""
        batch = canonical_policy_batch(batch, contract=self.policy_conditioning_contract,num_classes=self.num_process_classes)
        packed, _ = pack_event_inputs(batch, self.spec)
        category = batch['event_category'].detach().cpu().numpy()
        c = condition_inputs(packed.cpu().numpy(), category,
            self.bundle['condition_mean'], self.bundle['condition_scale'], self.spec.to_dict())
        visible = [torch.stack([batch[f'lead_{leg}_visible_{key}'].reshape(-1)
                    for key in ('E', 'px', 'py', 'pz')], -1) for leg in ('a', 'b')]
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
        return dict(packed=packed, condition=c, condition_tensor=torch.as_tensor(c, device=self.device),
            visible=visible, visible_numpy=[v.cpu().numpy() for v in visible],
            trunk_batch=canonical_policy_batch(unpack_event_inputs(packed.to(self.device), self.spec),
                                              contract=self.policy_conditioning_contract,num_classes=self.num_process_classes))

    @torch.no_grad()
    def features(self, batch, delta, *, prepared=None):
        from scripts.tau_backbone_alignment import candidate_hidden
        prepared = self.prepare_inputs(batch) if prepared is None else prepared
        packed, c, visible = prepared['packed'], prepared['condition'], prepared['visible_numpy']
        coordinate = candidate_coordinates(*visible, delta.detach().cpu().numpy(),
                                            self.bundle['head']['relative_preprocessing'])
        hidden = []
        # Only the ORIGINAL packed observable inputs reach the frozen trunk.
        # x_invisible and the truth/generated BCE target never enter the trunk;
        # the observed process classification is preserved in packed inputs.
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
        with feature_precision():
            for start in range(0, len(delta), self.microbatch):
                stop = start+self.microbatch
                sub = canonical_policy_batch(unpack_event_inputs(packed[start:stop].to(self.device), self.spec),
                                             contract=self.policy_conditioning_contract,num_classes=self.num_process_classes)
                hidden.append(candidate_hidden(self.backbone, sub, delta[start:stop].to(self.device)))
        f = np.concatenate((np.concatenate(hidden), coordinate), axis=1).astype('float32')
        if self.bundle['head'].get('general_spin_angles'):
            from scripts.tau_general_spin_angles import features
            protocol = self.bundle['head']['general_spin_angles']['spin_protocol']
            f = np.concatenate((f, features(*visible, delta.detach().cpu().numpy(), protocol=protocol)), axis=1)
        if f.shape[1] != self.bundle['head']['candidate_dim'] or not np.isfinite(f).all():
            raise ValueError('Candidate feature contract changed')
        return c, f

    @torch.no_grad()
    def compute(self, candidates, batch, mask=None):
        if (self.policy_conditioning_contract == SUPERVISED_POLICY_CONDITIONING_CONTRACT
                and not self.installed_head.get('fitting_current_policy', False)):
            raise ValueError('Fit the process-conditioned classifier before using its reward')
        if candidates.ndim != 4 or candidates.shape[-2:] != (2, 2):
            raise ValueError('Conditional tau reward expects (K,B,2,2) angular deltas')
        scores = []
        prepared = self.prepare_inputs(batch)
        for delta in candidates:
            c, f = self.features(batch, delta, prepared=prepared)
            with feature_precision():
                scores.append(self.head(torch.as_tensor(c, device=self.device),
                                        torch.as_tensor(f, device=self.device)))
        result = torch.stack(scores)
        bound = self.installed_head['ratio_bound']
        if not torch.isfinite(result).all() or (bound is not None and result.max() > np.log(bound)+1e-5):
            raise ValueError('Nonfinite tau reward or score above the declared ratio bound')
        if mask is not None:
            result = result*mask.to(result)
        return apply_event_valid_to_rewards(result, batch)


def load_reward(path, device, microbatch=256):
    from evenet.control.global_config import Config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from scripts.diagnose_h4_spike_coverage import load_raw_state
    from scripts.tau_fresh_negatives import isolated_rng
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    # Do not reload the process-global actor configuration!
    config = Config()
    config.load_yaml(bundle['backbone_runtime'])
    with isolated_rng(device, 42):
        policy = build_evenet_on_device(config, load_normalization_dict(config), device)
        raw = torch.load(bundle['source_checkpoint'], map_location='cpu', weights_only=False)
        expected_step = int(bundle.get('source_global_step', 1110))
        if int(raw['global_step']) != expected_step:
            raise ValueError('Classifier trunk checkpoint step differs from the pinned source')
        if bundle.get('policy_conditioning_contract') == SUPERVISED_POLICY_CONDITIONING_CONTRACT:
            from scripts.evaluate_diffusion_pretrain_cij import strict_load
            strict_load(policy, raw)
        else:
            load_raw_state(policy, raw, 'step1110')
        reward = ConditionalTauReward(bundle, policy, device, microbatch)
    return reward
