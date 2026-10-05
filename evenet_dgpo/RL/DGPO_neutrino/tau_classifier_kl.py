"""Balanced current/anchor log-ratio in the reward's unchanged feature space.

The anchor is fixed; only the independent KL head is refreshed. Finite fitted
ratios approximate feature-space KL, not a certified full conditional KL.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.tau_classifier_kl import validate_classifier_kl


class ZeroLogRatio(torch.nn.Module):
    def forward(self, condition, features):
        return features.sum(-1) * 0


def paired_kl_panel(panel, current_features, anchor_features):
    if (current_features.shape != anchor_features.shape
            or current_features.shape != panel['candidate_truth'].shape
            or not np.isfinite(current_features).all() or not np.isfinite(anchor_features).all()):
        raise ValueError('Require one finite current and anchor feature vector for every conditioning event')
    # Positive BCE class = CURRENT, negative BCE class = ANCHOR. Same split,
    # condition and event weight in both classes; no class-prior correction.
    positive = dict(panel, candidate_truth=current_features)
    negative = dict(features=anchor_features)
    return positive, negative


class TauClassifierKL:
    def __init__(self, cycle, cfg, folder, *, anchor_directory):
        self.cycle, self.cfg = cycle, validate_classifier_kl(cfg)
        self.folder, self.anchor_directory = Path(folder), Path(anchor_directory)
        self.anchor = self.source = None
        self.fit_step = None
        self.report = None
        self.pending_fit = None

    def _load_anchor(self):
        panel = self.cycle.train
        if (panel is None or len(panel['source_ids']) != 416701
                or len(self.cycle.validation['source_ids']) != 119002
                or np.intersect1d(panel['source_ids'], self.cycle.validation['source_ids']).size):
            raise ValueError('Classifier KL requires the complete disjoint filtered panels')
        anchor = np.empty_like(panel['candidate_truth'])
        seen = np.zeros(len(anchor), dtype=int)
        for rank in range(self.cycle.world):
            with np.load(self.anchor_directory / f'rank-{rank:02d}.npz', allow_pickle=False) as f:
                positions = f['positions']
                if positions.ndim != 1 or (positions < 0).any() or (positions >= len(anchor)).any():
                    raise ValueError('Invalid anchor shard positions')
                np.add.at(seen, positions, 1)
                anchor[positions] = f['features']
        if not np.all(seen == 1) or not np.isfinite(anchor).all():
            raise ValueError('Anchor features must cover each train event exactly once')
        self.anchor = anchor

    def fit_candidate(self, step, folder, *, log_step=None):
        """Fit the actual tentative actor; never install it into either reward."""
        from RL.DGPO_neutrino.checkpoint_transfer import isolated_evaluation
        if self.anchor is None:
            self._load_anchor()
        with isolated_evaluation(self.cycle.actor, self.cycle.cfg['refit_seed'] + self.cycle.rank):
            current = self.cycle.generate(self.cycle.train, folder / 'current-candidates', 1,
                self.cycle.cfg['refit_seed'] + self.cfg['anchor_step'], False)
        positive, negative = paired_kl_panel(self.cycle.train, current['features'], self.anchor)
        model, saved, status, _ = self.cycle.fit(positive, negative, folder / 'classifier',
            audit=False, step=step, reward_ratio_bound=None, min_selected_steps=1000,
            log_phase='classifier_kl_fit' if log_step is None else 'classifier_trust_fit',
            log_step=log_step)
        if (not status['minimum_fit_steps_met'] or status['total_steps'] < max(1000, saved['min_steps'])
                or status['best_steps'] < 1000
                or saved['ratio_bound'] is not None or saved['source_policy_step'] != step):
            raise ValueError('Current/anchor KL head did not satisfy the unbounded fit contract')
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('Nonfinite fitted classifier KL head')
        return model.eval().requires_grad_(False), saved, status

    def refresh(self, step):
        from RL.DGPO_neutrino.conditional_tau_cycle import barrier
        if self.anchor is None:
            self._load_anchor()
        anchor_step = self.cfg['anchor_step']
        if (step < anchor_step or (self.fit_step is None and step != anchor_step)
                or (self.fit_step is not None and step != self.fit_step + 1)):
            raise ValueError('Classifier KL requires sequential fresh fits beginning at step1920')
        folder = self.folder / f'policy-step-{step:08d}'
        folder.mkdir(parents=True, exist_ok=True)
        status = None
        if step == anchor_step:
            # Current == anchor is established by the unchanged startup actor.
            # The exact density ratio is one; fitting a noisy nonzero head here
            # would invent a reference gradient at the anchor.
            model = ZeroLogRatio().to(self.cycle.device).eval().requires_grad_(False)
            saved = dict(self.cycle.reward.installed_head, ratio_bound=None,
                         source_policy_step=step, state_dict={})
            exact_zero = True
        else:
            # Same conditioning rows and fixed generation noise as the startup
            # anchor. Sampling/fitting never advances actor optimizer or RNG.
            if self.pending_fit is not None:
                pending_step, model, saved, status = self.pending_fit
                if pending_step != step:
                    raise ValueError('Accepted trust head must match the next current policy step')
                self.pending_fit = None
            else:
                model, saved, status = self.fit_candidate(step, folder)
            exact_zero = False
        # Independent head; share only the frozen observable feature extractor.
        # Never call reward.install(), alter reward rounds or recenter reference.
        source = copy.copy(self.cycle.reward)
        source.head = model.eval().requires_grad_(False)
        source.installed_head = saved
        self.source, self.fit_step = source, step
        self.report = dict(anchor_step=anchor_step, current_policy_step=step,
            coefficient=1, exact_zero_at_anchor=exact_zero, fit=status,
            positive_class='current_policy', negative_class='fixed_step1920_anchor',
            current_events=len(self.anchor), anchor_events=len(self.anchor),
            ratio_bound=None, class_priors_equal=True, paired_conditions_weights_and_splits=True,
            feature_map='Same frozen raw1110/context/coordinate map as reward',
            scope='Estimated feature-space KL; not exact full conditional KL or hard trust region')
        if self.cycle.rank == 0:
            (folder / 'kl_fit.json').write_text(json.dumps(self.report, indent=2) + '\n')
            self.cycle.emit({'tau/classifier_kl/current_policy_step': step,
                'tau/classifier_kl/anchor_step': anchor_step,
                'tau/classifier_kl/coefficient': 1,
                'tau/classifier_kl/exact_zero_at_anchor': int(exact_zero),
                'tau/classifier_kl/velocity_penalty_enabled': 0}, step)
        barrier(self.cycle.world)
        return self.source

    def checkpoint_payload(self):
        if self.source is None:
            raise ValueError('Classifier KL checkpoint requested before initialization')
        return dict(report=self.report,
            head={**self.source.installed_head, 'state_dict':
                {k: v.detach().cpu().clone() for k, v in self.source.head.state_dict().items()}},
            anchor_directory=str(self.anchor_directory),
            scope='Head fitted at current_policy_step before the saved actor update; refit before reuse')
