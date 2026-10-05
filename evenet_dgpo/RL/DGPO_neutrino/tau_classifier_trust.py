"""Transactional post-proposal gate on held-out estimated feature-space KL.

The acceptance boundary is hard for this estimator, not a certificate of the
full conditional KL. Confidence bounds describe sampling error conditional on
the fitted critic; classifier bias is not covered.
"""
import copy
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
import torch.distributed as dist

from scripts.tau_classifier_kl import validate_classifier_trust
from RL.DGPO_neutrino.optimizer_transaction import (
    assign_scaled_trainable_update_, snapshot_adamw_state_for_transaction,
    restore_adamw_state_from_transaction_,
)
from RL.DGPO_neutrino.projection_cpo import snapshot_params, assign_params_
from RL.DGPO_neutrino.conditional_tau_reward import feature_precision


def kl_confidence_bound(scores, weights, *, confidence, trials):
    """One draw per independent event; never clamp signed classifier logits."""
    scores, weights = np.asarray(scores, dtype=float), np.asarray(weights, dtype=float)
    if (scores.ndim != 1 or scores.shape != weights.shape or len(scores) < 2
            or not np.isfinite(scores).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('KL gate requires finite per-event logits and nonnegative weights')
    w = weights / weights.max()
    w /= w.sum()
    mean = float(np.sum(w*scores))
    neff = float(1 / np.sum(w*w))
    if neff <= 1:
        raise ValueError('KL gate needs more than one effective event')
    se = float(np.sqrt(np.sum(w*w*(scores-mean)**2) * neff/(neff-1)))
    # Bonferroni across the predeclared line-search grid within ONE proposal.
    # This is a large-sample normal approximation, not a finite-sample theorem.
    z = NormalDist().inv_cdf(1-(1-confidence)/trials)
    upper = mean + z*se
    if not all(math.isfinite(v) for v in (mean, se, upper, neff)):
        raise ValueError('Nonfinite classifier KL confidence bound')
    return dict(mean=mean, standard_error=se, upper=upper, effective_events=neff,
                critical_value=z, valid_nonnegative_kl=upper >= 0)


class TauClassifierTrust:
    def __init__(self, critic, cfg, folder):
        self.critic, self.cycle = critic, critic.cycle
        self.cfg, self.folder = validate_classifier_trust(cfg), Path(folder)
        self.report = None
        self.accepted_step = self.cfg['anchor_step']
        self.last_accepted_estimate = dict(mean=0., upper=0., exact_zero_at_anchor=True)
        if self.cfg['validation_seed'] == self.cycle.cfg['refit_seed'] + self.cfg['anchor_step']:
            raise ValueError('Trust validation noise must be independent of classifier fitting noise')

    def _evaluate(self, fitted, folder, step):
        from RL.DGPO_neutrino.checkpoint_transfer import isolated_evaluation
        model = fitted[0]
        panel = self.cycle.validation
        # Full filtered validation, disjoint from the classifier's training and
        # selection rows. No truth labels or truth candidates enter this gate.
        if (len(panel['source_ids']) != 119002
                or np.intersect1d(panel['source_ids'], self.cycle.train['source_ids']).size):
            raise ValueError('Trust gate requires complete disjoint filtered validation')
        seed = self.cfg['validation_seed'] + (step-self.cfg['anchor_step'])*100003
        with isolated_evaluation(self.cycle.actor, seed+self.cycle.rank):
            generated = self.cycle.generate(panel, folder/'held-out-candidates', 1, seed, False)
        ii = np.arange(self.cycle.rank, len(panel['source_ids']), self.cycle.world)
        # Evaluate only CURRENT, once. There is no truth/anchor candidate score
        # in E_current log(current/anchor), and no second head forward is needed.
        scores = []
        with torch.no_grad(), feature_precision():
            for start in range(0, len(ii), self.cycle.cfg['fit']['batch_size']):
                take = ii[start:start+self.cycle.cfg['fit']['batch_size']]
                c = torch.as_tensor(panel['condition'][take], device=self.cycle.device)
                f = torch.as_tensor(generated['features'][take], device=self.cycle.device)
                scores.append(model(c, f).float().cpu().numpy())
        scores = np.concatenate(scores)
        pieces = [None]*self.cycle.world if self.cycle.rank == 0 else None
        if self.cycle.world > 1:
            dist.gather_object((ii, scores), pieces, dst=0)
        else:
            pieces = [(ii, scores)]
        outcome = [None, None]
        if self.cycle.rank == 0:
            try:
                all_scores = np.empty(len(panel['source_ids']), dtype=float)
                seen = np.zeros(len(all_scores), dtype=int)
                for positions, values in pieces:
                    np.add.at(seen, positions, 1)
                    all_scores[positions] = values
                if not np.all(seen == 1):
                    raise ValueError('Trust scores must cover each conditioning event exactly once')
                stats = kl_confidence_bound(all_scores, panel['event_weight'],
                    confidence=self.cfg['confidence'], trials=self.cfg['max_backtracks']+1)
                stats['accepted'] = bool(stats['valid_nonnegative_kl'] and stats['upper'] <= self.cfg['max_kl'])
                outcome[0] = stats
                np.savez(folder/'held_out_scores.npz', source_ids=panel['source_ids'],
                         log_ratio=all_scores, weight=panel['event_weight'])
            except Exception as exc:
                outcome[1] = str(exc)
        if self.cycle.world > 1:
            dist.broadcast_object_list(outcome, src=0)
        if outcome[1] is not None:
            raise ValueError(f'Invalid classifier trust estimate: {outcome[1]}')
        return outcome[0]

    def _publish(self, folder):
        error = [None]
        if self.cycle.rank == 0:
            try:
                target = folder/'trust_proposal.json'
                pending = target.with_suffix('.pending.json')
                pending.write_text(json.dumps(self.report, indent=2, allow_nan=False)+'\n')
                pending.replace(target)
            except Exception as exc:
                error[0] = str(exc)
        if self.cycle.world > 1:
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise OSError(f'Cannot publish classifier trust decision: {error[0]}')

    def propose(self, model, optimizer, step):
        """One AdamW moment update; shrink its displacement, or restore it all.

        Scheduler/EMA are deliberately outside this transaction and advance
        only after the returned acceptance flag. A rejected line search ends
        the pilot at its last accepted policy, without spending update budget.
        """
        if self.critic.fit_step != step or step != self.accepted_step:
            raise ValueError('Trust proposal requires the incumbent policy KL head')
        old = snapshot_params(model)
        buffers = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        moments = snapshot_adamw_state_for_transaction(optimizer)
        previous_pending = self.critic.pending_fit
        folder = self.folder/f'from-step-{step:08d}'
        folder.mkdir(parents=True, exist_ok=True)
        self.report = dict(complete=False, accepted=False, incumbent_policy_step=step,
            accepted_policy_step=step, anchor_step=self.cfg['anchor_step'], max_kl=self.cfg['max_kl'],
            trials=[], accepted_scale=0., optimizer_state_restored=False,
            scope='Hard gate on classifier-estimated feature-space mean KL; not exact full conditional KL',
            uncertainty='Large-sample event SE conditional on each fitted head; Bonferroni over this line search only; excludes classifier bias and adaptation across updates',
            validation_events=119002, validation_candidates=1, confidence=self.cfg['confidence'],
            classifier_training_events=416701, minimum_selected_fit_steps=1000,
            reference='fixed_step1920_anchor', coefficient=1, velocity_penalty_enabled=False,
            incumbent_estimate=copy.deepcopy(self.last_accepted_estimate))
        accepted = False
        try:
            optimizer.step()
            candidate = snapshot_params(model)
            finite = torch.tensor(int(all(torch.isfinite(v).all() for v in candidate.values())),
                                  device=self.cycle.device)
            if self.cycle.world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError('Nonfinite AdamW classifier-trust proposal')
            for attempt in range(self.cfg['max_backtracks']+1):
                scale = self.cfg['backtrack_factor']**attempt
                # Write even alpha=1 before fitting: exactly the point tested is
                # retained on acceptance, with no post-gate parameter mutation.
                assign_scaled_trainable_update_(model, old, candidate, scale)
                trial = folder/f'trial-{attempt:02d}'
                trial.mkdir(parents=True, exist_ok=True)
                fitted = self.critic.fit_candidate(step+1, trial, log_step=step)
                stats = self._evaluate(fitted, trial, step)
                self.report['trials'].append(dict(attempt=attempt, scale=scale, **stats,
                    classifier_fit=fitted[2]))
                if self.cycle.rank == 0:
                    self.cycle.emit({f'tau/classifier_trust/trial/{k}':v for k,v in
                        dict(attempt=attempt, scale=scale, mean=stats['mean'], upper=stats['upper'],
                             accepted=int(stats['accepted'])).items()}, step)
                if stats['accepted']:
                    # This head describes the exact accepted policy with the
                    # same fit contract/noise. Reuse it at the next update.
                    self.report.update(accepted=True, accepted_policy_step=step+1,
                                       accepted_scale=scale, accepted_estimate=stats)
                    break
            self.report['complete'] = True
            if self.report['accepted']:
                # Commit only after every rank has a successfully published
                # decision. A publication failure also restores the proposal.
                self._publish(folder)
                self.critic.pending_fit = (step+1, *fitted)
                self.accepted_step = step+1
                self.last_accepted_estimate = stats
                accepted = True
        except Exception as exc:
            self.report['error'] = str(exc)
            raise
        finally:
            if not accepted:
                assign_params_(model, old)
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(buffers[name].to(value.device))
                restore_adamw_state_from_transaction_(optimizer, moments)
                self.critic.pending_fit = previous_pending
                optimizer.zero_grad(set_to_none=True)
                self.report.update(optimizer_state_restored=True, accepted=False,
                                   accepted_policy_step=step, accepted_scale=0.)
                self._publish(folder)
        last = self.report['trials'][-1]
        metrics = {'tau/classifier_trust/enabled':1., 'tau/classifier_trust/max_kl':self.cfg['max_kl'],
            'tau/classifier_trust/accepted':float(accepted),
            'tau/classifier_trust/accepted_scale':self.report['accepted_scale'],
            'tau/classifier_trust/trials':len(self.report['trials']),
            'tau/classifier_trust/candidate_mean':last['mean'],
            'tau/classifier_trust/candidate_upper':last['upper'],
            'tau/classifier_trust/optimizer_state_restored':float(not accepted),
            '_tau_classifier_trust_stop':not accepted}
        if accepted:
            metrics['tau/classifier_trust/accepted_mean'] = last['mean']
            metrics['tau/classifier_trust/accepted_upper'] = last['upper']
        return accepted, metrics

    def checkpoint_payload(self):
        pending = self.critic.pending_fit
        return dict(report=copy.deepcopy(self.report), cfg=self.cfg,
            accepted_policy_step=self.accepted_step, retained_estimate=self.last_accepted_estimate,
            accepted_head=None if pending is None else
                {**pending[2], 'state_dict':{k:v.detach().cpu().clone() for k,v in pending[1].state_dict().items()}})
