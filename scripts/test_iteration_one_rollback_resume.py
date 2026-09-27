import copy
import unittest

from train_dgpo_iteration_one_rollback import CONFIG, ROOT, read_yaml, validate_payload
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    adaptive_audit_protocol_signature, resolve_adaptive_config,
)


class IterationOneRollbackResumeTests(unittest.TestCase):
    def setUp(self):
        self.c = read_yaml(CONFIG)
        self.d = self.c['dgpo']
        a = resolve_adaptive_config(self.d)
        self.p = dict(epoch=15, global_step=155, dgpo_next_epoch=15, dgpo_epoch_step=5,
                      state_dict={}, dgpo_checkpoint_version=1, dgpo_ref_state_dict={},
                      dgpo_round_ref_state_dict={}, dgpo_round_ref_sha256='test',
                      dgpo_omnifold_reward_stack={'test': True}, dgpo_omnifold_reward_metadata={},
                      dgpo_optimizer_state_dict={'optimizer': {}, 'scheduler': {},
                                                'lr_schedule': {'total_steps': 1500, 'min_lr_ratio': .1}},
                      dgpo_adaptive_omnifold_state={
                          'audit_protocol_signature': adaptive_audit_protocol_signature(a),
                          'raw_global_initialized': True, 'raw_best_global_step': 155,
                          'raw_best_epoch': 15, 'raw_best_next_epoch': 15,
                          'raw_best_auc_gap': .40751809708503384, 'raw_monitor_state': {'test': True},
                          'reward_round_id': 4, 'policy_warmup_round_id': 4,
                          'policy_warmup_completed_updates': 10,
                          'policy_warmup_protocol': {'steps': 10, 'start_factor': .1}})

    def test_valid_mid_epoch_resume(self):
        state = validate_payload(self.p, self.d, initial_branch=True)
        self.assertEqual((state.raw_best_global_step, state.raw_best_next_epoch), (155, 15))
        self.assertEqual(state.policy_warmup_completed_updates, 10)

    def test_wrong_step_or_epoch_rejected(self):
        for key, value in [('global_step', 160), ('dgpo_next_epoch', 16), ('dgpo_epoch_step', 0)]:
            p = copy.deepcopy(self.p); p[key] = value
            with self.assertRaises(ValueError):
                validate_payload(p, self.d, initial_branch=True)

    def test_protocol_mismatch_rejected(self):
        self.p['dgpo_adaptive_omnifold_state']['audit_protocol_signature'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            validate_payload(self.p, self.d, initial_branch=True)

    def test_missing_stack_or_optimizer_rejected(self):
        for key in ['dgpo_omnifold_reward_stack', 'dgpo_optimizer_state_dict']:
            p = copy.deepcopy(self.p); del p[key]
            with self.assertRaises(ValueError):
                validate_payload(p, self.d, initial_branch=True)

    def test_continuation_can_use_later_branch_last(self):
        self.p.update(epoch=16, global_step=170, dgpo_next_epoch=17, dgpo_epoch_step=0)
        validate_payload(self.p, self.d, initial_branch=False)

    def test_reward_and_audit_protocols_preserved(self):
        old = read_yaml(ROOT/'config/dgpo_omnifold_ztautau_10pct_iteration1_only.yaml')['dgpo']
        old_a, new_a = resolve_adaptive_config(old), resolve_adaptive_config(self.d)
        self.assertEqual(adaptive_audit_protocol_signature(old_a), adaptive_audit_protocol_signature(new_a))
        before, after = copy.deepcopy(old['adaptive_omnifold']['recalibration']), copy.deepcopy(self.d['adaptive_omnifold']['recalibration'])
        before.pop('bootstrap_on_start'); after.pop('bootstrap_on_start')
        self.assertEqual(before, after)
        self.assertEqual(old['lr_schedule'], self.d['lr_schedule'])
        self.assertEqual(old['reference_trust'], self.d['reference_trust'])
        self.assertFalse(new_a.refit_once_on_resume)
        self.assertTrue(new_a.raw_rollback_to_best_on_plateau)
        self.assertTrue(self.c['logger']['wandb']['fresh_run'])


if __name__ == '__main__':
    unittest.main()
