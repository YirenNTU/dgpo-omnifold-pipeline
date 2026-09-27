"""Bootstrap global best must survive later-round checkpoint restoration."""
import unittest

from .adaptive import AdaptiveOmniFoldState, initialize_global_raw_best


class TestGlobalBestResume(unittest.TestCase):
    def test_certified_bootstrap_best_is_not_replaced_by_current_round_history(self):
        payload = dict(raw_global_initialized=True, raw_best_auc_gap=.12,
            raw_best_epoch=-1, raw_best_global_step=0, raw_best_next_epoch=0,
            raw_best_checkpoint='/checkpoint/bootstrap.ckpt', reward_round_id=4,
            raw_no_improvement_streak=3, probe_history=[
                dict(epoch=-1, global_step=0, checkpoint_next_epoch=0,
                     raw_auc_gap=.12, raw_audit_saturated=1., reward_round_id=1),
                dict(epoch=9, global_step=100, checkpoint_next_epoch=10,
                     raw_auc_gap=.14, raw_audit_saturated=1., reward_round_id=4)])
        state = AdaptiveOmniFoldState.from_dict(payload)
        initialize_global_raw_best(state)
        for key in ('raw_best_epoch','raw_best_global_step','raw_best_next_epoch',
                    'raw_best_checkpoint','raw_best_auc_gap','raw_no_improvement_streak'):
            self.assertEqual(getattr(state,key), payload[key])

    def test_legacy_round_best_migration_remains_available(self):
        state = AdaptiveOmniFoldState.from_dict(dict(raw_best_auc_gap=.12,
            reward_round_id=4, probe_history=[dict(epoch=9, global_step=100,
            raw_auc_gap=.12, reward_round_id=4)]))
        self.assertEqual((state.raw_best_epoch,state.raw_best_global_step),(9,100))


if __name__ == '__main__':
    unittest.main()
