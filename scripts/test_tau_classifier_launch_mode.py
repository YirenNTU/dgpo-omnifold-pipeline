import unittest
from scripts.train_tau_classifier_current_dgpo import configure_classifier_mode


class LaunchModeTests(unittest.TestCase):
    def config(self):
        return dict(reward_config={'type':'conditional_tau'},
                    dgpo={'tau_ratio':{}, 'adaptive_omnifold':{'enabled':False}},
                    experiment={'classifier_only':True})

    def test_legacy_h4_disabled_tau_terminal_path_enabled(self):
        cfg = self.config()
        configure_classifier_mode(cfg)
        self.assertFalse(cfg['experiment']['classifier_only'])
        self.assertTrue(cfg['dgpo']['tau_ratio']['classifier_only'])
        self.assertTrue(cfg['experiment']['tau_classifier_only'])
        self.assertEqual(cfg['experiment']['actor_updates'], 0)

    def test_wrong_reward_rejected(self):
        cfg = self.config(); cfg['reward_config']['type'] = 'omnifold'
        with self.assertRaises(ValueError): configure_classifier_mode(cfg)

    def test_adaptive_conflict_rejected(self):
        cfg = self.config(); cfg['dgpo']['adaptive_omnifold']['enabled'] = True
        with self.assertRaises(ValueError): configure_classifier_mode(cfg)


if __name__ == '__main__': unittest.main()
