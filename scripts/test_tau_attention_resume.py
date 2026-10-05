import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from scripts.resume_tau_attention_dgpo import configure, validate_continuation

ROOT = Path(__file__).resolve().parents[1]


class ResumeTests(unittest.TestCase):
    def test_original_diffusion_resume_contract(self):
        settings = dict(require_existing_resume=True, require_original_diffusion=True)
        cfg = {'network':{'VisibleConditioning':{}}}
        state = dict(state_dict={'model.old.weight':0}, dgpo_omnifold_reward_stack={'head':{'cross_attention':True}})
        validate_continuation(settings,cfg,state,True)
        with self.assertRaises(ValueError): validate_continuation(settings,cfg,state,False)
        state['state_dict']['model.TruthGeneration.visible_conditioning.token_readout.output.weight'] = 0
        with self.assertRaises(ValueError): validate_continuation(settings,cfg,state,True)

    def test_runtime_preserves_fit_and_optimizers(self):
        cfg = dict(reward_config={'type':'conditional_tau'}, experiment={'actor_updates':0},
            dgpo=dict(adaptive_omnifold={'enabled':False}, tau_ratio={'fit':{'batch_size':1024,'lr':.0002}},
                      lr_schedule={'resume_use_config':False},
                      reference_trust={'enabled':True,'coefficient':1,'objective':'velocity_mse'}),
            platform={'number_of_workers':16}, options={'Training':{'EMA':{'enable':False}}},
            logger={'local':{},'wandb':{}}, nersc={'ray':{}})
        fit = copy.deepcopy(cfg['dgpo']['tau_ratio']['fit'])
        out = configure(cfg, {'wandb_name':'test', 'validation_every_epochs':5,
                              'validation_relative_to_install':True}, Path('/checkpoint'), Path('/output'))
        self.assertEqual(out['dgpo']['tau_ratio']['fit'], fit)
        self.assertFalse(out['dgpo']['tau_ratio']['classifier_only'])
        self.assertTrue(out['dgpo']['tau_ratio']['production_cross_attention'])
        self.assertEqual(out['dgpo']['tau_ratio']['refit_every_epochs'],5)
        self.assertEqual(out['dgpo']['tau_ratio']['validation_every_epochs'],5)
        self.assertTrue(out['dgpo']['tau_ratio']['validation_relative_to_install'])
        self.assertEqual(out['dgpo']['checkpoint_load_mode'],'resume')
        self.assertNotIn('actor_updates',out['experiment'])

    def test_five_completed_epochs_since_install(self):
        # Execute the real method without importing optional production dependencies.
        path = ROOT/'evenet_dgpo/RL/DGPO_neutrino/conditional_tau_cycle.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ConditionalTauCycle')
        fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='epoch_end')
        namespace = {}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        calls = []
        order = []
        reward = SimpleNamespace(last_refit_epoch=177)
        def refit(epoch, step):
            order.append(('refit',epoch))
            calls.append(epoch); reward.last_refit_epoch=epoch
        obj = SimpleNamespace(cfg={'refit_every_epochs':5,'validation_every_epochs':5,
                                  'validation_relative_to_install':True,
                                  'refit_relative_to_install':True}, reward=reward,
                              refit=refit,evaluate=lambda epoch,step:order.append(('validation',epoch)))
        for epoch in range(178,188):
            namespace['epoch_end'](obj,epoch,1780+(epoch-177)*10)
        self.assertEqual(calls,[182,187])
        self.assertEqual(order,[('validation',182),('refit',182),('validation',187),('refit',187)])
        namespace['epoch_end'](obj,192,1930,final=True)
        self.assertEqual(calls,[182,187])


if __name__ == '__main__': unittest.main()
