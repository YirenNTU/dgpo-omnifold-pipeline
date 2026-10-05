import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


Token = load('evenet_dgpo/evenet/network/body/token_conditioning.py','token_test').TokenSpecificConditioning
migrate = load('evenet_dgpo/RL/DGPO_neutrino/conditioning_probe.py','migration_test').extend_optimizer_for_probe


class MigrationTests(unittest.TestCase):
    def test_zero_output_mask_and_learning(self):
        branch = Token(8,6,width=8,heads=2,mode='residual')
        q, memory = torch.randn(3,2,8), torch.randn(3,4,6)
        qm = torch.ones(3,2,dtype=torch.bool)
        mm = torch.tensor([[1,1,0,0],[0,0,0,0],[1,0,0,0]],dtype=torch.bool)
        self.assertEqual(branch(q,memory,qm,mm).abs().sum().item(),0)
        opt = torch.optim.AdamW(branch.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad(); (branch(q,memory,qm,mm)-1).square().mean().backward(); opt.step()
        result = branch(q,memory,qm,mm)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[1].abs().sum().item(),0)
        changed = memory.masked_fill(~mm[...,None],9999)
        torch.testing.assert_close(result,branch(q,changed,qm,mm))
        self.assertGreater(branch.attention.in_proj_weight.grad.abs().sum().item(),0)

    def test_old_adam_moments_preserved_new_parameters_empty(self):
        old = nn.Module(); old.TruthGeneration = nn.Module()
        old.TruthGeneration.visible_conditioning = nn.Module()
        old.TruthGeneration.visible_conditioning.encoder = nn.Linear(3,4)
        optimizer = torch.optim.AdamW([{'params':list(old.parameters()),'group_name':'visible_conditioning'}],lr=1e-4)
        old.TruthGeneration.visible_conditioning.encoder(torch.randn(5,3)).square().sum().backward()
        optimizer.step()
        checkpoint = dict(state_dict=copy.deepcopy(old.state_dict()), dgpo_optimizer_state_dict={
            'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':{'last_epoch':1800}})
        model = copy.deepcopy(old)
        model.TruthGeneration.visible_conditioning.token_readout = Token(8,6,width=8,heads=2,mode='residual')
        live = torch.optim.AdamW([{'params':list(model.parameters()),'group_name':'visible_conditioning'}],lr=1e-4)
        migrated,count = migrate(model,live,checkpoint,branch_name='token_readout')
        self.assertGreater(count,0)
        self.assertEqual(migrated['dgpo_optimizer_state_dict']['scheduler'],{'last_epoch':1800})
        live.load_state_dict(migrated['dgpo_optimizer_state_dict']['optimizer'])
        for a,b in zip(old.parameters(),model.TruthGeneration.visible_conditioning.encoder.parameters()):
            torch.testing.assert_close(optimizer.state[a]['exp_avg'],live.state[b]['exp_avg'],rtol=0,atol=0)
        for p in model.TruthGeneration.visible_conditioning.token_readout.parameters():
            self.assertFalse(live.state.get(p))
        self.assertEqual(len(checkpoint['dgpo_optimizer_state_dict']['optimizer']['param_groups'][0]['params']),2)


if __name__ == '__main__': unittest.main()
