"""CPU contract tests; production fitting remains user-launched on 16 GPUs."""
import math
import unittest
import torch
from scripts.train_conditional_spin_ratio import build_classifier


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(condition_dim=36, candidate_dim=21, relative_dim=6,
                        hidden=128, dropout=.05, head_kind='film', head_depth=3,
                        condition_hidden=256, condition_width=256, ratio_bound=30,
                        packing_spec={'shapes': {'x': [4, 4], 'x_mask': [4]}},
                        attention_heads=4)
        torch.manual_seed(100)
        self.c, self.t = torch.randn(6, 36), torch.randn(6, 21)
        self.c[:, 16:20] = torch.tensor([1., 1., 0., 0.])

    def models(self):
        torch.manual_seed(42)
        plain = build_classifier(self.cfg).eval()
        torch.manual_seed(42)
        attention = build_classifier(dict(self.cfg, cross_attention=True)).eval()
        return plain, attention

    def test_initial_function_and_shared_parameters_match(self):
        plain, attention = self.models()
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, attention.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(plain(self.c, self.t), attention(self.c, self.t), rtol=0, atol=0)

    def test_mask_and_empty_event(self):
        _, model = self.models()
        torch.nn.init.normal_(model.attention_output.weight, std=.1)
        h = model.spin_encoder(self.t)
        other = self.c.clone(); other[:, 8:16] = 10000
        # Test only the added path: original global MLP retains its old contract.
        torch.testing.assert_close(model.condition_candidate(h, self.c), model.condition_candidate(h, other))
        empty = self.c.clone(); empty[:, 16:20] = 0
        torch.testing.assert_close(model.condition_candidate(h, empty), h, rtol=0, atol=0)

    def test_learns_query_and_tokens_and_reloads(self):
        _, model = self.models()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(self.c, self.t)
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.arange(6) % 2.).backward()
            optimizer.step()
        for name in ('token_encoder.0.weight', 'cross_attention.in_proj_weight', 'attention_output.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        for grad in model.cross_attention.in_proj_weight.grad.chunk(3, dim=0):
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue((model(self.c, self.t).abs() <= math.log(30)+1e-6).all())
        clone = build_classifier(dict(self.cfg, cross_attention=True)).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(model(self.c, self.t), clone(self.c, self.t))

    def test_multiple_tokens_allow_query_dependent_selection(self):
        _, model = self.models()
        torch.nn.init.normal_(model.attention_output.weight, std=.1)
        c = self.c[:1].expand(2, -1)
        h = torch.randn(2, 64)
        residual = model.condition_candidate(h, c)-h
        self.assertGreater((residual[0]-residual[1]).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
