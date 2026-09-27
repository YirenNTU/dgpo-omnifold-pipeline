import ast
import math
from pathlib import Path
import unittest

import torch

from evenet.network.body.angular_conditioning import VisibleAngularFourier
from evenet.network.body.embedding import PETBody


class TestAngularConditioning(unittest.TestCase):
    def setUp(self):
        self.branch = VisibleAngularFourier(["energy", "pt", "eta", "phi"], 8, "eta", "phi")
        self.x = torch.tensor([[[5., 2., 0., 0.3], [2., 1., 1., -2.]]])
        self.mask = torch.ones(1, 2, 1, dtype=torch.bool)

    def test_formula_and_all_four_harmonics(self):
        f = self.branch.features(self.x, self.mask)
        k = torch.arange(1, 5)
        expected = torch.cat(((math.pi / 2 * k).sin(), (math.pi / 2 * k).cos(),
                              (0.3 * k).sin(), (0.3 * k).cos()))
        torch.testing.assert_close(f[0, 0], expected)
        self.assertEqual(f.shape[-1], 16)

    def test_phi_periodicity(self):
        shifted = self.x.clone()
        shifted[..., 3] += 2 * math.pi
        torch.testing.assert_close(self.branch.features(self.x, self.mask),
                                   self.branch.features(shifted, self.mask), atol=3e-6, rtol=1e-5)

    def test_padding_nan_is_zero(self):
        self.x[:, 1] = float("nan")
        self.mask[:, 1] = False
        f = self.branch.features(self.x, self.mask)
        self.assertTrue(torch.isfinite(f).all())
        self.assertEqual(f[:, 1].abs().sum().item(), 0)

    def test_extreme_eta_and_distinct_poles(self):
        self.x[0, :, 2] = torch.tensor([1000., -1000.])
        f = self.branch.features(self.x, self.mask)
        self.assertTrue(torch.isfinite(f).all())
        self.assertAlmostEqual(f[0, 0, 4].item(), 1.)
        self.assertAlmostEqual(f[0, 1, 4].item(), -1.)

    def test_theta_input_equivalence(self):
        other = VisibleAngularFourier(["energy", "pt", "theta", "phi"], 8, theta_source="theta", phi_source="phi")
        x = self.x.clone()
        x[..., 2] = 2 * torch.atan(torch.exp(-x[..., 2]))
        torch.testing.assert_close(self.branch.features(self.x, self.mask), other.features(x, self.mask))

    def test_zero_init_and_gradient(self):
        out = self.branch(self.x, self.mask)
        self.assertEqual(out.abs().sum().item(), 0)
        out.sum().backward()
        self.assertGreater(self.branch.projection.weight.grad.norm().item(), 0)

    def test_feature_schema_fails_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            VisibleAngularFourier(["truth_eta", "truth_phi"], 8)

    def test_production_feature_names(self):
        branch = VisibleAngularFourier(["Part_energy", "Part_pt", "Part_eta", "Part_phi"], 8)
        torch.testing.assert_close(branch.features(self.x, self.mask), self.branch.features(self.x, self.mask))

    def test_pet_old_checkpoint_equivalence_and_learned_response(self):
        def pet(branch):
            return PETBody(num_feat=4, num_keep=4, feature_drop=0., projection_dim=8,
                           local=False, K=1, num_local=0, num_layers=1, num_heads=2,
                           drop_probability=0., talking_head=False, layer_scale=False,
                           layer_scale_init=1., dropout=0., mode="all", angular_conditioning=branch).eval()
        old, new = pet(None), pet(self.branch)
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(missing, ["angular_conditioning.projection.weight"])
        self.assertEqual(unexpected, [])
        inputs = torch.randn(1, 3, 4)
        kw = dict(input_features=inputs, input_points=inputs, mask=torch.ones(1, 3, 1), time=torch.zeros(1))
        torch.testing.assert_close(old(**kw), new(**kw, visible_raw=self.x), atol=0, rtol=0)
        with torch.no_grad():
            self.branch.projection.weight.normal_(std=.1)
        self.assertFalse(torch.allclose(old(**kw), new(**kw, visible_raw=self.x)))
        # Without the visible-only argument (e.g. event reconstruction), no bypass.
        torch.testing.assert_close(old(**kw), new(**kw), atol=0, rtol=0)
        self.assertIn("angular_conditioning.projection.weight", dict(new.named_parameters()))

    def test_training_and_sampling_source_contract(self):
        # Guard the actual call sites: never normalized x, x_invisible or truth.
        path = Path(__file__).resolve().parents[1] / "evenet_model.py"
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "PET"]
        sources = [ast.unparse(k.value) for c in calls for k in c.keywords if k.arg == "visible_raw"]
        self.assertEqual(sorted(sources), sorted([
            "x['x'] if schedule_name == 'neutrino_generation' else None", "cond_x['x']"]))

    def test_sampling_ignores_clean_invisible_truth_in_condition_dictionary(self):
        # Exercise the real sampling method with lightweight surrounding modules.
        from evenet.network.evenet_model import EveNetModel

        class Identity(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x

        class CapturePET(torch.nn.Module):
            def __init__(self, branch):
                super().__init__()
                self.angular_conditioning = branch
                self.seen = None

            def forward(self, input_features, visible_raw, mask, **kwargs):
                self.seen = visible_raw.detach().clone()
                n = visible_raw.shape[1]
                r = self.angular_conditioning(visible_raw, mask[:, :n])
                # Stand-in attention makes the observed condition affect output.
                return input_features + r.mean(dim=1, keepdim=True)[..., :4]

        model = EveNetModel.__new__(EveNetModel)
        torch.nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.sequential_normalizer = Identity()
        model.global_normalizer = Identity()
        model.GlobalEmbedding = Identity()
        model.TruthGeneration = Identity()
        model.project_sequential_inputs = lambda x, mask: x
        model.project_invisible_inputs = lambda x, mask: x
        model.local_feature_indices = [0, 1]
        model.neutrino_position_encode = False
        model.PET = CapturePET(self.branch)
        with torch.no_grad():
            self.branch.projection.weight.normal_(std=.1)
        cond = dict(x=self.x, x_mask=self.mask.squeeze(-1), conditions=torch.zeros(1, 2),
                    conditions_mask=torch.ones(1), x_invisible=torch.randn(1, 1, 4))
        noise, mask, time = torch.randn(1, 1, 4), torch.ones(1, 1, 1), torch.ones(1)
        a = model.predict_diffusion_vector(noise, cond, time, "neutrino", mask)
        cond["x_invisible"] = torch.full((1, 1, 4), float("nan"))
        b = model.predict_diffusion_vector(noise, cond, time, "neutrino", mask)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(model.PET.seen, self.x, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
