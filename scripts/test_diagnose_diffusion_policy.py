"""CPU contracts; real checkpoint inference must be run on NERSC."""
import contextlib
import io
import unittest
import torch

from diagnose_diffusion_policy import CheckedPolicy, check_batches


class FakePolicy:
    invisible_input_dim = 2
    invisible_normalizer = None

    def predict_diffusion_vector(self, noise_x, **kwargs):
        return noise_x + 1


class TestPolicyDiagnostic(unittest.TestCase):
    def test_checks_all_rows_including_late_outlier_and_partial_batch(self):
        fields = {"x": torch.zeros(2050, 1, 1)}
        fields["x"][1147] = 1.e9
        visited = []

        def generate(model, batch, sampler, **kwargs):
            visited.append(batch["x"].clone())
            return model.predict_diffusion_vector(batch["x_invisible"]).unsqueeze(0)

        with contextlib.redirect_stdout(io.StringIO()):
            result = check_batches(FakePolicy(), fields, device=torch.device('cpu'), batch_size=256,
                                   ddim_steps=20, seed=42, generate=generate, sampler=None)
        self.assertEqual(result['status'], 'finite')
        self.assertEqual(result['events'], 2050)
        self.assertEqual(result['batches'][-1]['stop'], 2050)
        self.assertTrue(torch.equal(torch.cat(visited), fields['x']))
        self.assertEqual(len(result['batches']), 9)

    def test_velocity_nonfinite_is_detected_before_sampler_can_hide_it(self):
        policy = CheckedPolicy(FakePolicy())
        with self.assertRaises(FloatingPointError):
            policy.predict_diffusion_vector(torch.tensor([float('nan')]))
        self.assertEqual(policy.calls[-1]['nonfinite'], 1)

    def test_nonfinite_final_samples_fail_case(self):
        def generate(model, batch, sampler, **kwargs):
            model.predict_diffusion_vector(batch['x_invisible'])
            return torch.full((1, len(batch['x']), 2, 2), float('inf'))

        with contextlib.redirect_stdout(io.StringIO()):
            result = check_batches(FakePolicy(), {'x': torch.zeros(2, 1, 1)}, device=torch.device('cpu'),
                                   batch_size=256, ddim_steps=20, seed=42, generate=generate, sampler=None)
        self.assertEqual(result['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
