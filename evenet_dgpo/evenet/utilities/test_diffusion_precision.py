import unittest
import torch
from evenet.utilities.diffusion_sampler import DDIMSampler, add_noise, get_logsnr_alpha_sigma


class DiffusionPrecisionTest(unittest.TestCase):
    def test_real_generation_heads_forward_backward(self):
        from evenet.network.heads.generation.generation_head import EventGenerationHead, GlobalCondGenerationHead
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                head = EventGenerationHead(input_dim=8, projection_dim=8,
                    num_global_cond=8, num_classes=3, output_dim=2,
                    num_layers=1, num_heads=2, dropout=0., layer_scale=False,
                    layer_scale_init=1e-5, drop_probability=0., feature_drop=0.).to(dtype=dtype)
                label = torch.tensor([[0], [2]], dtype=torch.long)
                time = torch.tensor([.2, .8], dtype=dtype)
                result = head(x=torch.randn(2, 2, 8, dtype=dtype),
                    global_cond=torch.randn(2, 1, 8, dtype=dtype),
                    global_cond_mask=torch.ones(2, 1, 1, dtype=torch.bool),
                    num_x=None, x_mask=torch.ones(2, 2, 1, dtype=torch.bool),
                    time=time, label=label)
                self.assertEqual(result.dtype, dtype)
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(head.label_dense.weight.grad).all())
                global_head = GlobalCondGenerationHead(num_layer=1, num_resnet_layer=1,
                    input_dim=1, hidden_dim=8, output_dim=1, input_cond_indices=[0, 1],
                    num_classes=3, resnet_dim=8, layer_scale_init=1e-5,
                    feature_drop_for_stochastic_depth=0., activation='gelu', dropout=0.).to(dtype=dtype)
                result = global_head(torch.randn(2, 1, dtype=dtype), time,
                    global_cond=torch.randn(2, 1, 2, dtype=dtype), label=label)
                self.assertEqual(result.dtype, dtype)
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(global_head.label_embedding[0].weight.grad).all())

    def test_schedule_and_noise_preserve_precision(self):
        for dtype in (torch.float32, torch.float64):
            t = torch.tensor([0.1, 0.9], dtype=dtype)
            for value in get_logsnr_alpha_sigma(t, (2, 1, 1)):
                self.assertEqual(value.dtype, dtype)
                self.assertTrue(torch.isfinite(value).all())
            for value in add_noise(torch.ones(2, 2, 2, dtype=dtype), t):
                self.assertEqual(value.dtype, dtype)

    def test_sampling_float64_model(self):
        layer = torch.nn.Linear(2, 2).double()
        def predict(noise_x, time):
            self.assertEqual(noise_x.dtype, torch.float64)
            self.assertEqual(time.dtype, torch.float64)
            return layer(noise_x)
        result = DDIMSampler('cpu', dtype=torch.float64).sample(
            (3, 2, 2), predict, num_steps=3)
        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
