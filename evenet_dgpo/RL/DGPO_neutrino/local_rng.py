"""Isolate CPU and explicitly owned CUDA generators, never all visible GPUs."""
from contextlib import contextmanager
import torch


def model_cuda_devices(model):
    return sorted({value.device.index for value in list(model.parameters()) + list(model.buffers())
                   if value.device.type == 'cuda'})


@contextmanager
def seeded_torch_rng(seed, devices):
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield
