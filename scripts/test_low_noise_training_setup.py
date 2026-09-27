"""Check actual optimizer factory behavior without constructing the large model."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]


def test_resolved_body_and_head_lr_and_schedule():
    from evenet.control.global_config import DotDict

    # generated_event_info.yaml is a NERSC preprocessing output, absent locally.
    # Resolve the training/default merge using the production merge operation.
    config = DotDict(yaml.safe_load((ROOT / 'config/train_diffusion_low_noise_10pct_nersc.yaml').read_text()))
    overrides = config.options.to_dict()
    defaults = overrides.pop('default')
    config.options = DotDict(yaml.safe_load((ROOT / 'config' / defaults).read_text()))
    config.options.merge(overrides)
    training = config.options.Training
    assert training.epochs == training.total_epochs == 250
    assert not training.EMA.enable
    assert training.model_checkpoint_load_path is None
    assert config.platform.number_of_workers == 16
    assert 'stic_filtered' in str(config.platform.data_parquet_val_dir)
    assert training.Components.PET.learning_rate == training.Components.GlobalEmbedding.learning_rate
    assert training.Components.TruthGeneration.low_noise_weight == 4

    tree = ast.parse((ROOT / 'evenet_dgpo/evenet/engine.py').read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'create_optim_schedule')
    ns = dict(torch=torch, math=math, world_size=16, lr_factor=1,
              warmup_steps=13, get_cosine_schedule_with_warmup=get_cosine_schedule_with_warmup,
              self=SimpleNamespace(config=config, total_steps=250*13))
    exec(compile(ast.Module(body=[factory], type_ignores=[]), 'optimizer_factory', 'exec'), ns)
    for component, expected in [('PET', 2e-5), ('GlobalEmbedding', 2e-5), ('TruthGeneration', 1e-4)]:
        opt, scheduler = ns['create_optim_schedule'](
            [torch.nn.Parameter(torch.zeros(1))], training.Components[component].learning_rate,
            0.001, optimizer_type='adamW')
        for _ in range(13):
            opt.step()
            scheduler.step()
        assert math.isclose(opt.param_groups[0]['lr'], expected, rel_tol=1e-12)
        for _ in range(249*13):
            opt.step()
            scheduler.step()
        assert abs(opt.param_groups[0]['lr']) < 1e-12


def test_existing_configs_keep_gpu_lr_scaling_by_default():
    tree = ast.parse((ROOT / 'evenet_dgpo/evenet/engine.py').read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'create_optim_schedule')
    ns = dict(torch=torch, math=math, world_size=16, lr_factor=1, warmup_steps=0,
              get_cosine_schedule_with_warmup=get_cosine_schedule_with_warmup,
              self=SimpleNamespace(config=SimpleNamespace(options=SimpleNamespace(Training={})), total_steps=100))
    exec(compile(ast.Module(body=[factory], type_ignores=[]), 'optimizer_factory', 'exec'), ns)
    opt, _ = ns['create_optim_schedule']([torch.nn.Parameter(torch.zeros(1))], 2e-4, 0.001, warm_up=False, optimizer_type='adamW')
    assert math.isclose(opt.param_groups[0]['lr'], 8e-4)
