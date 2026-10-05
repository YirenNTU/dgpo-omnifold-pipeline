"""Exercise production manual-update code without optional segmentation imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def method(name):
    source = Path(__file__).resolve().parents[1] / "evenet_dgpo/evenet/engine.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EveNetEngine")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    scope = {"torch": torch, "clip_grad_norm_": torch.nn.utils.clip_grad_norm_}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("container", [None, list, tuple])
def test_manual_update_single_and_multiple_optimizer_returns(container):
    model = torch.nn.Linear(1, 1)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.01)]
    if container is not None:
        optimizers.append(torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01))
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1) for opt in optimizers]
    loss = model(torch.ones(2, 1)).square().mean()
    before = model.weight.detach().clone()
    engine = SimpleNamespace(
        optimizers=lambda: optimizers[0] if container is None else container(optimizers),
        lr_schedulers=lambda: schedulers[0] if container is None else container(schedulers),
        current_step=0, eval_metrics=False, include_famo=False, log_gradient_step=1,
        global_rank=0, model=model, ema_model=None,
        prepare_heads_loss=lambda: ({}, {}),
        shared_step=lambda **kwargs: (loss, {}, {}, None, {}),
        prepare_mtl_parameters=lambda heads: ({"generation": loss}, [], [], None),
        safe_manual_backward=lambda value: value.backward(),
        check_gradient=lambda heads: None,
    )
    engine.log_task_gradient = lambda tasks, params: method("log_task_gradient")(engine, tasks, params)
    method("training_step")(engine, {"x": torch.ones(2, 1)}, 0)
    assert not torch.equal(before, model.weight)
    assert all(scheduler.last_epoch == 1 for scheduler in schedulers)
