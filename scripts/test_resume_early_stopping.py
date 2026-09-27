from types import SimpleNamespace
import torch
from lightning.pytorch.callbacks import EarlyStopping
from evenet.utilities.resume_early_stopping import ApplyConfiguredPatience


def test_resume_uses_new_patience_preserves_history():
    old = EarlyStopping('val/loss', patience=25)
    old.wait_count = 25
    old.best_score = torch.tensor(.197498)
    resumed = EarlyStopping('val/loss', patience=75)
    policy = ApplyConfiguredPatience(resumed.patience)
    resumed.load_state_dict(old.state_dict())
    policy.on_fit_start(SimpleNamespace(callbacks=[resumed, policy]), None)
    assert resumed.patience == 75
    assert resumed.wait_count == 25
    assert torch.equal(resumed.best_score, old.best_score)
