from lightning.pytorch.callbacks import Callback, EarlyStopping


class ApplyConfiguredPatience(Callback):
    """Apply the current policy after Lightning restores historical stop state."""

    def __init__(self, patience):
        self.patience = int(patience)

    def on_fit_start(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                callback.patience = self.patience

