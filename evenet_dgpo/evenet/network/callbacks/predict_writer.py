import os
import torch
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from typing import Any
from pathlib import Path


def _to_cpu(obj):
    """Recursively move tensors in a nested container to CPU (for safe saving)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu(v) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


class PredWriter(BasePredictionWriter):
    """Write all predictions into a single ``.pt`` file.

    Produces one final file at ``output_dir / filename`` (the original EveNet
    behaviour), but merges ranks via the shared filesystem instead of
    ``torch.distributed.all_gather_object``. Object-gathering the full list of
    predictions -- which still holds CUDA tensors -- segfaults under the NCCL
    backend for non-trivial payloads (point clouds + extras). Instead:

      1. Each rank moves its predictions to CPU and writes a part file
         ``.{filename}.part_rank{R}.pt``.
      2. A barrier ensures every rank has flushed its part.
      3. Rank 0 loads all part files, concatenates, writes ``filename``, and
         removes the parts.

    The earlier per-batch variant emitted one shard per (rank, batch)
    (``{prefix}_rank{R}_batch{B}.pt``), i.e. dozens of files per run; this
    restores the single-file output without the gather-induced crash.
    """

    def __init__(
            self,
            output_dir: Path,
            filename_prefix: str = "predictions.pt",
            write_interval: str = "epoch",
    ):
        super().__init__(write_interval=write_interval)
        self.output_dir = Path(output_dir)
        # Accept the historical ``filename_prefix`` kwarg used by predict.py and
        # treat it as the single output filename.
        self.filename = filename_prefix
        os.makedirs(self.output_dir, exist_ok=True)

    def _part_path(self, rank: int) -> Path:
        return self.output_dir / f".{self.filename}.part_rank{rank}.pt"

    def write_on_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            predictions: Any,
            batch_indices: list[list[list[int]]],
    ) -> None:
        rank = trainer.global_rank
        # Use the same source of truth as evenet_origin (torch.distributed) when
        # available; fall back to the Lightning trainer's world_size otherwise.
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            world_size = torch.distributed.get_world_size()
        else:
            world_size = max(1, int(getattr(trainer, "world_size", 1) or 1))

        # 1) Each rank writes its own part (CPU tensors) to the shared FS.
        torch.save(_to_cpu(predictions), self._part_path(rank))

        # 2) Make sure every rank has finished writing before rank 0 merges.
        if distributed:
            torch.distributed.barrier()

        if not trainer.is_global_zero:
            return

        # 3) Rank 0 merges all parts into a single file, then cleans up.
        all_predicts: list = []
        for r in range(world_size):
            part = self._part_path(r)
            if not part.exists():
                print(f"[PredWriter] WARNING: missing part file for rank {r}: {part}")
                continue
            part_preds = torch.load(part, map_location="cpu")
            if isinstance(part_preds, list):
                all_predicts.extend(part_preds)
            else:
                all_predicts.append(part_preds)

        save_path = os.path.join(self.output_dir, self.filename)
        torch.save(all_predicts, save_path)
        print(f"--> Saved {len(all_predicts)} predictions to {save_path}")

        for r in range(world_size):
            part = self._part_path(r)
            try:
                part.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[PredWriter] WARNING: could not remove part {part}: {exc}")
