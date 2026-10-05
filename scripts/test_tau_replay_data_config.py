"""Regression: 416701 rows must survive sixteen-way replay ingestion."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from ray.data import DataContext
from ray.train import DataConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evenet_dgpo'))
from RL.DGPO_neutrino.tau_replay_data_config import CompleteReplayDataConfig


class Dataset:
    def __init__(self, rows):
        self.rows = rows
        self.context = SimpleNamespace(execution_options=DataContext.get_current().execution_options)
        self.blocks = None

    def repartition(self, blocks):
        self.blocks = blocks
        return self

    def streaming_split(self, world, *, equal, locality_hints):
        count, remainder = divmod(self.rows, world)
        return [count + (rank < remainder and not equal) for rank in range(world)]


def test_complete_training_remainder_and_default_validation_routing():
    train, validation = Dataset(416701), object()
    config = CompleteReplayDataConfig()
    with patch.object(DataConfig, 'configure', return_value=[{'validation': i} for i in range(16)]) as parent:
        shards = config.configure({'train': train, 'validation': validation}, 16, None, None)
    parent.assert_called_once_with({'validation': validation}, 16, None, None)
    assert train.blocks == 16
    assert sum(shard['train'] for shard in shards) == 416701
    assert sorted(shard['train'] for shard in shards) == [26043]*3 + [26044]*13
    assert [shard['validation'] for shard in shards] == list(range(16))
