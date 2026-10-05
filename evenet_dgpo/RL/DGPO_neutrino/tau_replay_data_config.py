"""Ray ingestion for complete-population native replay capture."""
from __future__ import annotations

import copy

from ray.train import DataConfig


class CompleteReplayDataConfig(DataConfig):
    """Keep the training remainder; replay handles unequal per-rank tails.

    Ordinary live DDP training keeps Ray's default equal split. This class is
    only installed for full-trajectory runs that capture then cycle saved batches.
    """

    def configure(self, datasets, world_size, worker_handles, worker_node_ids, **kwargs):
        output = super().configure(
            {name: ds for name, ds in datasets.items() if name != 'train'},
            world_size, worker_handles, worker_node_ids, **kwargs)
        # Enough blocks for every worker, without dropping N % world_size rows.
        train = datasets['train'].repartition(world_size)
        options = copy.deepcopy(self._execution_options)
        if options.is_resource_limits_default():
            resource_type = type(options.exclude_resources)
            options.exclude_resources = options.exclude_resources.add(
                resource_type(cpu=self._num_train_cpus, gpu=self._num_train_gpus))
        train.context.execution_options = options
        hints = worker_node_ids if options.locality_with_output else None
        for rank, shard in enumerate(train.streaming_split(
                world_size, equal=False, locality_hints=hints)):
            output[rank]['train'] = shard
        return output
