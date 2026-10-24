"""CPU-only regression for dataset deserialization before train_worker starts."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.ttbar_dgpo_general.preprocessing_bridge import bind_process_fn, native_preprocessor


def cpu_worker(config):
    import ray.train
    batches = list(ray.train.get_dataset_shard('train').iter_batches(batch_size=2))
    assert sum(len(b['x']) for b in batches) == 2
    np.testing.assert_array_equal(batches[0]['x'], [[1., 3.], [2., 4.]])
    assert 'EXTRA/keep' in batches[0]
    ray.train.report({'rows_checked': 2})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evenet-root', required=True)
    p.add_argument('--ray', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, a.evenet_root)
    import ray.cloudpickle as pickle
    from evenet import shared
    table = {'x:1': np.array([3., 4.]), 'x:0': np.array([1., 2.]), 'EXTRA/keep': np.array([5., 6.])}
    metadata = {'x': [2]}
    options = SimpleNamespace(prediction={'extra_save': True}, Training=SimpleNamespace(
        Components=SimpleNamespace(**{k: SimpleNamespace(include=False) for k in ('Regression', 'Assignment', 'Segmentation')})))
    options.get = lambda key, default=None: getattr(options, key, default)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder/'shape_metadata.json').write_text(json.dumps(metadata))
        with patch.object(shared, 'global_config', SimpleNamespace(options=options)):
            original = shared.make_process_fn(folder)
            fn = bind_process_fn(shared.make_process_fn, a.evenet_root)(folder)
        for key, expected in original(table).items():
            np.testing.assert_array_equal(fn(table)[key], expected)
        # Explicitly check column dropping and multidimensional metadata as well.
        dropped = native_preprocessor(a.evenet_root)(table, metadata, ['EXTRA/'])
        assert 'EXTRA/keep' not in dropped
        (folder/'transport.pkl').write_bytes(pickle.dumps((fn, table)))
        # Start with ml_pipeline first, as on the failing Ray actor. No driver modules survive.
        program = """import sys
import ray.cloudpickle as pickle
import numpy as np
sys.path.insert(0, sys.argv[1])
with open(sys.argv[2], 'rb') as f: fn, table = pickle.load(f)
assert 'preprocessing.preprocess' not in sys.modules
np.testing.assert_array_equal(fn(table)['x'], [[1., 3.], [2., 4.]])
assert 'preprocessing.preprocess' not in sys.modules
print('Fresh-process Ray deserialization and original-source parity PASS')
"""
        subprocess.run([sys.executable, '-c', program, str(ROOT), str(folder/'transport.pkl')], cwd=ROOT,
                       env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT), a.evenet_root))}, check=True)
        if a.ray:
            import ray
            from ray.train import RunConfig, ScalingConfig
            from ray.train.torch import TorchTrainer
            # This is a two-row CPU transport check, no policy/model/optimizer.
            ray.init(num_cpus=4, include_dashboard=False, runtime_env={
                'env_vars': {'PYTHONPATH': os.pathsep.join((str(ROOT), a.evenet_root))}})
            try:
                ds = ray.data.from_numpy(np.arange(2)).map_batches(
                    partial(make_table, table=table)).map_batches(fn)
                result = TorchTrainer(cpu_worker, scaling_config=ScalingConfig(
                    num_workers=1, use_gpu=False), datasets={'train': ds},
                    run_config=RunConfig(storage_path=str(folder/'ray-results'), name='preprocessing-transport-check')).fit()
                assert result.metrics['rows_checked'] == 2
                print('Ray TorchTrainer dataset deserialization PASS; no policy training')
            finally:
                ray.shutdown()


def make_table(batch, *, table):
    return table


if __name__ == '__main__':
    main()
