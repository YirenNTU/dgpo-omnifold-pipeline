"""Serialize EveNet preprocessing without the ambiguous `preprocessing` name.

Execute the original source file, not a copied implementation. Ray imports this
bridge by its unique module name before it has entered the training worker.
"""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path
import sys


@lru_cache(maxsize=None)
def native_preprocessor(evenet_root):
    root = Path(evenet_root).expanduser().resolve()
    source = root / 'preprocessing/preprocess.py'
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(root))
    name = 'experiments.ttbar_dgpo_general._native_preprocessing'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    # Do not modify sys.modules['preprocessing']; ml_pipeline owns that name too.
    spec.loader.exec_module(module)
    return module.unflatten_dict


def unflatten(table, shape_metadata, drop_column_prefix=None, delimiter=':', *, evenet_root):
    return native_preprocessor(evenet_root)(
        table, shape_metadata, drop_column_prefix=drop_column_prefix, delimiter=delimiter)


def bind_process_fn(original_factory, evenet_root):
    """Keep upstream metadata/column-selection semantics; replace only transport."""
    def make_process_fn(base_dir):
        fn = original_factory(base_dir)
        original = fn.keywords['unflatten']
        expected = Path(evenet_root).resolve() / 'preprocessing/preprocess.py'
        if Path(original.__code__.co_filename).resolve() != expected:
            raise RuntimeError('Driver imported the wrong EveNet preprocessing source')
        return partial(fn.func, *fn.args, **{
            **fn.keywords, 'unflatten': partial(unflatten, evenet_root=str(Path(evenet_root).resolve()))})
    return make_process_fn
