"""Reject raw/incomplete/mismatched inputs before distributed work starts."""
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))
from evenet.dataset.filtered_data import validate_filtered_dataset
from train_neutrino_backend import read_overlay_yaml


@pytest.mark.parametrize("file_manifest", [True, False])
def test_existing_manifest_formats_and_failures(tmp_path, file_manifest):
    data = tmp_path / "train"
    data.mkdir()
    parquet = data / "part.parquet"
    pq.write_table(pa.table({"value": [1., 2.]}), parquet)
    (data / "shape_metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="no raw-data fallback"):
        validate_filtered_dataset(data)
    manifest = dict(complete=True, rows_in=3, rows_out=2, events_removed=1)
    if file_manifest:
        manifest["files"] = [{"output": str(parquet)}]
    else:
        manifest["output"] = str(data)
    target = tmp_path / "filter_manifest.json"
    target.write_text(json.dumps(manifest))
    assert validate_filtered_dataset(data)["rows"] == 2
    target.write_text(json.dumps({**manifest, "complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        validate_filtered_dataset(data)
    target.write_text(json.dumps({**manifest, "rows_out": 3}))
    with pytest.raises(ValueError, match="row counts"):
        validate_filtered_dataset(data)
    target.write_text(json.dumps(manifest))
    pq.write_table(pa.table({"value": [3.]}), data / "extra.parquet")
    with pytest.raises(ValueError, match="(file outputs|row counts)"):
        validate_filtered_dataset(data)


def test_conditioning_family_resolves_to_same_verified_filtered_inputs():
    for name in ("relation_context", "relation_relations", "context_shift_only",
                 "pair_attention", "conditional_preconditioning", "relation_context_filtered"):
        config = read_overlay_yaml(ROOT / f"config/train_diffusion_{name}.yaml")
        platform = config["platform"]
        assert platform["require_filtered_data"] is True
        assert platform["number_of_workers"] == 16
        assert platform["data_parquet_dir"].endswith("/omnifold_attention_10pct_stic_filtered_test1/train")
        assert platform["data_parquet_val_dir"].endswith("/diffusion_val_20pct_seed42_stic_filtered_test1/val")
