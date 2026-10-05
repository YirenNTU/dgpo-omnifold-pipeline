"""Validate existing filtered inputs before creating distributed data workers."""
import json
from pathlib import Path


def validate_filtered_dataset(directory):
    import pyarrow.parquet as pq

    directory = Path(directory).expanduser().resolve()
    manifest_path = directory.parent / "filter_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Filtered data required: missing {manifest_path}; no raw-data fallback")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError(f"Filtered dataset is incomplete: {manifest_path}")
    files = sorted(directory.glob("*.parquet"))
    if not files or not (directory / "shape_metadata.json").is_file():
        raise ValueError(f"Missing filtered parquet files or shape_metadata.json: {directory}")
    # The existing train manifest lists files; the validation manifest records
    # its output directory. Verify both formats against the actual requested input.
    if "files" in manifest:
        declared = {Path(item["output"]).resolve() for item in manifest["files"]}
        if declared != {path.resolve() for path in files}:
            raise ValueError(f"Filtered manifest file outputs do not match {directory}")
    elif Path(manifest.get("output", "")).resolve() != directory:
        raise ValueError(f"Filtered manifest output does not match {directory}")
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if (rows != manifest.get("rows_out")
            or manifest.get("rows_in") != rows + manifest.get("events_removed", -1)):
        raise ValueError(f"Filtered manifest row counts do not match {directory}")
    return {"directory": str(directory), "manifest": str(manifest_path),
            "rows": rows, "events_removed": manifest["events_removed"]}
