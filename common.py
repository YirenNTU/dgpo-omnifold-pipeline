from pathlib import Path
import yaml
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}

def ordered_class_labels(analysis_config: dict[str, Any], selected_keys: set[str] | None) -> list[str]:
    samples_cfg = analysis_config.get("Samples") or {}
    subcategories_cfg = analysis_config.get("Subcategories") or {}
    labels: list[str] = []

    for sample_key, sample_cfg in samples_cfg.items():
        if selected_keys and sample_key not in selected_keys:
            continue
        if bool(sample_cfg.get("is_data", False)):
            continue
        sample_subcategories = subcategories_cfg.get(sample_key)
        if sample_subcategories:
            labels.extend(str(label) for label in sample_subcategories.keys())
        else:
            sample_name = str(sample_cfg.get("name", sample_key))
            labels.append(sample_name)
    deduplicated: list[str] = []
    seen: set[str] = set()

    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduplicated.append(label)

    return deduplicated
