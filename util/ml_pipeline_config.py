from __future__ import annotations

from dataclasses import dataclass

from evenet_parquet_common import DEFAULT_GLOBAL_FIELDS, DEFAULT_PART_AUX_FIELDS, FOUR_VECTOR_FEATURES


@dataclass(frozen=True)
class FeatureConfig:
    part_momentum_fields: tuple[str, ...]
    part_aux_fields: tuple[str, ...]
    global_fields: tuple[str, ...]


@dataclass(frozen=True)
class EveNetConfig:
    classification_name: str
    process_topologies: dict[str, dict[str, tuple[str, ...]]]
    generation_conditions: tuple[str, ...]
    generation_global_targets: tuple[str, ...]
    generation_events: tuple[str, ...]
    sequential_tags: dict[str, str]
    global_tags: dict[str, str]
    invisible_tags: dict[str, str]


def parse_feature_config(config: dict) -> FeatureConfig:
    inputs_cfg = config.get("Inputs", {})

    part_cfg = inputs_cfg.get("Part", {})
    global_cfg = inputs_cfg.get("Global", {})

    momentum_fields = tuple(part_cfg.get("Momentum", FOUR_VECTOR_FEATURES))
    if momentum_fields != tuple(FOUR_VECTOR_FEATURES):
        raise ValueError(
            "Inputs.Part.Momentum must be exactly ['energy', 'pt', 'eta', 'phi'] "
            "because the converter rewrites four-momentum into this basis."
        )

    part_aux_fields = tuple(part_cfg.get("Auxiliary", DEFAULT_PART_AUX_FIELDS))
    global_fields = tuple(global_cfg.get("Fields", DEFAULT_GLOBAL_FIELDS))

    if len(set(part_aux_fields)) != len(part_aux_fields):
        raise ValueError("Inputs.Part.Auxiliary contains duplicate fields.")
    if len(set(global_fields)) != len(global_fields):
        raise ValueError("Inputs.Global.Fields contains duplicate fields.")

    return FeatureConfig(
        part_momentum_fields=momentum_fields,
        part_aux_fields=part_aux_fields,
        global_fields=global_fields,
    )


def _parse_process_topologies(raw_processes: dict) -> dict[str, dict[str, tuple[str, ...]]]:
    process_topologies: dict[str, dict[str, tuple[str, ...]]] = {}
    for process_name, resonance_map in raw_processes.items():
        process_topologies[process_name] = {
            resonance_name: tuple(products)
            for resonance_name, products in resonance_map.items()
        }
    return process_topologies


def _default_sequential_tags(feature_config: FeatureConfig) -> dict[str, str]:
    tags = {field: "none" for field in feature_config.part_aux_fields}
    for field in feature_config.part_momentum_fields:
        full_name = f"Part_{field}"
        if field in {"energy", "pt"}:
            tags[full_name] = "log_normalize"
        elif field == "eta":
            tags[full_name] = "normalize"
        elif field == "phi":
            tags[full_name] = "normalize_uniform"
        else:
            tags[full_name] = "none"
    return tags


def _default_global_tags(feature_config: FeatureConfig) -> dict[str, str]:
    tags: dict[str, str] = {}
    for field in feature_config.global_fields:
        if field in {
            "Event_totalChargedEnergy",
            "Event_totalEMEnergy",
            "Event_totalHadronicEnergy",
            "charged_E",
            "missing_pt",
        }:
            tags[field] = "log_normalize"
        elif field in {
            "thrust_Mag",
            "missing_px",
            "missing_py",
            "isolation_angle",
            "thrust_x",
            "thrust_y",
            "thrust_z",
        }:
            tags[field] = "normalize"
        else:
            tags[field] = "none"
    return tags


def _merge_tags(default_tags: dict[str, str], raw_tags: dict | None) -> dict[str, str]:
    merged = dict(default_tags)
    if not raw_tags:
        return merged
    raw_default = raw_tags.get("default")
    if raw_default is not None:
        for key in list(merged):
            merged[key] = raw_default
    for key, value in raw_tags.items():
        if key == "default":
            continue
        merged[key] = value
    return merged


def _resolve_normalization_cfg(config: dict) -> dict:
    normalization_cfg = config.get("Normalization")
    if normalization_cfg is not None:
        return normalization_cfg

    # Backward-compatible fallback for older analysis.yaml files.
    return config.get("EveNet", {}).get("FeatureTags", {})


def parse_evenet_config(config: dict, feature_config: FeatureConfig) -> EveNetConfig:
    evenet_cfg = config.get("EveNet", {})

    raw_processes = evenet_cfg.get("Processes", {})
    if not raw_processes:
        raise ValueError("analysis.yaml is missing EveNet.Processes for multi-process event_info generation.")

    classification_cfg = evenet_cfg.get("Classification", {})
    classification_name = classification_cfg.get("Name", "process")

    generations_cfg = evenet_cfg.get("Generations", {})
    generation_conditions = tuple(generations_cfg.get("Conditions", feature_config.global_fields))
    generation_global_targets = tuple(generations_cfg.get("GlobalTargets", ()))
    generation_events = tuple(
        generations_cfg.get(
            "Events",
            tuple(f"Part_{field}" for field in feature_config.part_momentum_fields),
        )
    )

    feature_tags_cfg = _resolve_normalization_cfg(config)
    sequential_tags = _merge_tags(
        _default_sequential_tags(feature_config),
        feature_tags_cfg.get("Sequential"),
    )
    global_tags = _merge_tags(
        _default_global_tags(feature_config),
        feature_tags_cfg.get("Global"),
    )
    invisible_tags = _merge_tags(
        {
            "energy": "log_normalize",
            "pt": "log_normalize",
            "eta": "normalize",
            "phi": "normalize_uniform",
        },
        feature_tags_cfg.get("Invisible"),
    )

    return EveNetConfig(
        classification_name=classification_name,
        process_topologies=_parse_process_topologies(raw_processes),
        generation_conditions=generation_conditions,
        generation_global_targets=generation_global_targets,
        generation_events=generation_events,
        sequential_tags=sequential_tags,
        global_tags=global_tags,
        invisible_tags=invisible_tags,
    )
