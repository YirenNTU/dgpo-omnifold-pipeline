from __future__ import annotations

from dataclasses import dataclass

from evenet_parquet_common import DEFAULT_GLOBAL_FIELDS, DEFAULT_PART_AUX_FIELDS, FOUR_VECTOR_FEATURES


@dataclass(frozen=True)
class FeatureConfig:
    part_momentum_fields: tuple[str, ...]
    part_aux_fields: tuple[str, ...]
    global_fields: tuple[str, ...]
    raw_sequential_fields: tuple[str, ...]
    projected_sequential_feature_names: tuple[str, ...]
    grouped_sequential_config: dict | None


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


def _normalize_part_feature_name(raw_name: str) -> str:
    if raw_name in FOUR_VECTOR_FEATURES:
        return f"Part_{raw_name}"
    return raw_name


def _is_group_config(raw_value) -> bool:
    return isinstance(raw_value, dict) and "input" in raw_value


def _flatten_leaf_features(raw_inputs, *, allow_top_level_groups: bool) -> list[str]:
    leaves: list[str] = []
    for child in raw_inputs:
        if isinstance(child, str):
            leaves.append(_normalize_part_feature_name(child))
            continue

        if not isinstance(child, dict) or len(child) != 1:
            raise ValueError("Grouped sequential inputs must contain strings or single-key subgroup mappings.")

        child_name, child_value = next(iter(child.items()))
        if _is_group_config(child_value):
            child_inputs = child_value.get("input", [])
        elif isinstance(child_value, list):
            child_inputs = child_value
        else:
            raise ValueError(
                f"Subgroup '{child_name}' must be a list or a dict with an 'input' field."
            )
        leaves.extend(_flatten_leaf_features(child_inputs, allow_top_level_groups=False))

    return leaves


def _normalize_group_node(name: str, raw_value, *, top_level: bool) -> dict:
    if not _is_group_config(raw_value):
        raise ValueError(f"Top-level grouped input '{name}' must define 'index' and 'input'.")

    raw_indices = raw_value.get("index")
    if raw_indices is None:
        if top_level:
            raise ValueError(f"Top-level grouped input '{name}' is missing 'index'.")
        indices: tuple[int, ...] = ()
    else:
        indices = tuple(int(index) for index in raw_indices)

    if top_level and len(indices) == 0:
        raise ValueError(f"Top-level grouped input '{name}' must project to at least one slot.")

    children = []
    raw_inputs = raw_value.get("input", [])
    if not isinstance(raw_inputs, list) or len(raw_inputs) == 0:
        raise ValueError(f"Grouped input '{name}' must contain a non-empty 'input' list.")

    for child in raw_inputs:
        if isinstance(child, str):
            children.append(_normalize_part_feature_name(child))
            continue

        if not isinstance(child, dict) or len(child) != 1:
            raise ValueError(
                f"Grouped input '{name}' must contain strings or single-key subgroup mappings."
            )

        child_name, child_value = next(iter(child.items()))
        if _is_group_config(child_value):
            normalized_child = _normalize_group_node(child_name, child_value, top_level=False)
        elif isinstance(child_value, list):
            normalized_child = {
                "name": child_name,
                "output_dim": 1,
                "input": _normalize_group_children(child_name, child_value),
            }
        else:
            raise ValueError(
                f"Subgroup '{child_name}' under '{name}' must be a list or a dict with an 'input' field."
            )
        children.append({child_name: normalized_child})

    explicit_dim = raw_value.get("dim", raw_value.get("output_dim"))
    if explicit_dim is None:
        if top_level:
            if len(indices) == 0:
                raise ValueError(f"Top-level grouped input '{name}' must define 'dim'.")
            output_dim = len(indices)
        else:
            output_dim = 1
    else:
        output_dim = int(explicit_dim)

    if output_dim <= 0:
        raise ValueError(f"Grouped input '{name}' must have a positive 'dim'.")
    if top_level and len(indices) > 0 and output_dim != len(indices):
        raise ValueError(
            f"Top-level grouped input '{name}' has dim={output_dim} but index width={len(indices)}. "
            "Root groups must project exactly to their assigned slot count."
        )

    normalized = {
        "name": name,
        "output_dim": output_dim,
        "input": children,
    }
    if len(indices) > 0:
        normalized["index"] = list(indices)
    return normalized


def _normalize_group_children(name: str, raw_inputs: list) -> list:
    normalized = []
    for child in raw_inputs:
        if isinstance(child, str):
            normalized.append(_normalize_part_feature_name(child))
            continue

        if not isinstance(child, dict) or len(child) != 1:
            raise ValueError(
                f"Subgroup '{name}' must contain strings or single-key subgroup mappings."
            )

        child_name, child_value = next(iter(child.items()))
        if _is_group_config(child_value):
            normalized_child = _normalize_group_node(child_name, child_value, top_level=False)
        elif isinstance(child_value, list):
            normalized_child = {
                "name": child_name,
                "output_dim": 1,
                "input": _normalize_group_children(child_name, child_value),
            }
        else:
            raise ValueError(
                f"Subgroup '{child_name}' under '{name}' must be a list or a dict with an 'input' field."
            )
        normalized.append({child_name: normalized_child})
    return normalized


def _build_grouped_sequential_config(
    part_cfg: dict,
) -> tuple[dict, tuple[str, ...], tuple[str, ...]] | None:
    grouped_roots = []
    for root_name, raw_value in part_cfg.items():
        if isinstance(raw_value, list):
            continue
        grouped_roots.append(_normalize_group_node(root_name, raw_value, top_level=True))

    if not grouped_roots:
        return None

    # Require all top-level part inputs to use the grouped form together.
    for root_name, raw_value in part_cfg.items():
        if isinstance(raw_value, list):
            raise ValueError(
                f"Inputs.Part.{root_name} still uses the legacy flat list while grouped mode is enabled. "
                "Convert all top-level Inputs.Part entries to the grouped form."
            )

    raw_sequential_fields: list[str] = []
    seen_raw_fields: set[str] = set()
    for root in grouped_roots:
        root_leaves = _flatten_leaf_features(root["input"], allow_top_level_groups=True)
        for raw_name in root_leaves:
            if raw_name in seen_raw_fields:
                raise ValueError(f"Raw sequential feature '{raw_name}' is used multiple times in grouped inputs.")
            seen_raw_fields.add(raw_name)
            raw_sequential_fields.append(raw_name)

    projected_dim = len(raw_sequential_fields)
    projected_feature_names = list(raw_sequential_fields)
    occupied_slots: dict[int, str] = {}

    for root in grouped_roots:
        root_leaves = _flatten_leaf_features(root["input"], allow_top_level_groups=True)
        for slot in root["index"]:
            if slot < 0 or slot >= projected_dim:
                raise ValueError(
                    f"Projected sequential slot {slot} for '{root['name']}' is out of range "
                    f"for raw sequential dim {projected_dim}."
                )
            if slot in occupied_slots:
                raise ValueError(
                    f"Projected sequential slot {slot} is assigned to both "
                    f"'{occupied_slots[slot]}' and '{root['name']}'."
                )
            occupied_slots[slot] = root["name"]

        if root["name"] == "Momentum":
            expected_momentum = [f"Part_{field}" for field in FOUR_VECTOR_FEATURES]
            if root_leaves == expected_momentum and len(root["index"]) == len(expected_momentum):
                for slot, feature_name in zip(root["index"], expected_momentum):
                    projected_feature_names[slot] = feature_name
                continue

        for local_index, slot in enumerate(root["index"]):
            projected_feature_names[slot] = (
                root["name"] if len(root["index"]) == 1 else f"{root['name']}_{local_index}"
            )

    return {
        "SEQUENTIAL": {
            "Source": {
                "projected_feature_names": projected_feature_names,
                "groups": {
                    root["name"]: {
                        key: value
                        for key, value in root.items()
                        if key != "name"
                    }
                    for root in grouped_roots
                },
            }
        }
    }, tuple(raw_sequential_fields), tuple(projected_feature_names)


def parse_feature_config(config: dict) -> FeatureConfig:
    inputs_cfg = config.get("Inputs", {})

    part_cfg = inputs_cfg.get("Part", {})
    global_cfg = inputs_cfg.get("Global", {})

    momentum_raw = part_cfg.get("Momentum", FOUR_VECTOR_FEATURES)
    if isinstance(momentum_raw, list):
        momentum_fields = tuple(momentum_raw)
        if momentum_fields != tuple(FOUR_VECTOR_FEATURES):
            raise ValueError(
                "Inputs.Part.Momentum must be exactly ['energy', 'pt', 'eta', 'phi'] "
                "because the converter rewrites four-momentum into this basis."
            )
    elif isinstance(momentum_raw, dict):
        momentum_children = tuple(momentum_raw.get("input", ()))
        momentum_fields = tuple(
            child for child in momentum_children if isinstance(child, str)
        )
        if tuple(momentum_fields) != tuple(FOUR_VECTOR_FEATURES):
            raise ValueError(
                "Grouped Inputs.Part.Momentum.input must contain ['energy', 'pt', 'eta', 'phi'] "
                "as direct leaves so the projected local-point coordinates stay aligned."
            )
    else:
        raise ValueError("Inputs.Part.Momentum must be either a list or a grouped input dict.")

    grouped_result = _build_grouped_sequential_config(part_cfg)
    if grouped_result is None:
        grouped_sequential_config = None
        part_aux_fields = tuple(part_cfg.get("Auxiliary", DEFAULT_PART_AUX_FIELDS))
        raw_sequential_fields = tuple(f"Part_{field}" for field in momentum_fields) + tuple(part_aux_fields)
        projected_sequential_feature_names = raw_sequential_fields
    else:
        grouped_sequential_config, raw_sequential_fields, projected_sequential_feature_names = grouped_result
        part_aux_fields = tuple(
            field
            for field in raw_sequential_fields
            if field not in {f"Part_{field}" for field in FOUR_VECTOR_FEATURES}
        )

    global_fields = tuple(global_cfg.get("Fields", DEFAULT_GLOBAL_FIELDS))

    if len(set(part_aux_fields)) != len(part_aux_fields):
        raise ValueError("Inputs.Part.Auxiliary contains duplicate fields.")
    if len(set(global_fields)) != len(global_fields):
        raise ValueError("Inputs.Global.Fields contains duplicate fields.")
    if len(set(raw_sequential_fields)) != len(raw_sequential_fields):
        raise ValueError("Inputs.Part contains duplicate raw sequential features.")

    return FeatureConfig(
        part_momentum_fields=tuple(FOUR_VECTOR_FEATURES),
        part_aux_fields=part_aux_fields,
        global_fields=global_fields,
        raw_sequential_fields=raw_sequential_fields,
        projected_sequential_feature_names=projected_sequential_feature_names,
        grouped_sequential_config=grouped_sequential_config,
    )


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


def _parse_process_topologies(raw_processes: dict) -> dict[str, dict[str, tuple[str, ...]]]:
    process_topologies: dict[str, dict[str, tuple[str, ...]]] = {}
    for process_name, resonance_map in raw_processes.items():
        process_topologies[process_name] = {
            resonance_name: tuple(products)
            for resonance_name, products in resonance_map.items()
        }
    return process_topologies


def parse_evenet_config(config: dict, feature_config: FeatureConfig) -> EveNetConfig:
    if "Processes" in config or "Classification" in config or "Generations" in config:
        evenet_cfg = config
    else:
        evenet_cfg = config.get("EveNet", {})

    raw_processes = evenet_cfg.get("Processes", {})
    if not raw_processes:
        raise ValueError(
            "EveNet schema is missing Processes for multi-process event_info generation."
        )

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
