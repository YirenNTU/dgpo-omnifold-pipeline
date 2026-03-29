from __future__ import annotations

from dataclasses import dataclass

from evenet_parquet_common import DEFAULT_GLOBAL_FIELDS, DEFAULT_PART_AUX_FIELDS, FOUR_VECTOR_FEATURES


@dataclass(frozen=True)
class FeatureConfig:
    part_momentum_fields: tuple[str, ...]
    part_aux_fields: tuple[str, ...]
    global_fields: tuple[str, ...]


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
