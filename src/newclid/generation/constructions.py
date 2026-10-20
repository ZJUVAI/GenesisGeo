"""
Construction set loading and config resolution for geometry generation.

Default construction data is stored in `constructions.json` so experiments can
track it as data rather than large Python constants.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

STEP_KEYS = (
    "step1_sets",
    "step2_intersect_sets",
    "step3_single_sets",
)


@lru_cache(maxsize=1)
def load_default_construction_config() -> dict:
    """Load the bundled default construction config once per process."""
    config_path = Path(__file__).with_name("constructions.json")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Default construction config must contain a top-level object.")
    return config


def _copy_named_lists(source: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: list(values) for name, values in source.items()}


def _default_sets() -> dict[str, list[str]]:
    config = load_default_construction_config()
    sets = config.get("construction_sets", {})
    if not isinstance(sets, dict):
        raise ValueError(
            "Default construction config field 'construction_sets' must be a dict."
        )
    return _copy_named_lists(sets)


CONSTRUCTION_SETS = _default_sets()

BASIC = list(CONSTRUCTION_SETS["BASIC"])
BASIC_FREE = list(CONSTRUCTION_SETS["BASIC_FREE"])
INTERSECT = list(CONSTRUCTION_SETS["INTERSECT"])
OTHER = list(CONSTRUCTION_SETS["OTHER"])


def _resolve_config_source(
    construction_config: dict | None,
) -> dict:
    if not construction_config:
        return deepcopy(load_default_construction_config())
    if not isinstance(construction_config, dict):
        raise ValueError("Construction config must be a dict.")
    return deepcopy(construction_config)


def _expand_set_names(
    set_names: list[str],
    all_sets: dict[str, list[str]],
    available_constructions: set[str] | None,
    step_key: str,
) -> list[str]:
    if not isinstance(set_names, list) or not all(
        isinstance(v, str) for v in set_names
    ):
        raise ValueError(
            f"Construction config field '{step_key}' must be a list of set names."
        )

    expanded: list[str] = []
    seen: set[str] = set()
    for set_name in set_names:
        if set_name not in all_sets:
            raise ValueError(
                f"Construction config references unknown construction set '{set_name}'."
            )
        for construction in all_sets[set_name]:
            if (
                available_constructions is not None
                and construction not in available_constructions
            ):
                raise ValueError(
                    f"Construction '{construction}' from set '{set_name}' is not defined in construction defs."
                )
            if construction not in seen:
                expanded.append(construction)
                seen.add(construction)

    if not expanded:
        raise ValueError(
            f"Construction config expands to an empty pool for '{step_key}'."
        )

    return expanded


def resolve_construction_config(
    construction_config: dict | None = None,
    available_constructions: set[str] | None = None,
) -> dict[str, list[str]]:
    """
    Resolve a construction config into the three sampling pools used by ProblemSampler.

    Args:
        construction_config: Optional external config. When provided, it is used
            as the only config source instead of the bundled default config.
        available_constructions: Optional set of valid construction names for validation.

    Returns:
        Dict with keys `step1_pool`, `step2_pool`, `step3_pool`, plus
        `weights` (dict[str, float]) and `compatibility` (dict[str, dict[str, float]]).
    """
    config = _resolve_config_source(construction_config)
    sets = config.get("construction_sets", {})
    if not isinstance(sets, dict):
        raise ValueError(
            "Construction config field 'construction_sets' must be a dict."
        )
    sets = _copy_named_lists(sets)
    for set_name, values in sets.items():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(
                f"Construction set '{set_name}' must be a list of strings."
            )

    missing = [key for key in STEP_KEYS if key not in config]
    if missing:
        raise ValueError(
            f"Construction config is missing required keys: {', '.join(missing)}."
        )

    # Load optional weights (default 1.0 for any construction not listed)
    weights: dict[str, float] = {}
    raw_weights = config.get("construction_weights", {})
    if isinstance(raw_weights, dict):
        for k, v in raw_weights.items():
            if isinstance(k, str) and not k.startswith("_") and isinstance(v, (int, float)):
                weights[k] = float(v)

    # Load optional compatibility matrix
    compatibility: dict[str, dict[str, float]] = {}
    raw_compat = config.get("construction_compatibility", {})
    if isinstance(raw_compat, dict):
        for src, targets in raw_compat.items():
            if isinstance(src, str) and not src.startswith("_") and isinstance(targets, dict):
                compatibility[src] = {}
                for tgt, factor in targets.items():
                    if isinstance(tgt, str) and isinstance(factor, (int, float)):
                        compatibility[src][tgt] = float(factor)

    # Strength of argument reuse (see ProblemSampler._map_construction_args).
    # 0.0 keeps the historical "every argument a distinct point" behaviour.
    raw_repetition = config.get("construction_arg_repetition", 0.0)
    arg_repetition = (
        float(raw_repetition) if isinstance(raw_repetition, (int, float)) else 0.0
    )

    return {
        "arg_repetition": arg_repetition,
        "step1_pool": _expand_set_names(
            config["step1_sets"],
            sets,
            available_constructions,
            "step1_sets",
        ),
        "step2_pool": _expand_set_names(
            config["step2_intersect_sets"],
            sets,
            available_constructions,
            "step2_intersect_sets",
        ),
        "step3_pool": _expand_set_names(
            config["step3_single_sets"],
            sets,
            available_constructions,
            "step3_single_sets",
        ),
        "weights": weights,
        "compatibility": compatibility,
    }
