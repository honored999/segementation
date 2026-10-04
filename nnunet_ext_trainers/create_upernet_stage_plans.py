"""Generate one independent plans JSON; never preprocess or access medical data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping import resolve_upernet_selection


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"ambiguous duplicate JSON key: {key}")
        result[key] = value
    return result


def create_upernet_stage_plans(source, output, configuration: str, feature_indices) -> Path:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError("source and output must be different resolved paths")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    with source.open(encoding="utf-8") as stream:
        plans = json.load(stream, object_pairs_hook=_unique_json_object)
    if not isinstance(plans, dict) or not isinstance(plans.get("configurations"), dict):
        raise ValueError("source must be a plans JSON with configurations")
    if not isinstance(plans.get("plans_name"), str) or not plans["plans_name"]:
        raise ValueError("source requires an unambiguous nonempty plans_name")
    if configuration not in plans["configurations"]:
        raise ValueError(f"configuration {configuration!r} not found")
    if not isinstance(plans["configurations"][configuration], dict):
        raise ValueError("target configuration must be a JSON object")
    # Refuse old plans auto-repair; keep all original fields and inheritance intact.
    manager = PlansManager(deepcopy(plans))
    try:
        resolved = manager._internal_resolve_configuration_inheritance(configuration)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"invalid configuration inheritance: {error}") from error
    if "architecture" not in resolved:
        raise ValueError("source needs current architecture format; no automatic plans repair")
    if not isinstance(resolved.get("data_identifier"), str) or not resolved["data_identifier"]:
        raise ValueError("resolved configuration requires the original data_identifier")
    derived = deepcopy(plans)
    derived["configurations"][configuration]["upernet_feature_indices"] = feature_indices
    selection = resolve_upernet_selection(PlansManager(derived).get_configuration(configuration))
    expected_name = "nnUNetPlansUPerNetStages_s" + "".join(str(i) for i in selection)
    if output.suffix != ".json" or output.stem != expected_name:
        raise ValueError(f"output must be a new file named {expected_name}.json")
    derived["plans_name"] = output.stem
    # Serialize/validate everything before opening output; x also prevents race overwrite.
    text = json.dumps(derived, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--feature-indices", required=True, type=int, nargs="+")
    args = parser.parse_args()
    try:
        result = create_upernet_stage_plans(args.source, args.output, args.configuration, args.feature_indices)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(result)


if __name__ == "__main__":
    main()
