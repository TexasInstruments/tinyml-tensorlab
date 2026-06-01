from typing import Optional, Dict, List, Tuple, Any, Union, Set
import os
import re
import json
import yaml
from pathlib import Path
from constants import TASK_TYPE_TO_MODULE


# ── Model complexity helpers ──────────────────────────────────────────────────

# Param count thresholds defining complexity tiers.
# Used to map dataset size_bucket → preferred model complexity.
_COMPLEXITY_TIERS = {
    "micro":  (0,      1_000),   # ≤ 1k params
    "tiny":   (1_001,  4_000),   # 1k–4k params
    "small":  (4_001,  10_000),  # 4k–10k params
    "medium": (10_001, 30_000),  # 10k–30k params
    "large":  (30_001, None),    # > 30k params
}

# Maximum param count preferred per dataset size_bucket.
# None = no restriction (larger models are fine).
_DATASET_PREFERRED_MAX_PARAMS = {
    "tiny":   2_000,    # tiny dataset (<500 samples)  → micro/tiny models
    "small":  10_000,   # small dataset (500–4,999)     → up to small models
    "medium": 30_000,   # medium dataset (5k–49,999)    → up to medium models
    "large":  None,     # large dataset (≥50k)          → no restriction
}


def _parse_model_params(model_name: str) -> Optional[int]:
    """Extract approximate parameter count from a model name.

    Handles:
      - '<N>k' / '<N>K' suffix  → N * 1000   (e.g. CLS_13k_NPU → 13000)
      - Standalone integers ≥ 100             (e.g. CLS_500_NPU → 500,
                                               ArcFault_model_1400_t → 1400)
    Returns None when no parseable count is found
    (e.g. MotorFault_model_1_t, AD_Linear, FCST_LSTM8).
    """
    m = re.search(r"(\d+)[kK]", model_name)
    if m:
        return int(m.group(1)) * 1000
    candidates = [int(n) for n in re.findall(r"\d+", model_name) if int(n) >= 100]
    return max(candidates) if candidates else None


def _complexity_tier(param_count: int) -> str:
    for tier, (lo, hi) in _COMPLEXITY_TIERS.items():
        if hi is None and param_count > lo:
            return tier
        if hi is not None and lo <= param_count <= hi:
            return tier
    return "micro"


class ExampleFinder:
    """Find matching examples from tinyml-modelzoo/examples"""

    @staticmethod
    def find_examples_path() -> Optional[str]:
        """
        Find the path to tinyml-modelzoo examples directory.
        Searches TINYML_MODELZOO_PATH (repo root) first, then common default locations.
        """
        EXAMPLES_SUBPATH = "examples"

        env_root = os.environ.get("TINYML_MODELZOO_PATH")
        if env_root:
            candidate = os.path.join(env_root, EXAMPLES_SUBPATH)
            if os.path.isdir(candidate):
                return candidate

        default_roots = [
            os.path.expanduser("~/tinyml-tensorlab/tinyml-modelzoo"),
            os.path.expanduser("~/tinyml-modelzoo"),
            os.path.expanduser("~/projects/tinyml-modelzoo"),
            "/opt/tinyml-modelzoo",
        ]
        for root in default_roots:
            candidate = os.path.join(root, EXAMPLES_SUBPATH)
            if os.path.isdir(candidate):
                return candidate

        return None

    @staticmethod
    def parse_example_config(config_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a config.yaml file from an example.
        Returns the common and training section config, or None if parse fails.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            return None

    @staticmethod
    def get_example_metadata(example_dir: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from an example directory.
        Looks for config.yaml to get task_type, target_device, and model info.
        """
        config_path = os.path.join(example_dir, "config.yaml")
        if not os.path.exists(config_path):
            return None

        config = ExampleFinder.parse_example_config(config_path)
        if not config:
            return None

        try:
            common = config.get("common", {})
            training = config.get("training", {})
            feat_ext = config.get("data_processing_feature_extraction", {})

            task_type = common.get("task_type")
            # Infer target_module from task_type when not explicitly in config
            target_module = common.get("target_module") or TASK_TYPE_TO_MODULE.get(task_type)

            # Normalize variables to an integer count
            raw_vars = feat_ext.get("variables")
            if isinstance(raw_vars, list):
                variables = len(raw_vars)
            elif isinstance(raw_vars, int):
                variables = raw_vars
            else:
                variables = None

            return {
                "task_type": task_type,
                "target_device": common.get("target_device"),
                "target_module": target_module,
                "variables": variables,
                "model_name": training.get("model_name"),
                "example_dir": example_dir,
                "config_path": config_path,
                "full_config": config,
            }
        except Exception:
            return None

    @staticmethod
    def list_all_examples(examples_root: str) -> List[Dict[str, Any]]:
        """
        List all valid examples with their metadata.
        Returns list of example metadata dicts.
        """
        examples = []

        if not os.path.isdir(examples_root):
            return examples

        for item in os.listdir(examples_root):
            example_dir = os.path.join(examples_root, item)
            if not os.path.isdir(example_dir):
                continue

            metadata = ExampleFinder.get_example_metadata(example_dir)
            if metadata:
                examples.append(metadata)

        return examples


class ModelSelector:
    """Select ideal model based on closest matching example"""

    @staticmethod
    def calculate_match_score(
        example: Dict[str, Any],
        target_task_type: str,
        target_device: str,
        target_module: str,
        target_variables: Optional[int] = None,
        dataset_size_bucket: Optional[str] = None,
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Calculate match score for an example (0–6 points).
        Returns (score, breakdown dict) where breakdown shows which criteria matched.

        Criteria:
        - Task type match:                    +1 point
        - Target device match:                +1 point
        - Target module match:                +1 point (inferred from task_type when absent)
        - Variables count exact match:        +2 points (strongest differentiator)
        - Dataset size complexity match:      +1 point (model param count suits dataset size)
          Only scored when dataset_size_bucket is provided. Fires when the model's
          approximate parameter count (parsed from its name) is within the preferred
          maximum for the given dataset size bucket:
            tiny   (<500 samples)  → prefer models ≤ 2,000 params
            small  (500-4,999)     → prefer models ≤ 10,000 params
            medium (5,000-49,999)  → prefer models ≤ 30,000 params
            large  (≥50,000)       → no restriction (complexity match always fires)
        """
        score = 0
        breakdown = {
            "task_type_match": False,
            "device_match": False,
            "module_match": False,
            "variables_match": False,
            "dataset_size_match": False,
        }

        if example.get("task_type") == target_task_type:
            score += 1
            breakdown["task_type_match"] = True

        if example.get("target_device") == target_device:
            score += 1
            breakdown["device_match"] = True

        if example.get("target_module") == target_module:
            score += 1
            breakdown["module_match"] = True

        if target_variables is not None and example.get("variables") == target_variables:
            score += 2
            breakdown["variables_match"] = True

        if dataset_size_bucket:
            max_params = _DATASET_PREFERRED_MAX_PARAMS.get(dataset_size_bucket)
            model_name = example.get("model_name", "")
            param_count = _parse_model_params(model_name)
            if max_params is None:
                # large dataset — any model complexity is fine
                score += 1
                breakdown["dataset_size_match"] = True
            elif param_count is not None and param_count <= max_params:
                score += 1
                breakdown["dataset_size_match"] = True

        return score, breakdown

    @staticmethod
    def find_best_matching_example(
        task_type: str,
        target_device: str,
        target_module: str,
        examples: List[Dict[str, Any]],
        target_variables: Optional[int] = None,
        dataset_size_bucket: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find best matching example. Returns example dict or None."""
        if not examples:
            return None

        best_example = None
        best_score = -1

        for example in examples:
            score, _ = ModelSelector.calculate_match_score(
                example, task_type, target_device, target_module,
                target_variables, dataset_size_bucket,
            )
            if score > best_score:
                best_score = score
                best_example = example

        return best_example

    @staticmethod
    def get_model_recommendations(
        examples: List[Dict[str, Any]],
        target_task_type: str,
        target_device: str,
        target_module: str,
        target_variables: Optional[int] = None,
        dataset_size_bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get top model recommendations with scoring details.
        Returns dict with best match, ranked list, and dataset complexity context.
        """
        if not examples:
            return {
                "success": False,
                "error": "No examples found in tinyml-modelzoo/examples",
                "recommended_model": None,
                "ranked_matches": [],
            }

        scored_examples = []
        for example in examples:
            score, breakdown = ModelSelector.calculate_match_score(
                example, target_task_type, target_device, target_module,
                target_variables, dataset_size_bucket,
            )
            model_name = example.get("model_name", "")
            param_count = _parse_model_params(model_name)
            scored_examples.append({
                "example": example,
                "score": score,
                "breakdown": breakdown,
                "param_count": param_count,
                "complexity_tier": _complexity_tier(param_count) if param_count is not None else "unknown",
            })

        scored_examples.sort(key=lambda x: x["score"], reverse=True)

        best_match = scored_examples[0]["example"] if scored_examples else None

        # Build dataset complexity note for agent to communicate to user
        complexity_note = None
        if dataset_size_bucket:
            max_p = _DATASET_PREFERRED_MAX_PARAMS.get(dataset_size_bucket)
            if max_p:
                complexity_note = (
                    f"Dataset size bucket: '{dataset_size_bucket}'. "
                    f"Preferred models with ≤{max_p:,} parameters. "
                    f"Simpler models reduce overfitting risk on smaller datasets."
                )
            else:
                complexity_note = (
                    f"Dataset size bucket: '{dataset_size_bucket}'. "
                    "No model complexity restriction — dataset is large enough for any model."
                )

        return {
            "success": True,
            "recommended_model": best_match.get("model_name") if best_match else None,
            "recommended_example_dir": best_match.get("example_dir") if best_match else None,
            "match_score": scored_examples[0]["score"] if scored_examples else 0,
            "match_breakdown": scored_examples[0]["breakdown"] if scored_examples else {},
            "dataset_complexity_note": complexity_note,
            "ranked_matches": [
                {
                    "model_name": item["example"].get("model_name"),
                    "task_type": item["example"].get("task_type"),
                    "target_device": item["example"].get("target_device"),
                    "target_module": item["example"].get("target_module"),
                    "variables": item["example"].get("variables"),
                    "param_count": item["param_count"],
                    "complexity_tier": item["complexity_tier"],
                    "score": item["score"],
                    "match_breakdown": item["breakdown"],
                }
                for item in scored_examples
            ],
            "error": None,
        }


def select_model_for_task(
    task_type: str,
    target_device: str,
    target_module: str,
    variables: Optional[int] = None,
    dataset_size_bucket: Optional[str] = None,
    modelzoo_path: 
    Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Select ideal model based on closest matching example in tinyml-modelzoo.

    Scoring criteria (max 6 points):
      +1  task_type matches example
      +1  target_device matches example
      +1  target_module matches example
      +2  variables count exactly matches example (strongest differentiator)
      +1  model complexity suits dataset size (see dataset_size_bucket)

    To use dataset size in model selection:
      1. Fetch dataset analysis details from $WORK_DIR/.tmp_data_stats.json. If this file is not present,
         call `analyse_dataset` function.
      2. Pass `dataset_bucket` from output of above function (or from $WORK_DIR/.tmp_data_stats.json) as dataset_size_bucket here.
      3. The ranked_matches in the response include param_count and complexity_tier
         for each model — show these to the user alongside the recommendation.

    Dataset size → preferred model complexity:
      'tiny'   (<500 samples)   → models ≤ 2,000 params  (+1 if matches)
      'small'  (500-4,999)      → models ≤ 10,000 params (+1 if matches)
      'medium' (5,000-49,999)   → models ≤ 30,000 params (+1 if matches)
      'large'  (≥50,000)        → no restriction          (+1 always)

    Args:
        task_type: Task type (e.g., 'motor_fault', 'generic_timeseries_classification')
        target_device: Target MCU device (e.g., 'F28P55', 'MSPM0G3507')
        target_module: Target module ('timeseries' or 'vision')
        variables: Number of input sensor channels/variables. Exact match adds +2 points.
        dataset_size_bucket: Optional. Output of analyze_dataset_for_model_guidance size_bucket
            field ('tiny' | 'small' | 'medium' | 'large'). Adds +1 to score for models
            whose parameter count suits the dataset size.
        modelzoo_path: Optional explicit path to modelzoo examples directory.

    Returns:
        Dict with:
        - recommended_model: model name to use
        - match_score: score 0-6
        - match_breakdown: which criteria matched
        - dataset_complexity_note: plain-text explanation to show the user (if dataset_size_bucket provided)
        - ranked_matches: all candidates ranked by score, each includes param_count and complexity_tier
        - error: error message if something went wrong
    """
    examples_path = modelzoo_path or ExampleFinder.find_examples_path()

    if not examples_path:
        return {
            "success": False,
            "recommended_model": None,
            "error": "Could not find tinyml-modelzoo/examples directory. Please provide modelzoo_path parameter.",
            "match_score": 0,
            "match_breakdown": {},
            "dataset_complexity_note": None,
            "ranked_matches": [],
        }

    examples = ExampleFinder.list_all_examples(examples_path)

    if not examples:
        return {
            "success": False,
            "recommended_model": None,
            "error": f"No valid examples found in {examples_path}",
            "match_score": 0,
            "match_breakdown": {},
            "dataset_complexity_note": None,
            "ranked_matches": [],
        }

    return ModelSelector.get_model_recommendations(
        examples, task_type, target_device, target_module,
        variables, dataset_size_bucket,
    )


class ModelDescriptionsParser:
    """Parse tinyml-modelzoo model_descriptions .py files without importing them."""

    # Default task type per description file (used when a model doesn't override task_type)
    FILE_DEFAULT_TASK_TYPE = {
        "classification.py": "generic_timeseries_classification",
        "regression.py": "generic_timeseries_regression",
        "anomalydetection.py": "generic_timeseries_anomalydetection",
        "forecasting.py": "generic_timeseries_forecasting",
    }

    DESCRIPTION_FILES = list(FILE_DEFAULT_TASK_TYPE.keys())

    @staticmethod
    def find_model_descriptions_path() -> Optional[str]:
        """
        Locate the model_descriptions directory inside a tinyml-modelzoo repo.

        Resolution order:
          1. TINYML_MODELZOO_PATH env var — set this to the repo root in the
             'env' config block, e.g. {"TINYML_MODELZOO_PATH": "/home/user/tinyml-modelzoo"}
          2. Common default locations on disk
        """
        MODEL_DESCRIPTIONS_SUBPATH = os.path.join("tinyml_modelzoo", "model_descriptions")

        # 1. Env var pointing at the repo root
        env_root = os.environ.get("TINYML_MODELZOO_PATH")
        if env_root:
            candidate = os.path.join(env_root, MODEL_DESCRIPTIONS_SUBPATH)
            if os.path.isdir(candidate):
                return candidate

        # 2. Common default locations
        default_roots = [
            os.path.expanduser("~/tinyml-tensorlab/tinyml-modelzoo"),
            os.path.expanduser("~/tinyml-modelzoo"),
            os.path.expanduser("~/projects/tinyml-modelzoo"),
            "/opt/tinyml-modelzoo",
        ]
        for root in default_roots:
            candidate = os.path.join(root, MODEL_DESCRIPTIONS_SUBPATH)
            if os.path.isdir(candidate):
                return candidate

        return None

    @staticmethod
    def _parse_constants(descriptions_dir: str) -> Dict[str, str]:
        """Read tinyml_modelzoo/constants.py and build a CONST_NAME -> 'value' map."""
        constants_path = os.path.join(descriptions_dir, "..", "constants.py")
        constants_path = os.path.normpath(constants_path)
        constants_map = {}
        if not os.path.isfile(constants_path):
            return constants_map
        try:
            with open(constants_path) as f:
                content = f.read()
            for m in re.finditer(r"^(\w+)\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE):
                constants_map[m.group(1)] = m.group(2)
        except Exception:
            pass
        return constants_map

    @staticmethod
    def _parse_file(filepath: str, constants_map: Dict[str, str], default_task_type: str) -> Dict[str, Any]:
        """
        Parse a single model descriptions .py file.
        Extracts enabled model names, task types, model details, and supported devices.
        """
        models = {}
        try:
            with open(filepath) as f:
                content = f.read()
        except Exception:
            return models

        # Extract enabled_models_list
        enabled_match = re.search(r"enabled_models_list\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not enabled_match:
            return models
        enabled_models = set(re.findall(r"'([\w]+)'", enabled_match.group(1)))

        # Split content into per-model blocks by finding top-level dict entries
        # Each entry starts with 4-space indent + quoted key + colon
        entry_positions = [(m.group(1), m.start()) for m in re.finditer(
            r"^\s{4}'(\w+)'\s*:\s+deep_update_dict", content, re.MULTILINE
        )]

        for i, (model_name, start_pos) in enumerate(entry_positions):
            if model_name not in enabled_models:
                continue

            end_pos = entry_positions[i + 1][1] if i + 1 < len(entry_positions) else len(content)
            block = content[start_pos:end_pos]

            # task_type: look for explicit override, fall back to file default
            task_match = re.search(r"task_type\s*=\s*constants\.(\w+)", block)
            if task_match:
                resolved = constants_map.get(task_match.group(1), default_task_type)
            else:
                resolved = default_task_type

            # model_details
            details_match = re.search(r"model_details\s*=\s*'([^']*)'", block)
            model_details = details_match.group(1) if details_match else ""

            # supported devices: all constants.TARGET_DEVICE_XXX keys in target_devices block
            target_devices_match = re.search(r"target_devices\s*=\s*\{(.*?)\}", block, re.DOTALL)
            devices = []
            if target_devices_match:
                for dm in re.finditer(r"constants\.(TARGET_DEVICE_\w+)\s*:", target_devices_match.group(1)):
                    device_val = constants_map.get(dm.group(1))
                    if device_val:
                        devices.append(device_val)

            models[model_name] = {
                "model_name": model_name,
                "task_type": resolved,
                "model_details": model_details,
                "supported_devices": devices,
                "device_count": len(devices),
            }

        return models

    @classmethod
    def parse_all(cls, descriptions_dir: str) -> Dict[str, Any]:
        """Parse all model description files and return a flat dict of model_name -> info."""
        constants_map = cls._parse_constants(descriptions_dir)
        all_models = {}
        for filename in cls.DESCRIPTION_FILES:
            filepath = os.path.join(descriptions_dir, filename)
            if not os.path.isfile(filepath):
                continue
            default_task_type = cls.FILE_DEFAULT_TASK_TYPE[filename]
            models = cls._parse_file(filepath, constants_map, default_task_type)
            all_models.update(models)
        return all_models


def list_available_models(
    task_type: Optional[str] = None,
    target_device: Optional[str] = None,
    module: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Search available models from tinyml-modelzoo model descriptions.

    Reads the model_descriptions directory from the tinyml-modelzoo source tree directly —
    no package import required. Use this tool to discover which models are available before
    selecting one. To find models for a specific target device (e.g. 'F28P55'), pass it as
    target_device. To narrow by task type (e.g. 'motor_fault') use task_type. To list all
    timeseries or vision models use module='timeseries' or module='vision'.

    Args:
        task_type: Filter by task type (e.g., 'generic_timeseries_classification', 'motor_fault')
        target_device: Filter by supported target device (e.g., 'F28P55', 'MSPM0G3507').
                       Check model descriptions for devices each model supports.
        module: Filter by module type ('timeseries' or 'vision')

    Returns:
        Dict with matched models and their device support info.
    """
    descriptions_dir = ModelDescriptionsParser.find_model_descriptions_path()

    if not descriptions_dir:
        return {
            "success": False,
            "models": {},
            "error": "Could not find tinyml-modelzoo/tinyml_modelzoo/model_descriptions directory.",
        }

    all_models = ModelDescriptionsParser.parse_all(descriptions_dir)

    if not all_models:
        return {
            "success": False,
            "models": {},
            "error": f"No models parsed from {descriptions_dir}",
        }

    # Apply filters
    matched = {}
    for model_name, info in all_models.items():
        if task_type and info.get("task_type") != task_type:
            continue
        if module and TASK_TYPE_TO_MODULE.get(info.get("task_type")) != module:
            continue
        if target_device and target_device not in info.get("supported_devices", []):
            continue
        matched[model_name] = info

    filters_applied = {"task_type": task_type, "target_device": target_device, "module": module}

    if not matched:
        active_filters = [f"{k}={v}" for k, v in filters_applied.items() if v]
        return {
            "success": False,
            "models": {},
            "error": f"No models found matching: {', '.join(active_filters)}",
            "filters_applied": filters_applied,
        }

    return {
        "success": True,
        "models": matched,
        "total_models": len(matched),
        "filters_applied": filters_applied,
        "error": None,
    }
