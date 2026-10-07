from typing import Optional, Dict, List, Tuple, Any
import os
import re
import yaml
from constants import TASK_TYPE_TO_MODULE


def _parse_model_params(model_name: str) -> Optional[int]:
    """Extract approximate parameter count from a model name.

    Handles '<N>k'/'<N>K' suffix and standalone integers >= 100.
    Returns None when no parseable count is found.
    """
    m = re.search(r"(\d+)[kK]", model_name)
    if m:
        return int(m.group(1)) * 1000
    candidates = [int(n) for n in re.findall(r"\d+", model_name) if int(n) >= 100]
    return max(candidates) if candidates else None


class ExampleFinder:
    """Find matching examples from tinyml-modelzoo/examples"""

    @staticmethod
    def find_examples_path() -> Optional[str]:
        EXAMPLES_SUBPATH = "examples"
        env_root = os.environ.get("TINYML_MODELZOO_PATH")
        if env_root:
            candidate = os.path.join(env_root, EXAMPLES_SUBPATH)
            if os.path.isdir(candidate):
                return candidate
        for root in [
            os.path.expanduser("~/tinyml-modelzoo"),
            os.path.expanduser("~/projects/tinyml-modelzoo"),
            "/opt/tinyml-modelzoo",
        ]:
            candidate = os.path.join(root, EXAMPLES_SUBPATH)
            if os.path.isdir(candidate):
                return candidate
        return None

    @staticmethod
    def find_model_descriptions_path() -> Optional[str]:
        """Locate tinyml_modelzoo/model_descriptions inside tinyml-modelzoo repo."""
        subpath = os.path.join("tinyml_modelzoo", "model_descriptions")
        env_root = os.environ.get("TINYML_MODELZOO_PATH")
        if env_root:
            candidate = os.path.join(env_root, subpath)
            if os.path.isdir(candidate):
                return candidate
        for root in [
            os.path.expanduser("~/tinyml-modelzoo"),
            os.path.expanduser("~/projects/tinyml-modelzoo"),
            "/opt/tinyml-modelzoo",
        ]:
            candidate = os.path.join(root, subpath)
            if os.path.isdir(candidate):
                return candidate
        return None

    @staticmethod
    def parse_example_config(config_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    @staticmethod
    def get_example_metadata(example_dir: str) -> Optional[Dict[str, Any]]:
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
            target_module = common.get("target_module") or TASK_TYPE_TO_MODULE.get(task_type)
            raw_vars = feat_ext.get("variables")
            if isinstance(raw_vars, list):
                variables = len(raw_vars)
            elif isinstance(raw_vars, int):
                variables = raw_vars
            else:
                variables = None
            example_name = os.path.basename(example_dir)
            return {
                "task_type": task_type,
                "target_device": common.get("target_device"),
                "target_module": target_module,
                "variables": variables,
                "model_name": training.get("model_name"),
                "example_dir": example_dir,
                "example_name": example_name,
                "config_path": config_path,
                "full_config": config,
            }
        except Exception:
            return None

    @staticmethod
    def list_all_examples(examples_root: str) -> List[Dict[str, Any]]:
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
    """Select ideal model based on closest matching example."""

    @staticmethod
    def _normalize_task_name(name: str) -> str:
        """Convert underscore-separated names to lowercase for comparison."""
        return name.lower().replace("_", "").replace("-", "")

    @staticmethod
    def calculate_match_score(
        example: Dict[str, Any],
        target_task_type: str,
        target_device: str,
        target_module: str,
        target_variables: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score an example against target criteria.

        For generic_* task types:
          +3  specific app name match (e.g., anti_pinch_detection vs anti_pinch_detection)
          +0.5 generic type match (fallback when no app match)
        For specific task types:
          +1  exact task_type match

        Always add:
          +1  target_device match
          +1  target_module match
          +2  variables count exact match (strongest differentiator)
        """
        score = 0.0
        breakdown = {
            "task_type_match": False,
            "app_name_match": False,
            "generic_fallback": False,
            "device_match": False,
            "module_match": False,
            "variables_match": False,
        }

        example_task_type = example.get("task_type", "")
        example_name = example.get("example_name", "")

        if target_task_type.startswith("generic_"):
            if example_task_type.startswith("generic_"):
                if ModelSelector._normalize_task_name(example_name) == ModelSelector._normalize_task_name(target_task_type):
                    score += 3.0
                    breakdown["app_name_match"] = True
                elif example_task_type == target_task_type:
                    score += 0.5
                    breakdown["generic_fallback"] = True
        else:
            if example_task_type == target_task_type:
                score += 1.0
                breakdown["task_type_match"] = True

        if example.get("target_device") == target_device:
            score += 1.0
            breakdown["device_match"] = True
        if example.get("target_module") == target_module:
            score += 1.0
            breakdown["module_match"] = True
        if target_variables is not None and example.get("variables") == target_variables:
            score += 2.0
            breakdown["variables_match"] = True
        return score, breakdown

    @staticmethod
    def get_model_recommendations(
        examples: List[Dict[str, Any]],
        target_task_type: str,
        target_device: str,
        target_module: str,
        target_variables: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get top model recommendations ranked by match score.

        For generic_* task types: triggers fallback if best score < 1 (no app match found).
        For specific task types: has_good_match requires task_type_match (score includes +1).
        """
        if not examples:
            return {
                "success": False,
                "has_good_match": False,
                "error": "No examples found in tinyml-modelzoo/examples",
                "recommended_model": None,
                "ranked_matches": [],
            }

        scored = []
        for example in examples:
            score, breakdown = ModelSelector.calculate_match_score(
                example, target_task_type, target_device, target_module, target_variables
            )
            model_name = example.get("model_name", "")
            scored.append({
                "example": example,
                "score": score,
                "breakdown": breakdown,
                "param_count": _parse_model_params(model_name),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]

        if target_task_type.startswith("generic_"):
            has_good_match = best["score"] >= 1.0
        else:
            has_good_match = best["breakdown"].get("task_type_match", False)

        return {
            "success": True,
            "has_good_match": has_good_match,
            "recommended_model": best["example"].get("model_name") if has_good_match else None,
            "recommended_example_dir": best["example"].get("example_dir") if has_good_match else None,
            "match_score": best["score"],
            "match_breakdown": best["breakdown"],
            "ranked_matches": [
                {
                    "model_name": item["example"].get("model_name"),
                    "task_type": item["example"].get("task_type"),
                    "example_name": item["example"].get("example_name"),
                    "target_device": item["example"].get("target_device"),
                    "target_module": item["example"].get("target_module"),
                    "variables": item["example"].get("variables"),
                    "param_count": item["param_count"],
                    "score": item["score"],
                    "match_breakdown": item["breakdown"],
                }
                for item in scored
            ],
            "error": None,
        }


def select_model_for_task(
    task_type: str,
    target_device: str,
    target_module: str,
    variables: Optional[int] = None,
    modelzoo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: Select model by finding the closest matching example in tinyml-modelzoo/examples.

    PRIMARY PATH — example-based selection:
      Scans tinyml-modelzoo/examples/, scores each against the target, returns best match.

    For SPECIFIC task types (motor_fault, ac_arc_fault, etc):
      Scoring (max 5 points):
        +1  task_type exact match
        +1  target_device matches example
        +1  target_module matches example
        +2  variables count exactly matches example

    For GENERIC task types (generic_timeseries_classification, etc):
      Scoring (max 7 points):
        +3  app name match (e.g., anti_pinch_detection matches example dir name)
        +0.5 generic type match (fallback when no specific app match found)
        +1  target_device matches example
        +1  target_module matches example
        +2  variables count exactly matches example

    FALLBACK — when has_good_match is False:
      For generic_* with best_score < 1: no matching app example found.
      Includes fallback dict with model_descriptions_path — read those files
      to discover models, then use dataset analysis to select by complexity.

    Args:
        task_type: Task type (e.g., 'motor_fault', 'generic_timeseries_classification')
        target_device: Target MCU (e.g., 'F28P55', 'MSPM0G3507')
        target_module: Target module ('timeseries' or 'vision')
        variables: Number of input sensor channels/variables. Exact match adds +2.
        modelzoo_path: Optional explicit path to modelzoo examples directory.

    Returns:
        has_good_match, recommended_model, match_score, match_breakdown, ranked_matches.
        Includes fallback dict when has_good_match is False.
    """
    examples_path = modelzoo_path or ExampleFinder.find_examples_path()
    descriptions_path = ExampleFinder.find_model_descriptions_path()

    if not examples_path:
        return {
            "success": False,
            "has_good_match": False,
            "recommended_model": None,
            "error": "Could not find tinyml-modelzoo/examples directory.",
            "fallback": {
                "instruction": "Read model_descriptions files to select a model manually.",
                "model_descriptions_path": descriptions_path,
            },
            "match_score": 0,
            "match_breakdown": {},
            "ranked_matches": [],
        }

    examples = ExampleFinder.list_all_examples(examples_path)

    if not examples:
        return {
            "success": False,
            "has_good_match": False,
            "recommended_model": None,
            "error": f"No valid examples found in {examples_path}",
            "fallback": {
                "instruction": "Read model_descriptions files to discover available models, "
                "then use dataset analysis stats to guide model complexity selection.",
                "model_descriptions_path": descriptions_path,
            },
            "match_score": 0,
            "match_breakdown": {},
            "ranked_matches": [],
        }

    result = ModelSelector.get_model_recommendations(
        examples, task_type, target_device, target_module, variables
    )

    if not result.get("has_good_match"):
        result["fallback"] = {
            "instruction": (
                "No matching example found. "
                "Read model_descriptions files to discover available models, "
                "then use dataset analysis stats to guide model complexity selection."
            ),
            "model_descriptions_path": descriptions_path,
        }

    return result
