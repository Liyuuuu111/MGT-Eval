"""YAML Service for template loading and validation"""

import os
import yaml
import json
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


def normalize_file_path(path: str) -> str:
    """
    Normalize file path to ensure only '/' is used as directory separator.
    This prevents '.' from being misinterpreted as a path separator.

    Args:
        path: Input file path

    Returns:
        Normalized path string
    """
    if not path:
        return path

    # Use Path to properly handle the path
    # Path only treats '/' (and os.sep) as separators, never '.'
    p = Path(path)

    # Return normalized string
    return str(p)


class YAMLService:
    """Service for loading and validating YAML templates"""

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            # Auto-detect project root (backend is in project_root/backend/)
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        self.examples_dir = self.project_root / "examples"

    def _is_writable_dir(self, directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".mgt_eval_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _resolve_temp_yaml_dir(self) -> Path:
        env_keys = ("MGT_EVAL_BACKEND_TMP_DIR", "UNCOVERAI_BACKEND_TMP_DIR")

        for env_key in env_keys:
            raw = os.getenv(env_key, "").strip()
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if self._is_writable_dir(candidate):
                return candidate

        candidates = [
            Path(tempfile.gettempdir()) / "mgt_eval_backend",
            Path.home() / ".mgt_eval" / "backend_tmp",
            self.project_root / ".runtime" / "backend_tmp",
        ]
        for candidate in candidates:
            if self._is_writable_dir(candidate):
                return candidate

        raise PermissionError(
            "No writable temporary directory available for YAML files. "
            "Set MGT_EVAL_BACKEND_TMP_DIR to a writable path."
        )

    def load_template(self, section: str, detector_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load YAML template for a section

        Args:
            section: One of 'build', 'attack', 'train', 'detect'
            detector_name: Required for train/detect sections

        Returns:
            Dictionary containing the YAML configuration
        """
        if section == "build":
            path = self.examples_dir / "build" / "build_dataset.yaml"
        elif section == "attack":
            path = self.examples_dir / "attack" / "build_attack_dataset_inline.yaml"
        elif section in ["train", "detect"]:
            if not detector_name:
                raise ValueError(f"detector_name required for {section}")
            path = self.examples_dir / section / f"{detector_name}.yaml"
        else:
            raise ValueError(f"Invalid section: {section}")

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f) or {}

        return self._inject_runtime_defaults(section, template)

    def _inject_runtime_defaults(self, section: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject runtime-aware defaults for frontend templates.

        Current behavior:
        - For build/attack templates, prefill vLLM tensor parallel size with the
          number of visible GPUs (fallback to 1 when unavailable).
        """
        if not isinstance(template, dict):
            return template

        if section in {"build", "attack"} and "vllm_tensor_parallel_size" in template:
            gpu_count = 0
            try:
                from backend.services.system_service import SystemService
                gpu_count = len(SystemService().detect_gpus())
            except Exception:
                gpu_count = 0

            template["vllm_tensor_parallel_size"] = max(1, int(gpu_count or 0))

        return template

    def load_attacks_json(self) -> Dict[str, Any]:
        """Load all attack types from attacks_all.json"""
        path = self.examples_dir / "attack" / "attacks_all.json"
        if not path.exists():
            raise FileNotFoundError(f"Attacks file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_detectors(self, section: str) -> List[str]:
        """
        List available detector templates

        Args:
            section: Either 'train' or 'detect'

        Returns:
            List of detector names (without .yaml extension)
        """
        if section not in ["train", "detect"]:
            raise ValueError(f"Invalid section for detector listing: {section}")

        dir_path = self.examples_dir / section
        if not dir_path.exists():
            return []

        return sorted([f.stem for f in dir_path.glob("*.yaml")])

    def validate_config(self, section: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration based on section requirements

        Args:
            section: One of 'build', 'attack', 'train', 'detect'
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Normalize file paths in config
        if 'data' in config and config['data']:
            config['data'] = normalize_file_path(config['data'])
        if 'out' in config and config['out']:
            config['out'] = normalize_file_path(config['out'])

        if section == "build":
            # Required fields for build
            if not config.get("data"):
                errors.append("Field 'data' is required")
            if not config.get("out"):
                errors.append("Field 'out' is required")
            # Validate backend
            backend = config.get("backend", "hf")
            if backend not in ["hf", "api"]:
                errors.append(f"Invalid backend: {backend}. Must be 'hf' or 'api'")

            # Optional dataset split validation
            enable_split_raw = config.get("enable_dataset_split", False)
            enable_split = bool(enable_split_raw)
            if isinstance(enable_split_raw, str):
                enable_split = enable_split_raw.strip().lower() in {"1", "true", "yes", "on"}

            ratio_keys = ["split_train_ratio", "split_dev_ratio", "split_test_ratio"]
            ratios: Dict[str, int] = {}
            for key in ratio_keys:
                raw = config.get(key, 0)
                try:
                    val = int(raw)
                except Exception:
                    errors.append(f"Field '{key}' must be an integer")
                    continue
                if val < 0 or val > 10:
                    errors.append(f"Field '{key}' must be in range [0, 10]")
                    continue
                ratios[key] = val

            if enable_split and len(ratios) == 3:
                if (ratios["split_train_ratio"] + ratios["split_dev_ratio"] + ratios["split_test_ratio"]) <= 0:
                    errors.append("When 'enable_dataset_split' is enabled, at least one split ratio must be > 0")

        elif section == "attack":
            if not config.get("data"):
                errors.append("Field 'data' is required")
            if not config.get("out"):
                errors.append("Field 'out' is required")
            # attacks_config can be None if no attacks are selected

        elif section == "train":
            if not config.get("detector"):
                errors.append("Field 'detector' is required")
            if not config.get("dataset_train"):
                errors.append("Field 'dataset_train' is required")

        elif section == "detect":
            if not config.get("detector"):
                errors.append("Field 'detector' is required")
            if not config.get("data"):
                errors.append("Field 'data' is required")
        else:
            errors.append(f"Invalid section: {section}")

        return len(errors) == 0, errors

    def save_temp_yaml(self, config: Dict[str, Any], prefix: str = "temp") -> Path:
        """
        Save configuration to a temporary YAML file

        Args:
            config: Configuration dictionary
            prefix: Filename prefix

        Returns:
            Path to the created temporary file
        """
        import uuid

        temp_dir = self._resolve_temp_yaml_dir()

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.yaml"
        temp_path = temp_dir / filename

        # Clean up empty attack_dataset fields before saving
        cleaned_config = dict(config)
        for key in ["attack_dataset", "attack_dataset_only"]:
            if key in cleaned_config:
                value = cleaned_config[key]
                # Remove if None, empty string, or whitespace-only string
                if value is None or (isinstance(value, str) and not value.strip()):
                    del cleaned_config[key]

        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cleaned_config, f, default_flow_style=False, sort_keys=False)

        return temp_path
