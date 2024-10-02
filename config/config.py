from typing import Any, Union, List, Dict
from ast import literal_eval
from pathlib import Path
import yaml


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_dir = Path(config_path).parent
        self.config = self.load_yaml(config_path)

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the configuration file.

        Handles nested keys, type conversion, and default values.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return self._convert_value(value)

    def _convert_value(self, value: Any) -> Union[int, float, bool, str, list, None]:
        """Converts a value to the appropriate Python type.

        Handles ints, floats (including scientific notation), booleans, and lists.
        """
        if not isinstance(value, str):
            return value

        # try to evaluate the string using python's built-in ast module
        try:
            return literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # handle boolean strings
        lower_value = value.lower()
        if lower_value in {"true", "yes", "on"}:
            return True
        if lower_value in {"false", "no", "off"}:
            return False

        # not among the handled types => return the value as is
        return value

    def get_all(self) -> dict:
        """Returns the entire configuration dictionary."""
        return self.config

    def get_except(self, key: str, exclude: Union[str, List[str]] = None) -> dict:
        """Retrieves all fields from a specific entry, optionally excluding certain keys."""
        if isinstance(exclude, str):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        nested_dict = self.get(key, {})
        if not isinstance(nested_dict, dict):
            return {}
        return {k: v for k, v in nested_dict.items() if k not in exclude}

    def merge_configs(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merges two configuration dictionaries."""
        merged = config1.copy()
        for key, value in config2.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def load_subconfig(self, subconfig_name: str, section: str) -> Dict[str, Any]:
        """Loads a subconfig file from the appropriate directory."""
        subconfig_path = self.config_dir / section / f"{subconfig_name}.yaml"
        return self.load_yaml(str(subconfig_path))

    def create_config(self, subconfigs: Dict[str, str]) -> None:
        """Creates a new configuration by merging subconfigs with the base config."""
        for section, subconfig_name in subconfigs.items():
            if subconfig_name:
                subconfig = self.load_subconfig(subconfig_name, section)
                self.config = self.merge_configs(self.config, {section: subconfig})
