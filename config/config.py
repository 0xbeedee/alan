from typing import Any, Union
from ast import literal_eval
import yaml


class Config:
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the configuration file.

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
        """Convert a value to the appropriate type.

        Handles ints, floats (including scientific notation), booleans, and lists.
        """
        if not isinstance(value, str):
            return value

        # try to evaluate the string using python's built-in ast module
        try:
            return literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # Handle boolean strings
        lower_value = value.lower()
        if lower_value in {"true", "yes", "on"}:
            return True
        if lower_value in {"false", "no", "off"}:
            return False

        # not among the handled types => return the value as is
        return value

    def get_all(self) -> dict:
        """Return the entire configuration dictionary."""
        return self.config
