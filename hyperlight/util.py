import copy
from typing import Any, Dict, List

from pydantic import validate_arguments

validate_types = validate_arguments(config=dict(arbitrary_types_allowed=True))


def without_keys(mapping: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Returns a copy of the given dictionary without the specified keys

    Args:
        mapping (Dict[str, Any]): The dictionary from which the keys are to be removed.
        keys (List[str]): The list of keys to be removed from the dictionary.

    Returns:
        Dict[str, Any]: A new dictionary that does not contain the specified keys.
    """
    return {k: copy.deepcopy(v) for k, v in mapping.items() if k not in keys}
