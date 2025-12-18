# python/trainer/common/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json
import os


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Loads a JSON config file or returns an empty dict
    
    :param path: Description
    :type path: Optional[str]
    :return: Description
    :rtype: Dict[str, Any]
    """
    if path is None:
        return {}
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        return json.load(f)