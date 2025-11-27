#!/usr/bin/env python3
"""
KCOR Mortality Configuration Management

This module provides configuration loading and management for the KCOR mortality analysis pipeline.

USAGE:
    from kcor_mortality_config import load_config, get_default_config
    
    config = load_config("config/kcor_mortality_config.yaml")
"""

import os
from typing import Dict, Optional, Any
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration dictionary.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "enrollment_dates": [
            {"year": 2021, "month": 1},
            {"year": 2021, "month": 7},
            {"year": 2022, "month": 1}
        ],
        "quiet_periods": [
            {"min": 3, "max": 10},
            {"min": 6, "max": 15},
            {"min": 9, "max": 18}
        ],
        "follow_up_months": [12, 18, 24],
        "cohort_definitions": [
            {"separate_doses": False},
            {"separate_doses": True}
        ],
        "reference_cohort": "dose0_unvaccinated",
        "age_bands": [
            {"label": "65-74", "min_age": 65, "max_age": 74},
            {"label": "75-84", "min_age": 75, "max_age": 84},
            {"label": "85+", "min_age": 85, "max_age": 200}
        ],
        "min_persons_per_age_band": 100,
        "statistical": {
            "alpha": 0.05,
            "normalization_t": None,
            "bootstrap": {
                "n_samples": 1000,
                "random_seed": None
            }
        },
        "output": {
            "base_dir": "data/Czech2/kcor_mortality_output",
            "create_subdirs": True,
            "save_intermediate": True
        },
        "visualization": {
            "figure_dpi": 150,
            "figure_size": [12, 6],
            "style": "whitegrid",
            "color_palette": "tab10"
        }
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return defaults.
    
    Args:
        config_path: Path to YAML config file (if None, looks for default)
    
    Returns:
        Configuration dictionary
    """
    # Try to find default config if not specified
    if config_path is None:
        # Look for config in common locations
        possible_paths = [
            "config/kcor_mortality_config.yaml",
            "../config/kcor_mortality_config.yaml",
            os.path.join(os.path.dirname(__file__), "../config/kcor_mortality_config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # Load from file if exists
    if config_path and os.path.exists(config_path):
        if not HAS_YAML:
            print(f"Warning: pyyaml not installed, cannot load {config_path}")
            print("Using default configuration")
            return get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Merge with defaults to ensure all keys exist
            default_config = get_default_config()
            config = _merge_config(default_config, config)
            
            return config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
            return get_default_config()
    else:
        return get_default_config()


def _merge_config(default: Dict, user: Dict) -> Dict:
    """
    Recursively merge user config into default config.
    
    Args:
        default: Default configuration dictionary
        user: User configuration dictionary
    
    Returns:
        Merged configuration dictionary
    """
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    if not HAS_YAML:
        raise ImportError("pyyaml is required to save configuration files")
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_enrollment_dates(config: Dict[str, Any]) -> list:
    """
    Extract enrollment dates from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of (year, month) tuples
    """
    return [(d["year"], d["month"]) for d in config["enrollment_dates"]]


def get_quiet_periods(config: Dict[str, Any]) -> list:
    """
    Extract quiet periods from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of (min, max) tuples
    """
    return [(p["min"], p["max"]) for p in config["quiet_periods"]]


def get_age_bands(config: Dict[str, Any]) -> list:
    """
    Extract age bands from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of (label, min_age, max_age) tuples
    """
    return [(b["label"], b["min_age"], b["max_age"]) for b in config["age_bands"]]


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Loaded configuration:")
    print(f"  Enrollment dates: {get_enrollment_dates(config)}")
    print(f"  Quiet periods: {get_quiet_periods(config)}")
    print(f"  Follow-up months: {config['follow_up_months']}")
    print(f"  Age bands: {get_age_bands(config)}")

