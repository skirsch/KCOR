#!/usr/bin/env python3
"""
Environment and runtime reporting script for KCOR paper.

This script prints environment information and runtime estimates
for the computational implementation section of the paper.
"""

import sys
import platform
import subprocess
import time
from pathlib import Path

def get_python_version():
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def get_platform_info():
    """Get platform information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

def get_top_dependencies(n=15):
    """Get top N dependencies from pip freeze."""
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        deps = result.stdout.strip().split("\n")
        # Filter out comments and empty lines
        deps = [d for d in deps if d and not d.startswith("#")]
        return deps[:n]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["pip freeze not available"]

def main():
    """Print environment report."""
    print("=" * 60)
    print("KCOR Environment Report")
    print("=" * 60)
    print()
    
    print("Python version:", get_python_version())
    print()
    
    print("Platform information:")
    platform_info = get_platform_info()
    for key, value in platform_info.items():
        print(f"  {key}: {value}")
    print()
    
    print("Top dependencies (from pip freeze):")
    deps = get_top_dependencies(15)
    for i, dep in enumerate(deps, 1):
        print(f"  {i:2d}. {dep}")
    print()
    
    print("=" * 60)
    print("Note: Fill in [RUNTIME] and [CPU/MEM] after running timing tests")
    print("=" * 60)

if __name__ == "__main__":
    main()

