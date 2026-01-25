"""
Utility functions for AX-CPT task
"""
import time
import json
from typing import Dict, Any


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_timestamp() -> float:
    """Get high-precision timestamp using perf_counter."""
    return time.perf_counter()


def ms_to_seconds(milliseconds: float) -> float:
    """Convert milliseconds to seconds."""
    return milliseconds / 1000.0


def get_filename_timestamp() -> str:
    """Get formatted timestamp for filename."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def calculate_elapsed_ms(start_time: float, end_time: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return (end_time - start_time) * 1000.0

