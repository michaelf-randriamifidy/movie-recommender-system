from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any
import yaml

from ..utils.helpers import sanitize_entry, ensure_float_list


def log_als_run_to_yaml(
    log_path: Path,
    arg: Dict[str, Any],
    suffix: str = ""
) -> Tuple[str, Path]:
    """
    Append a new ALS run entry to a YAML log file.

    Guarantees:
        - RMSE/loss values are stored as float lists (last 10 entries only)
        - Timestamps are sanitized
        - YAML file remains clean and safe
        - 'model_path' keys are removed via sanitization

    Args:
        log_path: Path to the YAML log file.
        arg: Dictionary containing ALS run metadata (e.g., hyperparameters, metrics).
        suffix: Optional suffix to append to the YAML filename before extension.

    Returns:
        Tuple containing:
            run_id: UUID string for this ALS run.
            log_path: Path to the YAML log file (same as input, saved with updates).
    """
    entry = arg.copy()

    entry["run_id"] = str(uuid.uuid4())
    entry["timestamp"] = datetime.now()

    for key in ["train_rmse", "train_loss", "test_rmse", "test_loss"]:
        if key in entry:
            entry[key] = ensure_float_list(entry[key])[-10:]
            
    entry = sanitize_entry(entry)

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                log_data = yaml.safe_load(f) or []
            except yaml.YAMLError:
                log_data = []
    else:
        log_data = []

    log_data.append(entry)

    base, ext = os.path.splitext(str(log_path))
    path = Path(f"{base}_{suffix}{ext}" if suffix else log_path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(log_data, f, sort_keys=True, allow_unicode=True)

    return entry["run_id"], path
