# mgt_eval/utils/paths.py
from __future__ import annotations
from pathlib import Path
import os
from importlib import resources
from platformdirs import user_data_dir

PKG = "mgt_eval"
PKG_CALIB_SUBDIR = "calibration_results"


def _repo_root_from(any_file: Path) -> Path:
    try:
        # src/utils/paths.py -> <repo_root>
        return any_file.resolve().parents[2]
    except Exception:
        return Path.cwd()

def user_calib_dir() -> Path:
    # 1)
    env = os.getenv("MGTEVAL_CALIB_DIR")
    if env:
        return Path(env)
    # 2) （）
    return Path(user_data_dir(PKG, None)) / PKG_CALIB_SUBDIR

def pkg_calib_dir():
    # Prefer repo-root calibration_results (dev layout)
    root = _repo_root_from(Path(__file__))
    cand = root / PKG_CALIB_SUBDIR
    if cand.exists():
        return cand
    # Fallback to packaged resources (if any)
    try:
        return resources.files(PKG).joinpath(PKG_CALIB_SUBDIR)
    except Exception:
        return None

def dev_calib_dir_from(any_file_in_pkg: Path) -> Path:
    # ：<repo_root>/calibration_results
    return _repo_root_from(any_file_in_pkg) / PKG_CALIB_SUBDIR
