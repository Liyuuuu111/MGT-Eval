# mgt_eval/utils/paths.py
from __future__ import annotations
from pathlib import Path
import os
from importlib import resources
from platformdirs import user_data_dir

PKG = "mgt_eval"
PKG_CALIB_SUBDIR = "calibration_results"

def user_calib_dir() -> Path:
    # 1) 环境变量覆盖
    env = os.getenv("MGTEVAL_CALIB_DIR")
    if env:
        return Path(env)
    # 2) 用户数据目录（跨平台）
    return Path(user_data_dir(PKG, None)) / PKG_CALIB_SUBDIR

def pkg_calib_dir():
    # 返回一个 Traversable（包内资源目录，read-only）
    try:
        return resources.files(PKG).joinpath(PKG_CALIB_SUBDIR)
    except Exception:
        return None

def dev_calib_dir_from(any_file_in_pkg: Path) -> Path:
    # 兼容开发树：<repo_root>/calibration_results
    return any_file_in_pkg.resolve().parents[2] / PKG_CALIB_SUBDIR
