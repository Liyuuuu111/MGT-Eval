import importlib.util
import os
from pathlib import Path
from typing import Optional, Sequence

# Shim to allow `python -m mgt_eval.cli` when cwd == package dir.
# This module turns itself into a package by defining __path__.
__path__ = [os.path.abspath(os.path.dirname(__file__))]


def _load_cli_module():
    """Load sibling `cli.py` by absolute path, independent of sys.path layout."""
    here = Path(__file__).resolve().parent
    cli_path = here / "cli.py"
    if not cli_path.exists():
        raise ModuleNotFoundError(f"Cannot find CLI module at: {cli_path}")

    spec = importlib.util.spec_from_file_location("mgt_eval__cli_runtime", str(cli_path))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Failed to create module spec for: {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[Sequence[str]] = None):
    """Entrypoint used by console scripts."""
    cli_module = _load_cli_module()
    if not hasattr(cli_module, "main"):
        raise AttributeError("Loaded cli.py does not export `main`")
    return cli_module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
