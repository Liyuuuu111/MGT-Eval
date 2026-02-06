import os

# Shim to allow `python -m mgt_eval.cli` when cwd == package dir.
# This module turns itself into a package by defining __path__.
__path__ = [os.path.abspath(os.path.dirname(__file__))]
