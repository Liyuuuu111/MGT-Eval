import os
import sys

# Allow `python -m mgt_eval.cli` to work when cwd is the package dir itself.
# If cwd ends with ".../mgt_eval", add its parent to sys.path.
try:
    cwd = os.path.abspath(os.getcwd())
    base = os.path.basename(cwd)
    if base == "mgt_eval":
        parent = os.path.dirname(cwd)
        if parent and parent not in sys.path:
            sys.path.insert(0, parent)
except Exception:
    pass
