from __future__ import annotations
import sys, importlib
from pathlib import Path
from contextlib import contextmanager

_PKG = "classification_python_codes"

def _purge_pkg():
    doomed = [k for k in list(sys.modules) if k == _PKG or k.startswith(_PKG + ".")]
    for k in doomed:
        sys.modules.pop(k, None)

@contextmanager
def _use_code_root(root: str):
    """
    Strict isolation for a code tree that contains classification_python_codes/.
    - Verifies root/ classification_python_codes/__init__.py exists
    - Purges any previously-loaded classification_python_codes* modules
    - Prepends root to sys.path for the duration
    - Purges again and restores sys.path on exit
    """
    root = str(Path(root).resolve())
    if not (Path(root) / _PKG / "__init__.py").exists():
        raise ImportError(f"Expected {_PKG} at {root}/{_PKG}/__init__.py")

    _purge_pkg()
    added = False
    if root not in sys.path:
        sys.path.insert(0, root)
        added = True
    importlib.invalidate_caches()
    try:
        yield
    finally:
        _purge_pkg()
        if added and sys.path and sys.path[0] == root:
            sys.path.pop(0)
        importlib.invalidate_caches()

# keep your public API names intact
def use_tda_code(root: str):   return _use_code_root(root)
def use_base_code(root: str):  return _use_code_root(root)
