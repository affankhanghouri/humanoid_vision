"""Reuse the reviewed lane annotation vocabulary and atomic writer."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "lane_acceptance/common.py"
_spec = importlib.util.spec_from_file_location("lane_acceptance_common", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)
