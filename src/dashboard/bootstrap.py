"""Ensure project root is on sys.path so `import src...` works under Streamlit."""

import sys
from pathlib import Path

# src/dashboard/bootstrap.py → parents[2] = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)
