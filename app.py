"""Hugging Face Spaces entry point.

HF expects a Streamlit app at the repo root. Streamlit re-executes the entry
file top-to-bottom, so we import the real app and run it.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from app.demo import main  # noqa: E402

main()
