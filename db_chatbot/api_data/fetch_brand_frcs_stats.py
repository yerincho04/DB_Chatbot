#!/usr/bin/env python3
"""Compatibility wrapper for brand_frcs_stats endpoint script."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).parent / "brand_frcs_stats" / "fetch_brand_frcs_stats.py"
    runpy.run_path(str(target), run_name="__main__")
