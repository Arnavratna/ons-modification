#!/usr/bin/env python3
"""
Online Newton Step (ONS) Portfolio Optimization — Backtesting Simulation

Usage:
  pip install -r requirements.txt
  python run_backtest.py

If network access is blocked, set USE_SYNTHETIC_DATA = True in src/ons/config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ons.main import main

if __name__ == "__main__":
    main()
