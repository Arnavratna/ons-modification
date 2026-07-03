"""Backtest configuration — edit these freely."""

# Data
USE_SYNTHETIC_DATA = False  # flip to True if yfinance is blocked

TICKERS = [
    "AAPL", "MSFT", "JPM", "GS", "XOM", "CVX",
    "JNJ", "PFE", "MCD", "KO", "WMT", "PG",
    "CAT", "BA", "GE", "F",
    "GLD", "TLT", "AGG", "SHY",
]

START_DATE = "2000-01-01"
END_DATE = "2010-12-31"
INITIAL_CAPITAL = 100_000

# ONS hyper-parameters
ONS_ETA = 0.0  # learning-rate offset (0 = use theoretical)
ONS_BETA = 10.0  # regularisation for A matrix
ONS_DELTA = 1 / 8  # step-size (as in Hazan et al. 2007)
ONS_ETA_FLOOR = 0.01  # minimum step size so algorithm never freezes

# EG
EG_ETA = 0.15  # higher = more responsive to market moves

# Markowitz rolling window (trading days) — shorter = more reactive
MKZ_WINDOW = 126

# Transaction cost applied to ALL strategies during evaluation (bps one-way)
TC_BPS = 50  # 50 bps = 0.5%, realistic for 2000s equities

# MA-ONS lambda linked to TC so internal penalty matches evaluation cost
MA_ONS_LAMBDA = TC_BPS / 10_000

# Plotting — Cursor's terminal can't pop up matplotlib windows
AUTO_OPEN_PLOTS = True   # open PNGs in Preview after saving (macOS)
SHOW_PLOTS = False       # set True only when running outside Cursor with a GUI
