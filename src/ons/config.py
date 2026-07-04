# Data
USE_SYNTHETIC_DATA = False

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
ONS_ETA = 0.0
ONS_BETA = 10.0
ONS_DELTA = 1 / 8
ONS_ETA_FLOOR = 0.01

# EG
EG_ETA = 0.15

# Markowitz
MKZ_WINDOW = 126

# Transaction cost
TC_BPS = 50

# MA-ONS
MA_ONS_LAMBDA = TC_BPS / 10_000

# Plotting
AUTO_OPEN_PLOTS = True
SHOW_PLOTS = False
