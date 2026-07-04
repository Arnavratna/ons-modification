from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_prices_yfinance() -> pd.DataFrame:
    import yfinance as yf

    print(f"Downloading data for {config.TICKERS} from {config.START_DATE} to {config.END_DATE} …")
    raw = yf.download(
        config.TICKERS,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=True,
        progress=False,
    )["Close"]
    raw = raw.dropna(how="all").ffill().dropna()
    print(f"  → {len(raw)} trading days, {raw.shape[1]} assets")
    return raw


def load_prices_synthetic() -> pd.DataFrame:
    """Generate plausible synthetic daily price series."""
    print("Generating synthetic price data …")
    np.random.seed(42)
    n_days = 1_500
    n_assets = len(config.TICKERS)

    base_mu = np.array([0.07, 0.10, 0.08, 0.06, 0.03, 0.05, 0.02, 0.07]) / 252
    base_vol = np.array([0.15, 0.20, 0.18, 0.16, 0.05, 0.14, 0.10, 0.18]) / np.sqrt(252)
    mu = np.resize(base_mu, n_assets)
    vol = np.resize(base_vol, n_assets)

    rho = 0.4
    corr = rho * np.ones((n_assets, n_assets)) + (1 - rho) * np.eye(n_assets)
    L = np.linalg.cholesky(corr)
    z = np.random.randn(n_days, n_assets)
    shocks = z @ L.T
    daily_ret = mu + shocks * vol
    prices = pd.DataFrame(
        100 * np.cumprod(1 + daily_ret, axis=0),
        columns=config.TICKERS,
        index=pd.bdate_range(start="2018-01-02", periods=n_days),
    )
    
    prices.iloc[500:540] *= np.linspace(1, 0.70, 40)[:, None]
    prices.iloc[540:600] *= np.linspace(0.70, 1.05, 60)[:, None]
    print(f"  → {len(prices)} synthetic trading days")
    return prices


def get_prices() -> pd.DataFrame:
    if config.USE_SYNTHETIC_DATA:
        return load_prices_synthetic()
    try:
        return load_prices_yfinance()
    except Exception as exc:
        print(f"yfinance failed ({exc}). Falling back to synthetic data.")
        return load_prices_synthetic()
