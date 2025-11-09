"""Panel data construction utilities."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .price_cache import get_prices_bulk


def build_adj_close_panel(
    tickers: List[str],
    start: str,
    end: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Build a wide DataFrame of adjusted close prices.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        interval: Data interval
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    prices = get_prices_bulk(tickers, start=start, end=end, interval=interval)
    
    # Extract adj_close for each ticker
    panel_dict = {}
    for ticker, df in prices.items():
        if not df.empty and "adj_close" in df.columns:
            panel_dict[ticker] = df["adj_close"]
    
    if not panel_dict:
        return pd.DataFrame()
    
    panel = pd.DataFrame(panel_dict)
    return panel


def build_volume_panel(
    tickers: List[str],
    start: str,
    end: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Build a wide DataFrame of volumes."""
    prices = get_prices_bulk(tickers, start=start, end=end, interval=interval)
    
    panel_dict = {}
    for ticker, df in prices.items():
        if not df.empty and "volume" in df.columns:
            panel_dict[ticker] = df["volume"]
    
    if not panel_dict:
        return pd.DataFrame()
    
    panel = pd.DataFrame(panel_dict)
    return panel


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Compute returns from price panel.
    
    Args:
        prices: Panel DataFrame (dates x tickers)
        periods: Number of periods for return calculation
    
    Returns:
        DataFrame of returns
    """
    return prices.pct_change(periods)


def compute_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute log returns from price panel."""
    import numpy as np
    return np.log(prices / prices.shift(periods))


def compute_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility.
    
    Args:
        returns: Returns panel
        window: Rolling window size
        annualize: Annualize volatility (multiply by sqrt(252))
    
    Returns:
        DataFrame of volatilities
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * (252 ** 0.5)
    return vol


def compute_momentum(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """
    Compute momentum (total return over lookback period).
    
    Args:
        prices: Price panel
        lookback: Lookback period in days (default: 126 ≈ 6 months)
    
    Returns:
        DataFrame of momentum values
    """
    return prices.pct_change(lookback)


def rank_cross_sectional(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    Rank values cross-sectionally (within each row/date).
    
    Args:
        df: Input DataFrame
        ascending: Rank in ascending order
    
    Returns:
        DataFrame with ranks (0 to 1)
    """
    return df.rank(axis=1, pct=True, ascending=ascending)


def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score values cross-sectionally (within each row/date).
    
    Returns:
        DataFrame with z-scores
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

