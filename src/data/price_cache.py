"""Price data caching with yfinance and Parquet storage."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf


DEFAULT_PRICE_CACHE_DIR = Path("data/curated/prices")


def _ticker_safe(ticker: str) -> str:
    """Convert ticker to filesystem-safe name."""
    return ticker.replace("/", "-").replace("^", "_")


def _cache_path(
    ticker: str,
    interval: str,
    auto_adjust: bool,
    cache_dir: Path,
) -> Path:
    """Generate cache file path for a ticker."""
    adj_part = "adj" if auto_adjust else "raw"
    safe = _ticker_safe(ticker)
    return cache_dir / adj_part / interval / f"{safe}.parquet"


def _flatten_yf_columns(df: pd.DataFrame, ticker_hint: Optional[str] = None) -> pd.DataFrame:
    """Handle yfinance MultiIndex columns by reducing to single-level OHLCV frame."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # If one level has a single unique value, drop it
    for level in range(df.columns.nlevels):
        if len(df.columns.get_level_values(level).unique()) == 1:
            try:
                return df.droplevel(level, axis=1)
            except Exception:
                pass

    # If a ticker hint is provided and exists on any level, select it
    if ticker_hint is not None:
        for level in range(df.columns.nlevels):
            if ticker_hint in df.columns.get_level_values(level):
                try:
                    return df.xs(ticker_hint, level=level, axis=1)
                except Exception:
                    continue

    # Try to detect the level that contains price fields
    expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    for level in range(df.columns.nlevels):
        values = set(map(str, df.columns.get_level_values(level)))
        if expected.issubset(values) or expected.intersection(values):
            try:
                other_levels = [l for l in range(df.columns.nlevels) if l != level]
                temp = df
                for l in other_levels:
                    uniques = temp.columns.get_level_values(l).unique()
                    if len(uniques) == 1:
                        temp = temp.droplevel(l, axis=1)
                if isinstance(temp.columns, pd.MultiIndex):
                    first_key = tuple(v[0] for v in temp.columns.levels if len(v) > 0)
                    try:
                        temp = temp.xs(first_key, axis=1, drop_level=True)
                    except Exception:
                        pass
                return temp
            except Exception:
                continue

    # Final fallback: flatten by joining with underscore
    df_flat = df.copy()
    df_flat.columns = ["_".join(map(str, c)) for c in df_flat.columns]
    return df_flat


def _normalize_price_df(df: pd.DataFrame, auto_adjust: bool, ticker_hint: Optional[str] = None) -> pd.DataFrame:
    """Normalize price DataFrame to standard format."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df = _flatten_yf_columns(df, ticker_hint=ticker_hint)
    
    # Standardize index timezone
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Standardize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df.rename(columns=rename_map, inplace=True)
    
    # If auto_adjust=True, yfinance returns adjusted OHLC without Adj Close
    if auto_adjust and "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    
    # Sort and keep only relevant columns
    available_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[sorted(available_cols)]
    
    # Drop duplicates and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    
    return df


def get_prices(
    ticker: str,
    start: str | pd.Timestamp,
    end: Optional[str | pd.Timestamp] = None,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
    cache_dir: Path = DEFAULT_PRICE_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch and cache price data from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker symbol (e.g., 'AAPL', 'MSFT')
        start: Start date for data fetch
        end: End date (defaults to today)
        interval: Data interval ('1d', '1h', '1wk', etc.)
        auto_adjust: Whether to use adjusted prices
        cache_dir: Root directory for cache storage
        force_refresh: Force download even if cached
    
    Returns:
        DataFrame with columns: open, high, low, close, adj_close, volume
        Index is UTC DateTimeIndex
    """
    cache_path = _cache_path(ticker, interval, auto_adjust, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load from cache
    df_cached: pd.DataFrame | None = None
    if cache_path.exists() and not force_refresh:
        try:
            df_cached = pd.read_parquet(cache_path)
            df_cached.index = pd.to_datetime(df_cached.index, utc=True)
        except Exception:
            df_cached = None

    # Determine if we need to download
    need_download = force_refresh or df_cached is None or df_cached.empty
    if not need_download:
        # Check if we need to extend the cache
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC")
            last = df_cached.index.max()
            if pd.isna(last) or end_ts > last:
                need_download = True

    if need_download:
        # Compute incremental start if we have cached data
        dl_start = start
        if df_cached is not None and not df_cached.empty and not force_refresh:
            last = df_cached.index.max()
            dl_start = (last + pd.Timedelta(seconds=1)).tz_localize(None)
        
        # Download from yfinance
        df_new = yf.download(
            ticker,
            start=dl_start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )
        df_new = _normalize_price_df(df_new, auto_adjust=auto_adjust, ticker_hint=ticker)

        if df_cached is None or df_cached.empty or force_refresh:
            df_final = df_new
        else:
            # Concatenate and drop duplicates
            df_final = (
                pd.concat([df_cached, df_new])
                .pipe(lambda d: d[~d.index.duplicated(keep="last")])
                .sort_index()
            )
        
        # Persist to cache
        if df_final is not None and not df_final.empty:
            df_final.to_parquet(cache_path, compression="snappy")
        return df_final

    # Return from cache with time slicing
    df_out = df_cached
    if df_out is None:
        return pd.DataFrame()
    if start is not None:
        df_out = df_out.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        df_out = df_out.loc[:pd.Timestamp(end, tz="UTC")]
    return df_out


def get_prices_bulk(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: Optional[str | pd.Timestamp] = None,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
    cache_dir: Path = DEFAULT_PRICE_CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch prices for multiple tickers with progress tracking.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        interval: Data interval
        auto_adjust: Use adjusted prices
        cache_dir: Cache directory
        force_refresh: Force refresh all
        verbose: Show progress
    
    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    results: dict[str, pd.DataFrame] = {}
    ticker_list = list(tickers)
    
    if verbose:
        try:
            from tqdm import tqdm
            ticker_iter = tqdm(ticker_list, desc="Fetching prices")
        except ImportError:
            ticker_iter = ticker_list
            print(f"Fetching prices for {len(ticker_list)} tickers...")
    else:
        ticker_iter = ticker_list
    
    for t in ticker_iter:
        try:
            results[t] = get_prices(
                t,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
        except Exception as e:
            if verbose:
                print(f"Error fetching {t}: {e}")
            results[t] = pd.DataFrame()
    
    return results

