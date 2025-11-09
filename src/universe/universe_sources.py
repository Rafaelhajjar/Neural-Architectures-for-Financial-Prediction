"""Alternative methods to automatically fetch stock universes."""
from __future__ import annotations

import io
import time
from typing import List, Optional
from pathlib import Path

import pandas as pd
import requests


def download_ishares_holdings(etf_ticker: str, save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Download holdings from iShares ETF.
    
    iShares publishes daily holdings as CSV files. This is the most reliable method.
    
    ETF Mappings:
    - IWM: Russell 2000 (small cap, ~2000 stocks)
    - IWR: Russell Mid-Cap (~800 stocks)
    - IJR: S&P 600 SmallCap (~600 stocks)
    - IJH: S&P 400 MidCap (~400 stocks)
    - IWB: Russell 1000 (large cap, ~1000 stocks)
    - IWV: Russell 3000 (all cap, ~3000 stocks)
    
    Args:
        etf_ticker: iShares ETF ticker (e.g., 'IWM', 'IJR', 'IJH')
        save_path: Optional path to save CSV
    
    Returns:
        DataFrame with columns: Ticker, Name, Sector, Market Value, Weight, etc.
    """
    # iShares holdings URL pattern
    # Format: https://www.ishares.com/us/products/{product_id}/ishares-{name}-etf/1467271812596.ajax?fileType=csv
    
    # Map tickers to iShares product IDs
    ISHARES_URLS = {
        "IWM": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
        "IWR": "https://www.ishares.com/us/products/239504/ishares-russell-midcap-etf/1467271812596.ajax?fileType=csv&fileName=IWR_holdings&dataType=fund",
        "IJR": "https://www.ishares.com/us/products/239774/ishares-core-sp-smallcap-etf/1467271812596.ajax?fileType=csv&fileName=IJR_holdings&dataType=fund",
        "IJH": "https://www.ishares.com/us/products/239763/ishares-core-sp-midcap-etf/1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund",
        "IWB": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
        "IWV": "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
    }
    
    if etf_ticker not in ISHARES_URLS:
        raise ValueError(f"Unknown ETF: {etf_ticker}. Supported: {list(ISHARES_URLS.keys())}")
    
    url = ISHARES_URLS[etf_ticker]
    
    try:
        print(f"Downloading {etf_ticker} holdings from iShares...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # iShares CSVs have metadata rows at the top, skip them
        content = response.text
        lines = content.split('\n')
        
        # Find where the actual data starts (look for "Ticker" header)
        start_idx = 0
        for i, line in enumerate(lines):
            if 'Ticker' in line or 'ticker' in line.lower():
                start_idx = i
                break
        
        # Read from that line
        csv_data = '\n'.join(lines[start_idx:])
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Clean up
        df = df.dropna(subset=['Ticker'])
        df = df[df['Ticker'] != '-']
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        
        # Save if requested
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Saved to {save_path}")
        
        print(f"Downloaded {len(df)} holdings")
        return df
        
    except Exception as e:
        print(f"Error downloading {etf_ticker} holdings: {e}")
        return pd.DataFrame()


def get_all_nyse_tickers() -> List[str]:
    """
    Get all NYSE-listed tickers using yfinance screener.
    
    Note: This may not be 100% complete but provides a large universe.
    """
    import yfinance as yf
    
    try:
        # Use yfinance screener (requires yfinance >= 0.2.28)
        screener = yf.Screener()
        nyse_query = {
            "exchange": ["NYQ"],  # NYSE
        }
        screener.set_predefined_body(nyse_query)
        data = screener.response
        
        if data and 'quotes' in data:
            tickers = [quote['symbol'] for quote in data['quotes']]
            return tickers
        return []
    except Exception as e:
        print(f"Error getting NYSE tickers: {e}")
        return []


def get_all_nasdaq_tickers() -> List[str]:
    """Get all NASDAQ-listed tickers."""
    import yfinance as yf
    
    try:
        screener = yf.Screener()
        nasdaq_query = {
            "exchange": ["NMS", "NGM"],  # NASDAQ
        }
        screener.set_predefined_body(nasdaq_query)
        data = screener.response
        
        if data and 'quotes' in data:
            tickers = [quote['symbol'] for quote in data['quotes']]
            return tickers
        return []
    except Exception as e:
        print(f"Error getting NASDAQ tickers: {e}")
        return []


def scrape_sp500_from_wikipedia() -> List[str]:
    """
    Scrape S&P 500 from Wikipedia with proper headers.
    
    Returns list of tickers.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Add headers to avoid 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        dfs = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
        
        if len(dfs) > 0:
            df = dfs[0]
            # S&P 500 Wikipedia table has "Symbol" column
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].astype(str).str.strip().tolist()
                return [t for t in tickers if t and t != 'nan']
        
        return []
    except Exception as e:
        print(f"Error scraping S&P 500: {e}")
        return []


def build_universe_from_ishares(
    etfs: List[str],
    output_path: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build universe by downloading iShares ETF holdings.
    
    Args:
        etfs: List of ETF tickers (e.g., ['IWM', 'IJR', 'IJH'])
        output_path: Where to save the combined universe CSV
        verbose: Print progress
    
    Returns:
        DataFrame with all unique holdings
    """
    all_holdings = []
    
    for etf in etfs:
        if verbose:
            print(f"\nDownloading {etf}...")
        
        df = download_ishares_holdings(etf)
        
        if not df.empty:
            # Add source column
            df['source_etf'] = etf
            all_holdings.append(df)
        
        time.sleep(1)  # Be nice to the server
    
    if not all_holdings:
        print("No holdings downloaded!")
        return pd.DataFrame()
    
    # Combine all holdings
    combined = pd.concat(all_holdings, ignore_index=True)
    
    # Deduplicate by ticker (keep first occurrence)
    combined = combined.drop_duplicates(subset=['Ticker'], keep='first')
    
    # Clean up ticker symbols
    combined['Ticker'] = combined['Ticker'].str.replace('.', '-', regex=False)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\nCombined universe:")
        print(f"  Total unique tickers: {len(combined)}")
        print(f"  Saved to: {output_path}")
        
        if 'Sector' in combined.columns:
            print(f"\nSector breakdown:")
            print(combined['Sector'].value_counts().head(10))
    
    return combined

