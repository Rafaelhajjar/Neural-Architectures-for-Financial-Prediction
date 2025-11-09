"""US stock universe builder focusing on small-to-mid cap stocks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import time

import pandas as pd
import yfinance as yf


# Russell 2000 constituents (sample - in practice, scrape from iShares IWM holdings)
# This is a representative sample, you'll want to get the full list
RUSSELL_2000_SAMPLE = [
    "HIMS", "CAVA", "EXAS", "RDFN", "SOFI", "COIN", "RBLX", "HOOD", "RIVN", "LCID",
    "UPST", "BILL", "CELH", "KVUE", "IONS", "WYNN", "MTCH", "FOUR", "BMBL", "DUOL",
    "RYAN", "TOST", "NU", "RAMP", "FRSH", "GTLB", "DDOG", "SNOW", "NET", "CFLT",
    "GTLB", "PATH", "PCOR", "PCTY", "JAMF", "SUMO", "S", "DOCN", "FSLY", "ESTC",
    "MDB", "ZS", "OKTA", "CRWD", "ZM", "DOCU", "PTON", "ROKU", "SPOT", "PINS",
]

# S&P 600 SmallCap sample
SP600_SAMPLE = [
    "ATGE", "COLL", "HWKN", "CALM", "PLAB", "OMCL", "POWI", "AVAV", "TILE", "PUMP",
    "ALKS", "SPSC", "CVLT", "EXPO", "SSD", "LCII", "YELP", "HURN", "HUBG", "KLIC",
]

# S&P 400 MidCap sample  
SP400_SAMPLE = [
    "WING", "MOD", "CADE", "PNFP", "CVLT", "SPSC", "WTFC", "SFNC", "EWBC", "FHB",
    "IESC", "FFIN", "UMBF", "ABCB", "ONB", "IBOC", "OZK", "TCBI", "BOKF", "CATY",
]


@dataclass
class UniverseBuildConfig:
    """Configuration for building stock universe."""
    include_sources: List[str]  # e.g., ['russell_2000', 'sp600', 'sp400']
    min_market_cap: float = 300e6  # $300M
    max_market_cap: float = 10e9   # $10B
    min_price: float = 5.0
    min_avg_volume: float = 1e6    # $1M average daily volume
    exchanges: List[str] = None    # ['NYSE', 'NASDAQ', 'AMEX']
    out_csv: Path = Path("data/universe/us_universe.csv")
    
    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ["NYSE", "NASDAQ", "AMEX", "NYQ", "NMS", "NGM"]


def get_ticker_info(ticker: str) -> dict:
    """
    Fetch basic info for a ticker from yfinance.
    
    Returns dict with: marketCap, exchange, quoteType, currentPrice, averageVolume
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Extract key fields
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap", 0),
            "exchange": info.get("exchange", ""),
            "quote_type": info.get("quoteType", ""),
            "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "avg_volume": info.get("averageVolume", 0),
            "avg_volume_10d": info.get("averageVolume10days", 0),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "name": info.get("longName") or info.get("shortName", ""),
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "market_cap": 0,
            "exchange": "",
            "quote_type": "",
            "price": 0,
            "avg_volume": 0,
            "avg_volume_10d": 0,
            "sector": "",
            "industry": "",
            "name": "",
            "error": str(e),
        }


def scrape_russell_2000_from_wikipedia() -> List[str]:
    """Scrape Russell 2000 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
        dfs = pd.read_html(url)
        
        # Find the table with ticker symbols
        for df in dfs:
            cols_lower = [str(c).lower() for c in df.columns]
            if "ticker" in cols_lower or "symbol" in cols_lower:
                # Get the ticker column
                ticker_col = None
                for i, col in enumerate(df.columns):
                    if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                        ticker_col = col
                        break
                
                if ticker_col is not None:
                    tickers = df[ticker_col].astype(str).str.strip()
                    tickers = tickers[tickers != "nan"]
                    return tickers.tolist()
        
        print("Warning: Could not find Russell 2000 table on Wikipedia")
        return RUSSELL_2000_SAMPLE
    except Exception as e:
        print(f"Error scraping Russell 2000: {e}")
        return RUSSELL_2000_SAMPLE


def scrape_sp600_from_wikipedia() -> List[str]:
    """Scrape S&P 600 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        dfs = pd.read_html(url)
        
        if len(dfs) > 0:
            df = dfs[0]
            # Look for Symbol or Ticker column
            for col in df.columns:
                if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                    tickers = df[col].astype(str).str.strip()
                    tickers = tickers[tickers != "nan"]
                    return tickers.tolist()
        
        print("Warning: Could not find S&P 600 table on Wikipedia")
        return SP600_SAMPLE
    except Exception as e:
        print(f"Error scraping S&P 600: {e}")
        return SP600_SAMPLE


def scrape_sp400_from_wikipedia() -> List[str]:
    """Scrape S&P 400 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        dfs = pd.read_html(url)
        
        if len(dfs) > 0:
            df = dfs[0]
            for col in df.columns:
                if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                    tickers = df[col].astype(str).str.strip()
                    tickers = tickers[tickers != "nan"]
                    return tickers.tolist()
        
        print("Warning: Could not find S&P 400 table on Wikipedia")
        return SP400_SAMPLE
    except Exception as e:
        print(f"Error scraping S&P 400: {e}")
        return SP400_SAMPLE


def filter_by_criteria(tickers: List[str], config: UniverseBuildConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Filter tickers by market cap, price, volume, and exchange criteria.
    
    Returns DataFrame with ticker info and filter results.
    """
    results = []
    
    if verbose:
        try:
            from tqdm import tqdm
            ticker_iter = tqdm(tickers, desc="Filtering tickers")
        except ImportError:
            ticker_iter = tickers
            print(f"Filtering {len(tickers)} tickers...")
    else:
        ticker_iter = tickers
    
    for ticker in ticker_iter:
        info = get_ticker_info(ticker)
        
        # Apply filters
        market_cap = info.get("market_cap", 0)
        price = info.get("price", 0)
        avg_vol = max(info.get("avg_volume", 0), info.get("avg_volume_10d", 0))
        exchange = info.get("exchange", "")
        quote_type = info.get("quote_type", "")
        
        # Calculate dollar volume
        dollar_volume = price * avg_vol if price and avg_vol else 0
        
        # Check filters
        pass_market_cap = config.min_market_cap <= market_cap <= config.max_market_cap if market_cap > 0 else False
        pass_price = price >= config.min_price if price > 0 else False
        pass_volume = dollar_volume >= config.min_avg_volume if dollar_volume > 0 else False
        pass_exchange = exchange in config.exchanges if exchange else False
        pass_quote_type = quote_type == "EQUITY"
        
        info["dollar_volume"] = dollar_volume
        info["pass_market_cap"] = pass_market_cap
        info["pass_price"] = pass_price
        info["pass_volume"] = pass_volume
        info["pass_exchange"] = pass_exchange
        info["pass_quote_type"] = pass_quote_type
        info["pass_all"] = all([pass_market_cap, pass_price, pass_volume, pass_exchange, pass_quote_type])
        
        results.append(info)
        
        # Rate limiting
        time.sleep(0.1)
    
    df = pd.DataFrame(results)
    return df


def build_us_universe(config: UniverseBuildConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Build a filtered universe of small-to-mid cap US stocks.
    
    Steps:
    1. Scrape tickers from specified sources
    2. Get info for each ticker
    3. Apply filters (market cap, price, volume, exchange)
    4. Save to CSV
    
    Returns:
        DataFrame with all tickers and their filter status
    """
    all_tickers = []
    
    # Gather tickers from sources
    if "russell_2000" in config.include_sources:
        if verbose:
            print("Fetching Russell 2000 constituents...")
        all_tickers.extend(scrape_russell_2000_from_wikipedia())
    
    if "sp600" in config.include_sources:
        if verbose:
            print("Fetching S&P 600 constituents...")
        all_tickers.extend(scrape_sp600_from_wikipedia())
    
    if "sp400" in config.include_sources:
        if verbose:
            print("Fetching S&P 400 constituents...")
        all_tickers.extend(scrape_sp400_from_wikipedia())
    
    # Deduplicate
    all_tickers = sorted(set(all_tickers))
    
    if verbose:
        print(f"\nTotal unique tickers collected: {len(all_tickers)}")
        print(f"Applying filters:")
        print(f"  Market cap: ${config.min_market_cap/1e6:.0f}M - ${config.max_market_cap/1e9:.1f}B")
        print(f"  Min price: ${config.min_price}")
        print(f"  Min avg dollar volume: ${config.min_avg_volume/1e6:.1f}M")
        print(f"  Exchanges: {config.exchanges}")
        print()
    
    # Filter tickers
    df = filter_by_criteria(all_tickers, config, verbose=verbose)
    
    # Create output directory
    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    df.to_csv(config.out_csv, index=False)
    
    # Create a filtered version with only passing tickers
    df_pass = df[df["pass_all"]].copy()
    filtered_path = config.out_csv.parent / f"{config.out_csv.stem}_filtered.csv"
    df_pass.to_csv(filtered_path, index=False)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Total tickers processed: {len(df)}")
        print(f"  Passed all filters: {len(df_pass)}")
        print(f"  Pass rate: {len(df_pass)/len(df)*100:.1f}%")
        print(f"\nFull results saved to: {config.out_csv}")
        print(f"Filtered results saved to: {filtered_path}")
        
        # Show filter breakdown
        print(f"\nFilter breakdown:")
        print(f"  Market cap: {df['pass_market_cap'].sum()}")
        print(f"  Price: {df['pass_price'].sum()}")
        print(f"  Volume: {df['pass_volume'].sum()}")
        print(f"  Exchange: {df['pass_exchange'].sum()}")
        print(f"  Quote type: {df['pass_quote_type'].sum()}")
    
    return df

