"""
Ticker extraction from news headlines.

Extracts stock tickers mentioned in news text.
"""
import re
import pandas as pd
from pathlib import Path
from typing import List, Set


# Company name to ticker mapping for common stocks
COMPANY_NAME_MAP = {
    # Tech companies
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "cisco": "CSCO",
    "oracle": "ORCL",
    "ibm": "IBM",
    "servicenow": "NOW",
    "snowflake": "SNOW",
    "palantir": "PLTR",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "doordash": "DASH",
    "shopify": "SHOP",
    "square": "SQ",
    "block": "SQ",
    "paypal": "PYPL",
    "stripe": "STRIPE",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "zoom": "ZM",
    "slack": "WORK",
    "twitter": "TWTR",
    "snap": "SNAP",
    "snapchat": "SNAP",
    "pinterest": "PINS",
    "reddit": "RDDT",
    "spotify": "SPOT",
    "dropbox": "DBX",
    "box": "BOX",
    "atlassian": "TEAM",
    "mongodb": "MDB",
    "datadog": "DDOG",
    "splunk": "SPLK",
    "crowdstrike": "CRWD",
    "zscaler": "ZS",
    "okta": "OKTA",
    "twilio": "TWLO",
    "docusign": "DOCU",
    "peloton": "PTON",
    "fitbit": "FIT",
    "gopro": "GPRO",
    "lucid": "LCID",
    "rivian": "RIVN",
    "nio": "NIO",
    "micron": "MU",
    "western digital": "WDC",
    "seagate": "STX",
    "applied materials": "AMAT",
    "lam research": "LRCX",
    "kla": "KLAC",
    "synopsys": "SNPS",
    "cadence": "CDNS",
    "asml": "ASML",
    "taiwan semiconductor": "TSM",
    "tsmc": "TSM",
    "samsung": "SSNLF",
    "sony": "SONY",
    "nintendo": "NTDOY",
    "activision": "ATVI",
    "electronic arts": "EA",
    "ea": "EA",
    "take-two": "TTWO",
    "roblox": "RBLX",
    "unity": "U",
    "match": "MTCH",
    "bumble": "BMBL",
    "zillow": "Z",
    "redfin": "RDFN",
    "opendoor": "OPEN",
    "carvana": "CVNA",
    "chewy": "CHWY",
    "wayfair": "W",
    "etsy": "ETSY",
    "ebay": "EBAY",
    "mercadolibre": "MELI",
    "sea limited": "SE",
    "cloudflare": "CFLT",
    "fastly": "FSLY",
    "digitalocean": "DOCN",
    "gitlab": "GTLB",
    "duolingo": "DUOL",
    "coursera": "COUR",
    "chegg": "CHGG",
    "upstart": "UPST",
    "affirm": "AFRM",
    "sofi": "SOFI",
    "lemonade": "LMND",
    "root": "ROOT",
    "fresh": "FRSH",
    "jamf": "JAMF",
    "bill.com": "BILL",
    "bill": "BILL",
    "q2": "QTWO",
    "paylocity": "PCTY",
    "workday": "WDAY",
    "coupa": "COUP",
    "veeva": "VEEV",
    "elastic": "ESTC",
    "confluent": "CFLT",
    "hashicorp": "HCP",
    "mobileye": "MBLY",
    "arm": "ARM",
    "sentiment": "SOUN",
    "soundhound": "SOUN",
    "c3.ai": "AI",
    "palantir": "PLTR",
    "ionq": "IONQ",
    "rigetti": "RGTI",
    "d-wave": "QBTS",
    "quantum computing": "QBTS",
    "marathon digital": "MARA",
    "riot platforms": "RIOT",
    "coinbase": "COIN",
    "microstrategy": "MSTR",
    "cipher mining": "CIFR",
    "applied digital": "APLD",
}


class TickerExtractor:
    """Extract ticker mentions from news text."""
    
    def __init__(self, universe_tickers: List[str] = None):
        """
        Initialize the ticker extractor.
        
        Args:
            universe_tickers: List of tickers to look for. If None, uses all in mapping.
        """
        self.universe_tickers = set(universe_tickers) if universe_tickers else None
        
        # Build regex patterns for ticker extraction
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for matching tickers."""
        # Pattern for $TICKER format (common on Reddit/Twitter)
        self.dollar_ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        
        # Pattern for standalone tickers (must be surrounded by word boundaries)
        self.standalone_ticker_pattern = re.compile(r'\b([A-Z]{2,5})\b')
        
        # Pattern for company names (case-insensitive)
        company_names = '|'.join(re.escape(name) for name in COMPANY_NAME_MAP.keys())
        self.company_name_pattern = re.compile(
            rf'\b({company_names})\b',
            re.IGNORECASE
        )
    
    def extract_tickers(self, text: str) -> Set[str]:
        """
        Extract all ticker mentions from text.
        
        Args:
            text: News headline or text
            
        Returns:
            Set of ticker symbols found
        """
        if not text or not isinstance(text, str):
            return set()
        
        tickers = set()
        
        # Extract $TICKER format
        dollar_tickers = self.dollar_ticker_pattern.findall(text)
        tickers.update(dollar_tickers)
        
        # Extract company names and map to tickers
        text_lower = text.lower()
        for company_name, ticker in COMPANY_NAME_MAP.items():
            if company_name in text_lower:
                tickers.add(ticker)
        
        # Filter to universe if provided
        if self.universe_tickers:
            tickers = tickers & self.universe_tickers
        
        return tickers
    
    def is_general_market_news(self, text: str) -> bool:
        """
        Check if news is about general market (not stock-specific).
        
        Args:
            text: News headline
            
        Returns:
            True if general market news
        """
        if not text or not isinstance(text, str):
            return True
        
        text_lower = text.lower()
        
        # Keywords that indicate general market news
        market_keywords = [
            'market', 'stocks', 'dow', 'nasdaq', 's&p', 's&p 500',
            'wall street', 'economy', 'fed', 'federal reserve',
            'inflation', 'gdp', 'unemployment', 'jobs report',
            'interest rate', 'treasury', 'bond', 'sector',
            'bull market', 'bear market', 'rally', 'selloff',
            'tech sector', 'financial sector', 'energy sector'
        ]
        
        return any(keyword in text_lower for keyword in market_keywords)


def load_universe_tickers(universe_path: str = "data/universe/us_universe_sample_filtered.csv") -> List[str]:
    """
    Load ticker list from universe file.
    
    Args:
        universe_path: Path to universe CSV file
        
    Returns:
        List of ticker symbols
    """
    try:
        df = pd.read_csv(universe_path)
        return df['ticker'].dropna().unique().tolist()
    except Exception as e:
        print(f"Warning: Could not load universe from {universe_path}: {e}")
        return []


if __name__ == "__main__":
    # Test the ticker extractor
    extractor = TickerExtractor()
    
    test_headlines = [
        "Apple reports record iPhone sales",
        "$AAPL and $TSLA rally today",
        "NVDA crushes earnings, AMD follows",
        "Tech sector tumbles on Fed news",
        "Microsoft acquires gaming company",
        "Intel delays chip manufacturing",
        "Amazon stock soars after earnings beat",
    ]
    
    print("Testing ticker extraction:\n")
    for headline in test_headlines:
        tickers = extractor.extract_tickers(headline)
        is_general = extractor.is_general_market_news(headline)
        print(f"Headline: {headline}")
        print(f"  Tickers: {tickers if tickers else 'None'}")
        print(f"  General market: {is_general}\n")

