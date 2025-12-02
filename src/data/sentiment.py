"""
Sentiment analysis using FinBERT.

Scores financial news text for sentiment (positive, negative, neutral).
"""
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
from tqdm import tqdm


class FinBERTSentiment:
    """FinBERT sentiment analyzer for financial text."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32):
        """
        Initialize FinBERT model.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for processing
        """
        print(f"Loading FinBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.batch_size = batch_size
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
        
        # FinBERT outputs: [positive, negative, neutral]
        self.label_names = ['positive', 'negative', 'neutral']
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for FinBERT processing.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove byte string prefix if present (b'...')
        if text.startswith("b'") or text.startswith('b"'):
            text = text[2:-1]
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def score_sentiment(self, text: str) -> Dict[str, float]:
        """
        Score sentiment for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and overall sentiment
        """
        text = self.clean_text(text)
        
        if not text:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment_score': 0.0,  # neutral
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # FinBERT outputs: [positive, negative, neutral]
        result = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
        }
        
        # Calculate overall sentiment score: positive - negative (range: -1 to 1)
        result['sentiment_score'] = result['positive'] - result['negative']
        
        return result
    
    def score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Score sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        results = []
        
        # Process in batches
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch_texts = cleaned_texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Process each result in batch
            for prob in probs:
                result = {
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2]),
                    'sentiment_score': float(prob[0] - prob[1]),
                }
                results.append(result)
        
        return results
    
    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Score sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame with text column
            text_column: Name of column containing text
            show_progress: Show progress bar
            
        Returns:
            DataFrame with sentiment scores added
        """
        df = df.copy()
        
        texts = df[text_column].tolist()
        
        all_results = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Scoring sentiment", unit="batch")
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self.score_batch(batch_texts)
            all_results.extend(batch_results)
        
        # Add results to dataframe
        for key in ['positive', 'negative', 'neutral', 'sentiment_score']:
            df[key] = [r[key] for r in all_results]
        
        return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = 'Date',
    ticker_col: str = 'ticker',
    sentiment_col: str = 'sentiment_score'
) -> pd.DataFrame:
    """
    Aggregate sentiment to daily level per ticker.
    
    Args:
        df: DataFrame with sentiment scores
        date_col: Date column name
        ticker_col: Ticker column name
        sentiment_col: Sentiment score column name
        
    Returns:
        DataFrame with daily aggregated features
    """
    # Convert date to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by date and ticker
    agg_df = df.groupby([date_col, ticker_col]).agg({
        sentiment_col: ['mean', 'std', 'min', 'max', 'count'],
        'positive': 'sum',
        'negative': 'sum',
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        date_col, ticker_col,
        'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max', 'news_count',
        'positive_count', 'negative_count'
    ]
    
    # Fill NaN std with 0 (when only 1 news item)
    agg_df['sentiment_std'] = agg_df['sentiment_std'].fillna(0)
    
    # Calculate positive/negative ratios
    agg_df['positive_ratio'] = agg_df['positive_count'] / agg_df['news_count']
    agg_df['negative_ratio'] = agg_df['negative_count'] / agg_df['news_count']
    
    return agg_df


if __name__ == "__main__":
    # Test FinBERT
    print("Testing FinBERT sentiment analysis\n")
    
    scorer = FinBERTSentiment(batch_size=4)
    
    test_headlines = [
        "Apple reports record quarterly earnings, stock soars",
        "Intel delays chip manufacturing, shares plunge",
        "Federal Reserve announces interest rate decision",
        "Tech sector sees mixed performance today",
        "Amazon beats revenue expectations in Q3",
    ]
    
    print("Scoring individual headlines:\n")
    for headline in test_headlines:
        result = scorer.score_sentiment(headline)
        print(f"Headline: {headline}")
        print(f"  Sentiment: {result['sentiment_score']:.3f} "
              f"(pos: {result['positive']:.3f}, "
              f"neg: {result['negative']:.3f}, "
              f"neu: {result['neutral']:.3f})\n")
    
    print("\nScoring batch:")
    results = scorer.score_batch(test_headlines)
    for headline, result in zip(test_headlines, results):
        print(f"{headline}: {result['sentiment_score']:.3f}")

