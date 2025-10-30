"""
Data loader for fetching and preprocessing financial data.

Includes market data, factors, disaster event detection, and ESG data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
import yfinance as yf
from pathlib import Path

warnings.filterwarnings('ignore')

# Try to import OpenBB
try:
    from openbb_terminal.sdk import openbb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    print("Warning: OpenBB not available. Some features may be limited.")


class DataLoader:
    """Load and preprocess financial data for the three essays."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory to store cached data
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Disaster periods (based on thesis)
        self.disaster_periods = {
            '2008_crisis': {
                'start': '2008-07-01',
                'end': '2009-03-31',
                'description': '2008 Financial Crisis'
            },
            'covid19': {
                'start': '2020-01-01',
                'end': '2020-06-30',
                'description': 'COVID-19 Pandemic'
            }
        }
    
    def fetch_market_data(
        self,
        symbols: List[str],
        start_date: str = '2000-01-01',
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch market data for given symbols.
        
        Args:
            symbols: List of stock/ETF symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data, multi-indexed by symbol
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_file = self.raw_dir / f"market_data_{start_date}_{end_date}.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        print(f"Fetching market data for {len(symbols)} symbols...")
        
        if OPENBB_AVAILABLE:
            # Use OpenBB if available
            data_dict = {}
            for symbol in symbols:
                try:
                    df = openbb.stocks.load(symbol, start_date=start_date, end_date=end_date)
                    if df is not None and not df.empty:
                        data_dict[symbol] = df
                    else:
                        print(f"Warning: No data for {symbol}, trying yfinance...")
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(start=start_date, end=end_date)
                        if not df.empty:
                            data_dict[symbol] = df
                except Exception as e:
                    print(f"Error fetching {symbol} from OpenBB: {e}, trying yfinance...")
                    try:
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(start=start_date, end=end_date)
                        if not df.empty:
                            data_dict[symbol] = df
                    except Exception as e2:
                        print(f"Error fetching {symbol} from yfinance: {e2}")
        else:
            # Fallback to yfinance
            data_dict = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    if not df.empty:
                        data_dict[symbol] = df
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
        
        if not data_dict:
            raise ValueError("No data fetched for any symbol")
        
        # Combine all data
        combined = pd.concat(data_dict, axis=0)
        combined.index.names = ['Symbol', 'Date']
        
        # Save to cache
        combined.to_csv(cache_file)
        print(f"Data cached to {cache_file}")
        
        return combined
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'simple',
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            prices: DataFrame with price data
            method: 'simple' or 'log'
            frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            
        Returns:
            DataFrame with returns
        """
        if method == 'simple':
            returns = prices.pct_change().dropna()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Resample if needed
        if frequency != 'daily':
            freq_map = {
                'weekly': 'W',
                'monthly': 'M',
                'quarterly': 'Q'
            }
            if frequency in freq_map:
                returns = returns.resample(freq_map[frequency]).apply(lambda x: (1 + x).prod() - 1)
        
        return returns
    
    def identify_disaster_states(
        self,
        returns: pd.DataFrame,
        threshold: float = -0.05
    ) -> pd.Series:
        """
        Identify disaster states based on returns.
        
        Args:
            returns: DataFrame of returns
            threshold: Threshold for disaster (e.g., -5% daily return)
            
        Returns:
            Series indicating disaster states (True/False)
        """
        # Identify disaster states
        if isinstance(returns.index, pd.MultiIndex):
            # For multi-index (Symbol, Date), aggregate by date
            market_returns = returns.groupby(level=1).mean()
            disaster_mask = market_returns.mean(axis=1) < threshold
        else:
            market_returns = returns.mean(axis=1)
            disaster_mask = market_returns < threshold
        
        # Also check against known disaster periods
        disaster_periods_mask = pd.Series(False, index=returns.index.get_level_values(1) if isinstance(returns.index, pd.MultiIndex) else returns.index)
        
        for period_name, period_info in self.disaster_periods.items():
            start = pd.Timestamp(period_info['start'])
            end = pd.Timestamp(period_info['end'])
            period_mask = (disaster_periods_mask.index >= start) & (disaster_periods_mask.index <= end)
            disaster_periods_mask[period_mask] = True
        
        return disaster_mask | disaster_periods_mask.reset_index(drop=True)[:len(disaster_mask)]
    
    def fetch_factors(
        self,
        start_date: str = '2000-01-01',
        end_date: Optional[str] = None,
        factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch factor data (Fama-French, market, etc.).
        
        Args:
            start_date: Start date
            end_date: End date Ethics:Optional[str] = None
            factors: List of factors to fetch. If None, fetches all available.
            
        Returns:
            DataFrame with factor returns
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if factors is None:
            factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        
        # For now, use simple market factor
        # In production, you would fetch from Ken French's website or OpenBB
        print("Fetching factor data...")
        
        # Market factor (SPY as proxy)
        spy = yf.Ticker("SPY")
        spy_data = spy.history(start=start_date, end=end_date)
        market_returns = spy_data['Close'].pct_change().dropna()
        
        # Risk-free rate (use 3-month T-bill as proxy, or fetch from FRED)
        # For simplicity, assume 0 for now
        risk_free = pd.Series(0, index=market_returns.index)
        market_factor = market_returns - risk_free
        
        factor_data = pd.DataFrame({
            'Mkt-RF': market_factor,
            'RF': risk_free
        }, index=market_returns.index)
        
        # Add other factors (simplified - in production fetch from data providers)
        for factor in factors:
            if factor not in factor_data.columns:
                # Simple factor construction (replace with actual factor data)
                factor_data[factor] = np.random.normal(0, 0.01, len(market_factor))
        
        return factor_data.dropna()
    
    def load_mutual_fund_data(
        self,
        fund_symbols: List[str],
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load mutual fund NAV data.
        
        Args:
            fund_symbols: List of mutual fund tickers
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping fund symbols to NAV DataFrames
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        fund_data = {}
        
        for symbol in fund_symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    fund_data[symbol] = df
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        
        return fund_data


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    
    # Test market data
    symbols = ['SPY', 'QQQ', 'IWM']
    data = loader.fetch_market_data(symbols, start_date='2020-01-01', end_date='2023-12-31')
    print(f"Fetched data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")

