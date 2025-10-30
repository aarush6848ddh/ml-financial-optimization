"""
OpenBB Terminal Professional Dark Theme Visualizations

Beautiful, professional financial charts with black backgrounds inspired by
professional trading platforms. Clean, focused charts with excellent contrast.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try to import OpenBB
try:
    from openbb_terminal.sdk import openbb
    OPENBB_AVAILABLE = True
except ImportError:
    try:
        from openbb import obb
        openbb = obb
        OPENBB_AVAILABLE = True
    except ImportError:
        OPENBB_AVAILABLE = False
        openbb = None

# Fallback to yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Professional Trading Platform Dark Theme Colors
DARK_THEME = {
    'bg_main': '#0d1117',           # Deep black-blue background
    'bg_panel': '#161b22',          # Slightly lighter panel background
    'grid': '#30363d',              # Subtle grid lines
    'text_primary': '#c9d1d9',      # Light gray text
    'text_secondary': '#8b949e',        # Medium gray text
    'price_up': '#00d4aa',          # Bright cyan-green for up
    'price_down': '#f85149',        # Red for down
    'primary': '#58a6ff',           # Bright blue (primary line)
    'accent': '#f0883e',            # Orange accent
    'warning': '#d29922',           # Yellow warning
    'success': '#3fb950',           # Green success
    'ma_fast': '#f0883e',           # Orange for MA20
    'ma_slow': '#8b5cf6'             # Purple for MA50
}


def fetch_data_openbb(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch market data using OpenBB SDK."""
    if not OPENBB_AVAILABLE:
        return None
    
    try:
        if hasattr(openbb, 'stocks') and hasattr(openbb.stocks, 'load'):
            data = openbb.stocks.load(symbol, start_date=start_date, end_date=end_date)
            return data if isinstance(data, pd.DataFrame) else None
        elif hasattr(openbb, 'equity') and hasattr(openbb.equity, 'price'):
            result = openbb.equity.price.historical(symbol, start_date=start_date, end_date=end_date)
            if hasattr(result, 'to_df'):
                return result.to_df()
            return result if isinstance(result, pd.DataFrame) else None
    except Exception:
        return None


def fetch_data_yfinance(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch market data using yfinance fallback."""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        return data if not data.empty else None
    except Exception:
        return None


def get_price_data(data: pd.DataFrame) -> pd.Series:
    """Extract price series from data."""
    if 'Close' in data.columns:
        return data['Close']
    elif 'close' in data.columns:
        return data['close']
    else:
        return data.iloc[:, 0]


def create_price_chart(symbol: str, data: pd.DataFrame, save_path: Path, title: str = "") -> bool:
    """Create professional price chart with dark theme."""
    try:
        prices = get_price_data(data)
        dates = prices.index if isinstance(prices.index, pd.DatetimeIndex) else pd.to_datetime(prices.index)
        
        # Create figure with black background
        fig = plt.figure(figsize=(16, 9), facecolor=DARK_THEME['bg_main'])
        ax = fig.add_subplot(111, facecolor=DARK_THEME['bg_main'])
        
        # Main price line - bright and clear
        ax.plot(dates, prices, linewidth=2.8, color=DARK_THEME['primary'], 
               label='Price', alpha=0.95, zorder=5)
        
        # Moving averages - distinct colors
        if len(prices) > 20:
            ma20 = prices.rolling(20).mean()
            ax.plot(dates, ma20, linewidth=2.2, color=DARK_THEME['ma_fast'], 
                   label='MA 20', alpha=0.85, linestyle='--', zorder=4)
        if len(prices) > 50:
            ma50 = prices.rolling(50).mean()
            ax.plot(dates, ma50, linewidth=2.2, color=DARK_THEME['ma_slow'], 
                   label='MA 50', alpha=0.85, linestyle=':', zorder=18)
        
        chart_title = title if title else f"{symbol} Price Chart"
        ax.set_title(chart_title, fontsize=18, fontweight='bold', 
                    color=DARK_THEME['text_primary'], pad=20)
        ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold', 
                     color=DARK_THEME['text_primary'])
        ax.set_xlabel('Date', fontsize=14, fontweight='bold', 
                     color=DARK_THEME['text_primary'])
        
        # Professional dark theme styling
        ax.tick_params(colors=DARK_THEME['text_secondary'], labelsize=11)
        ax.legend(loc='upper left', frameon=True, fancybox=False, 
                 fontsize=12, facecolor=DARK_THEME['bg_panel'], 
                 edgecolor=DARK_THEME['grid'], labelcolor=DARK_THEME['text_primary'],
                 framealpha=0.95)
        ax.grid(True, alpha=0.4, color=DARK_THEME['grid'], linestyle='-', linewidth=0.8)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME['grid'])
            spine.set_linewidth(1)
        
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color=DARK_THEME['text_secondary'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=DARK_THEME['bg_main'], edgecolor='none', pad_inches=0.3)
        plt.close()
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def create_volume_chart(symbol: str, data: pd.DataFrame, save_path: Path) -> bool:
    """Create professional volume chart with dark theme."""
    try:
        prices = get_price_data(data)
        dates = prices.index if isinstance(prices.index, pd.DatetimeIndex) else pd.to_datetime(prices.index)
        
        volume = data['Volume'] if 'Volume' in data.columns else data.get('volume', pd.Series())
        if volume.empty:
            return False
        
        fig = plt.figure(figsize=(16, 6), facecolor=DARK_THEME['bg_main'])
        ax = fig.add_subplot(111, facecolor=DARK_THEME['bg_main'])
        
        # Color bars by price direction
        colors_vol = []
        for i in range(len(prices)):
            if i == 0:
                colors_vol.append(DARK_THEME['price_up'])
            else:
                colors_vol.append(DARK_THEME['price_up'] if prices.iloc[i] >= prices.iloc[i-1] 
                                else DARK_THEME['price_down'])
        
        ax.bar(dates, volume, color=colors_vol, alpha=0.7, width=1)
        
        ax.set_title(f"{symbol} Trading Volume", fontsize=18, fontweight='bold', 
                    color=DARK_THEME['text_primary'], pad=20)
        ax.set_ylabel('Volume', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        ax.set_xlabel('Date', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        
        ax.tick_params(colors=DARK_THEME['text_secondary'], labelsize=11)
        ax.grid(True, alpha=0.4, color=DARK_THEME['grid'], axis='y', linestyle='-', linewidth=0.8)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME['grid'])
            spine.set_linewidth(1)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color=DARK_THEME['text_secondary'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=DARK_THEME['bg_main'], edgecolor='none', pad_inches=0.3)
        plt.close()
        return True
    except Exception:
        return False


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_rsi_chart(symbol: str, data: pd.DataFrame, save_path: Path) -> bool:
    """Create professional RSI chart with dark theme."""
    try:
        prices = get_price_data(data)
        dates = prices.index if isinstance(prices.index, pd.DatetimeIndex) else pd.to_datetime(prices.index)
        
        rsi = calculate_rsi(prices)
        
        fig = plt.figure(figsize=(16, 6), facecolor=DARK_THEME['bg_main'])
        ax = fig.add_subplot(111, facecolor=DARK_THEME['bg_main'])
        
        # RSI line
        ax.plot(dates, rsi, linewidth=2.5, color=DARK_THEME['primary'], label='RSI')
        ax.axhline(y=70, color=DARK_THEME['price_down'], linestyle='--', linewidth=2, 
                  alpha=0.8, label='Overbought (70)')
        ax.axhline(y=30, color=DARK_THEME['success'], linestyle='--', linewidth=2, 
                  alpha=0.8, label='Oversold (30)')
        ax.axhline(y=50, color=DARK_THEME['grid'], linestyle='-', linewidth=1, alpha=0.5)
        
        # Fill overbought/oversold zones
        ax.fill_between(dates, 70, 100, alpha=0.2, color=DARK_THEME['price_down'])
        ax.fill_between(dates, 0, 30, alpha=0.2, color=DARK_THEME['success'])
        
        ax.set_title(f"{symbol} Relative Strength Index (RSI)", fontsize=18, fontweight='bold', 
                    color=DARK_THEME['text_primary'], pad=20)
        ax.set_ylabel('RSI', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        ax.set_xlabel('Date', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        ax.set_ylim(0, 100)
        
        ax.tick_params(colors=DARK_THEME['text_secondary'], labelsize=11)
        ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
                 fontsize=12, facecolor=DARK_THEME['bg_panel'], edgecolor=DARK_THEME['grid'],
                 labelcolor=DARK_THEME['text_primary'], framealpha=0.95)
        ax.grid(True, alpha=0.4, color=DARK_THEME['grid'], linestyle='-', linewidth=0.8)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME['grid'])
            spine.set_linewidth(1)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color=DARK_THEME['text_secondary'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=DARK_THEME['bg_main'], edgecolor='none', pad_inches=0.3)
        plt.close()
        return True
    except Exception:
        return False


def create_returns_distribution(symbol: str, data: pd.DataFrame, save_path: Path) -> bool:
    """Create professional returns distribution with dark theme."""
    try:
        prices = get_price_data(data)
        returns = prices.pct_change().dropna()
        
        fig = plt.figure(figsize=(14, 8), facecolor=DARK_THEME['bg_main'])
        ax = fig.add_subplot(111, facecolor=DARK_THEME['bg_main'])
        
        n, bins, patches = ax.hist(returns, bins=70, color=DARK_THEME['primary'], 
                                   alpha=0.7, edgecolor=DARK_THEME['grid'], linewidth=0.5)
        ax.axvline(x=0, color=DARK_THEME['text_secondary'], linestyle='-', linewidth=2, alpha=0.6)
        ax.axvline(x=returns.mean(), color=DARK_THEME['success'], linestyle='--', 
                  linewidth=2.5, alpha=0.9, label=f'Mean: {returns.mean():.4f}')
        
        ax.set_title(f"{symbol} Daily Returns Distribution", fontsize=18, fontweight='bold', 
                    color=DARK_THEME['text_primary'], pad=20)
        ax.set_xlabel('Daily Return', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        
        # Stats box
        stats_text = f'Mean: {returns.mean():.4f}  |  Std: {returns.std():.4f}  |  Skew: {returns.skew():.2f}'
        ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
               color=DARK_THEME['text_primary'], ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor=DARK_THEME['bg_panel'], 
                        alpha=0.95, edgecolor=DARK_THEME['grid'], linewidth=1))
        
        ax.tick_params(colors=DARK_THEME['text_secondary'], labelsize=11)
        ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
                 fontsize=12, facecolor=DARK_THEME['bg_panel'], edgecolor=DARK_THEME['grid'],
                 labelcolor=DARK_THEME['text_primary'], framealpha=0.95)
        ax.grid(True, alpha=0.4, color=DARK_THEME['grid'], axis='y', linestyle='-', linewidth=0.8)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME['grid'])
            spine.set_linewidth(1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=DARK_THEME['bg_main'], edgecolor='none', pad_inches=0.3)
        plt.close()
        return True
    except Exception:
        return False


def create_portfolio_performance(results: pd.Series, save_path: Path, title: str = "Portfolio Performance") -> bool:
    """Create portfolio performance chart with dark theme."""
    try:
        if isinstance(results, pd.Series):
            cum_returns = (1 + results).cumprod() if (results < 1).all() else results.cumsum()
        else:
            return False
        
        dates = cum_returns.index if isinstance(cum_returns.index, pd.DatetimeIndex) else range(len(cum_returns))
        
        fig = plt.figure(figsize=(16, 9), facecolor=DARK_THEME['bg_main'])
        ax = fig.add_subplot(111, facecolor=DARK_THEME['bg_main'])
        
        ax.plot(dates, cum_returns, linewidth=3.2, color=DARK_THEME['success'], label='Cumulative Returns')
        
        ax.set_title(title, fontsize=18, fontweight='bold', color=DARK_THEME['text_primary'], pad=20)
        ax.set_ylabel('Cumulative Returns', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        ax.set_xlabel('Time', fontsize=14, fontweight='bold', color=DARK_THEME['text_primary'])
        
        ax.tick_params(colors=DARK_THEME['text_secondary'], labelsize=11)
        ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
                 fontsize=13, facecolor=DARK_THEME['bg_panel'], edgecolor=DARK_THEME['grid'],
                 labelcolor=DARK_THEME['text_primary'], framealpha=0.95)
        ax.grid(True, alpha=0.4, color=DARK_THEME['grid'], linestyle='-', linewidth=0.8)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME['grid'])
            spine.set_linewidth(1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=DARK_THEME['bg_main'], edgecolor='none', pad_inches=0.3)
        plt.close()
        return True
    except Exception:
        return False


def create_openbb_essay1_visualizations(results: Dict, save_dir: Path):
    """Create professional dark theme visualizations for Essay 1."""
    symbols = results.get('symbols', ['SPY', 'TLT', 'GLD'])
    
    print("\n📊 Creating Professional Dark Theme OpenBB Visualizations for Essay 1...")
    
    start_date = '2007-01-01'
    end_date = '2021-12-31'
    
    for symbol in symbols:
        print(f"  Processing {symbol}...")
        
        data = fetch_data_openbb(symbol, start_date, end_date)
        if data is None or data.empty:
            data = fetch_data_yfinance(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            create_price_chart(symbol, data, save_dir / f"{symbol}_price.png", f"{symbol} Price - Essay 1")
            create_volume_chart(symbol, data, save_dir / f"{symbol}_volume.png")
            create_rsi_chart(symbol, data, save_dir / f"{symbol}_rsi.png")
            create_returns_distribution(symbol, data, save_dir / f"{symbol}_returns_dist.png")
            print(f"    ✓ Created 4 professional charts for {symbol}")
    
    if 'portfolio_returns' in results:
        create_portfolio_performance(results['portfolio_returns'], 
                                   save_dir / "portfolio_performance.png",
                                   "Essay 1: SARSA-IS Portfolio Performance")
        print(f"    ✓ Created portfolio performance chart")


def create_openbb_essay2_visualizations(results: Dict, save_dir: Path):
    """Create professional dark theme visualizations for Essay 2."""
    symbols = ['SPY', 'TLT', 'GLD', 'IWM', 'EFA']
    
    print("\n📊 Creating Professional Dark Theme OpenBB Visualizations for Essay 2...")
    
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    
    for symbol in symbols:
        print(f"  Processing {symbol}...")
        
        data = fetch_data_openbb(symbol, start_date, end_date)
        if data is None or data.empty:
            data = fetch_data_yfinance(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            create_price_chart(symbol, data, save_dir / f"{symbol}_price.png", f"{symbol} Price - Essay 2")
            create_volume_chart(symbol, data, save_dir / f"{symbol}_volume.png")
            create_rsi_chart(symbol, data, save_dir / f"{symbol}_rsi.png")
            print(f"    ✓ Created 3 professional charts for {symbol}")
    
    if 'portfolio_returns' in results:
        create_portfolio_performance(results['portfolio_returns'], 
                                   save_dir / "portfolio_performance.png",
                                   "Essay 2: Inverse RL Portfolio Performance")
        print(f"    ✓ Created portfolio performance chart")


def create_openbb_essay3_visualizations(results: Dict, save_dir: Path):
    """Create professional dark theme visualizations for Essay 3."""
    symbols = ['SPY', 'QQQ', 'IWM', 'EFA']
    
    print("\n📊 Creating Professional Dark Theme OpenBB Visualizations for Essay 3...")
    
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    
    for symbol in symbols:
        print(f"  Processing {symbol}...")
        
        data = fetch_data_openbb(symbol, start_date, end_date)
        if data is None or data.empty:
            data = fetch_data_yfinance(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            create_price_chart(symbol, data, save_dir / f"{symbol}_price.png", f"{symbol} Price - Essay 3")
            create_volume_chart(symbol, data, save_dir / f"{symbol}_volume.png")
            create_rsi_chart(symbol, data, save_dir / f"{symbol}_rsi.png")
            print(f"    ✓ Created 3 professional charts for {symbol}")
