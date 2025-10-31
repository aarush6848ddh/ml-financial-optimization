"""
Essay 2: Risk Aversion and Portfolio Optimization - Using OpenBB Visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.inverse_opt import InverseOptimization
from src.data.data_loader import DataLoader
from src.environments.portfolio_env import PortfolioEnv
from src.utils.openbb_viz import create_openbb_essay2_visualizations

try:
    from stable_baselines3 import A2C
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


def run_inverse_rl_experiment(
    train: bool = True,
    epochs: int = 100,
    symbols: Optional[list] = None,
    start_date: str = '2010-01-01',
    end_date: str = '2023-12-31',
    visualize: bool = False,
    save_plots: bool = False
) -> Dict:
    """Run Essay 2 experiment: Inverse Optimization + Deep RL."""
    print("=" * 80)
    print("Essay 2: Risk Aversion and Portfolio Optimization")
    print("Inverse Optimization + Deep RL (A2C)")
    print("=" * 80)
    
    if symbols is None:
        symbols = ['SPY', 'TLT', 'GLD', 'IWM', 'EFA']
    
    loader = DataLoader()
    print("Loading data...")
    
    try:
        market_data = loader.fetch_market_data(symbols, start_date=start_date, end_date=end_date)
        prices = market_data.loc[(slice(None), slice(None)), 'Close']
        returns = loader.calculate_returns(prices, frequency='daily')
        
        if isinstance(returns.index, pd.MultiIndex):
            returns_wide = returns.unstack(level=0)
            returns_wide.columns = symbols
        else:
            returns_wide = returns
        
        disaster_states = loader.identify_disaster_states(returns_wide)
    except Exception as e:
        print(f"Error loading data: {e}, using synthetic data...")
        np.random.seed(42)
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]
        returns_wide = pd.DataFrame(
            np.random.randn(len(dates), len(symbols)) * 0.02,
            index=dates,
            columns=symbols
        )
        disaster_states = pd.Series(np.random.rand(len(returns_wide)) < 0.05, index=returns_wide.index)
    
    print(f"Data loaded: {len(returns_wide)} periods, {len(symbols)} assets")
    
    inverse_opt = InverseOptimization(n_assets=len(symbols), initial_risk_aversion=2.0, learning_rate=0.1)
    
    print("\nEstimating risk aversion from portfolio choices...")
    
    risk_aversion_normal, risk_aversion_disaster = [], []
    lookback = 60
    
    for i in range(lookback, len(returns_wide) - 30, 10):
        historical_returns = returns_wide.iloc[i-lookback:i]
        expected_returns = historical_returns.mean().values
        covariance = historical_returns.cov().values
        
        is_disaster = disaster_states.iloc[i] if i < len(disaster_states) else False
        state = 'disaster' if is_disaster else 'normal'
        
        risk_aversion = inverse_opt.get_risk_aversion(state)
        observed_weights = inverse_opt.forward_optimization(expected_returns, covariance, risk_aversion)
        
        observed_weights += np.random.normal(0, 0.01, len(symbols))
        observed_weights = np.clip(observed_weights, 0, 1)
        observed_weights = observed_weights / observed_weights.sum()
        
        new_ra = inverse_opt.online_update(observed_weights, expected_returns, covariance, state)
        
        if state == 'normal':
            risk_aversion_normal.append(new_ra)
        else:
            risk_aversion_disaster.append(new_ra)
        
        if (i - lookback) % 100 == 0:
            print(f"  Period {i}, Risk Aversion ({state}): {new_ra:.2f}")
    
    print(f"\nRisk Aversion Estimates:")
    print(f"  Normal State: {inverse_opt.normal_state_risk_aversion:.2f}")
    print(f"  Disaster State: {inverse_opt.disaster_state_risk_aversion:.2f}")
    
    model = None
    if train and SB3_AVAILABLE:
        print("\nTraining Deep RL agent (A2C)...")
        env = PortfolioEnv(returns=returns_wide.iloc[:int(len(returns_wide) * 0.8)], reward_type='utility')
        model = A2C('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=epochs * 1000)
        print("Deep RL training complete!")
    else:
        print("\nSkipping Deep RL training (not available or disabled)")
    
    print("\nEvaluating portfolio performance...")
    test_returns = returns_wide.iloc[int(len(returns_wide) * 0.8):]
    portfolio_returns = []
    
    for i in range(lookback, len(test_returns)):
        historical = returns_wide.iloc[:int(len(returns_wide) * 0.8) + i - lookback]
        recent = historical.iloc[-lookback:]
        expected_returns = recent.mean().values
        covariance = recent.cov().values
        
        is_disaster = disaster_states.iloc[int(len(returns_wide) * 0.8) + i] if len(disaster_states) > i else False
        state = 'disaster' if is_disaster else 'normal'
        risk_aversion = inverse_opt.get_risk_aversion(state)
        optimal_weights = inverse_opt.forward_optimization(expected_returns, covariance, risk_aversion)
        asset_returns = test_returns.iloc[i].values
        portfolio_return = np.dot(optimal_weights, asset_returns)
        portfolio_returns.append(portfolio_return)
    
    return {
        'inverse_opt': inverse_opt,
        'risk_aversion_history': {'normal': risk_aversion_normal, 'disaster': risk_aversion_disaster},
        'risk_aversion': {'normal': inverse_opt.normal_state_risk_aversion, 'disaster': inverse_opt.disaster_state_risk_aversion},
        'portfolio_returns': pd.Series(portfolio_returns),
        'model': model,
        'symbols': symbols
    }


def visualize_essay2(results: Dict, save: bool = False):
    """Visualize Essay 2 using OpenBB native visualizations."""
    save_dir = Path(__file__).parent.parent.parent / "visualizations" / "essay2"
    save_dir.mkdir(parents=True, exist_ok=True)
    create_openbb_essay2_visualizations(results, save_dir)
    if save:
        print(f"\nOpenBB visualizations saved to: {save_dir}")


if __name__ == "__main__":
    results = run_inverse_rl_experiment(train=False, visualize=True, save_plots=True)
    visualize_essay2(results, save=True)
