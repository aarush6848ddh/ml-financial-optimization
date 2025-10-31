"""
Essay 1: Robo-advising under Rare Disasters

SARSA-IS implementation for portfolio optimization during rare disasters.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.sarsa_is import SarsaIS
from src.data.data_loader import DataLoader
from src.utils.openbb_viz import create_openbb_essay1_visualizations


def run_sarsa_is_experiment(
    episodes: int = 1000,
    symbols: Optional[list] = None,
    start_date: str = '2007-01-01',
    end_date: str = '2021-12-31',
    visualize: bool = False,
    save_plots: bool = False
) -> Dict:
    """
    Run SARSA-IS experiment for Essay 1.
    """
    print("=" * 80)
    print("Essay 1: Robo-advising under Rare Disasters")
    print("SARSA with Importance Sampling")
    print("=" * 80)
    
    if symbols is None:
        symbols = ['SPY', 'TLT', 'GLD']
    
    loader = DataLoader()
    print(f"\nLoading data for symbols: {symbols}")
    
    try:
        market_data = loader.fetch_market_data(symbols, start_date=start_date, end_date=end_date)
        returns = loader.calculate_returns(market_data.loc[(slice(None), slice(None)), 'Close'], frequency='daily')
        
        if isinstance(returns.index, pd.MultiIndex):
            returns_wide = returns.unstack(level=0)
            returns_wide.columns = symbols
        else:
            returns_wide = returns
        
        disaster_states = loader.identify_disaster_states(returns_wide)
        print(f"Data loaded: {len(returns_wide)} periods")
        
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
    
    n_states = 10
    n_actions = len(symbols)
    
    agent = SarsaIS(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    print(f"\nTraining SARSA-IS for {episodes} episodes...")
    
    def get_state(market_return: float) -> int:
        bins = np.linspace(-0.1, 0.1, n_states)
        return min(n_states - 1, max(0, np.digitize(market_return, bins) - 1))
    
    rewards_history = []
    value_history = []
    
    for episode in range(episodes):
        period_idx = np.random.randint(30, len(returns_wide) - 1)
        market_return = returns_wide.iloc[period_idx].mean()
        state = get_state(market_return)
        action = agent.select_action(state)
        
        asset_returns = returns_wide.iloc[period_idx].values
        portfolio_return = asset_returns[action]
        
        is_disaster = disaster_states.iloc[period_idx] if hasattr(disaster_states, 'iloc') else False
        next_market_return = returns_wide.iloc[min(period_idx + 1, len(returns_wide) - 1)].mean()
        next_state = get_state(next_market_return)
        next_action = agent.select_action(next_state)
        
        agent.update(state, action, portfolio_return, next_state, next_action, is_disaster)
        
        rewards_history.append(portfolio_return)
        
        value_func = agent.get_value_function()
        if state in value_func:
            value_history.append(value_func[state])
        else:
            value_history.append(0.0)
        
        if (episode + 1) % 100 == 0:
            print(f"  Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(rewards_history[-100:]):.4f}")
    
    policy = agent.get_policy()
    value_function = agent.get_value_function()
    
    print("\nTraining complete!")
    
    portfolio_returns = []
    for i in range(30, len(returns_wide) - 1):
        market_return = returns_wide.iloc[i].mean()
        state = get_state(market_return)
        action = policy.get(state, agent.select_action(state, use_importance=False))
        asset_returns = returns_wide.iloc[i].values
        portfolio_returns.append(asset_returns[action])
    
    cumulative_returns = pd.Series(portfolio_returns).cumsum()
    quarterly_return = portfolio_returns[-1] * 90 if portfolio_returns else 0
    
    print(f"\nResults:")
    print(f"  Quarterly Return: {quarterly_return:.3%}")
    
    return {
        'agent': agent,
        'policy': policy,
        'value_function': value_function,
        'rewards_history': rewards_history,
        'value_history': value_history,
        'cumulative_returns': cumulative_returns,
        'portfolio_returns': pd.Series(portfolio_returns),
        'quarterly_return': quarterly_return,
        'disaster_states': disaster_states,
        'training_metrics': {'Rewards': rewards_history, 'Values': value_history},
        'symbols': symbols,
        'returns_data': returns_wide
    }


def visualize_essay1(results: Dict, save: bool = False):
    """Visualize Essay 1 results using OpenBB native visualizations."""
    save_dir = Path(__file__).parent.parent.parent / "visualizations" / "essay1"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    create_openbb_essay1_visualizations(results, save_dir)
    
    if save:
        print(f"\nAll OpenBB visualizations saved to: {save_dir}")


if __name__ == "__main__":
    results = run_sarsa_is_experiment(episodes=500, visualize=True, save_plots=True)
    visualize_essay1(results, save=True)
