"""
Portfolio Optimization Environment for Reinforcement Learning.

Gym-compatible environment for portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Portfolio optimization environment for RL.
    
    State: Market features (returns, volatility, etc.)
    Action: Portfolio weights (normalized to sum to 1)
    Reward: Portfolio return or utility
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        returns: pd.DataFrame,
        initial_portfolio_value: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 30,
        reward_type: str = 'return'  # 'return' or 'sharpe' or 'utility'
    ):
        """
        Initialize portfolio environment.
        
        Args:
            returns: DataFrame of asset returns
            initial_portfolio_value: Initial portfolio value
            transaction_cost: Transaction cost as fraction
            lookback_window: Number of past periods to include in state
            reward_type: Type of reward ('return', 'sharpe', 'utility')
        """
        super().__init__()
        
        self.returns = returns.values if isinstance(returns, pd.DataFrame) else returns
        self.n_assets = self.returns.shape[1]
        self.n_periods = self.returns.shape[0]
        self.initial_portfolio_value = initial_portfolio_value
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.reward_type = reward_type
        
        # Portfolio state
        self.current_step = 0
        self.portfolio_value = initial_portfolio_value
        self.weights = np.ones(self.n_assets) / self.n_assets  # Equal weights initially
        self.cash = 0.0
        
        # Observation space: market features + portfolio weights
        # Features: returns, volatility, momentum for each asset + current weights
        obs_dim = self.n_assets * 3 + self.n_assets  # 3 features per asset + weights
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: portfolio weights (will be normalized)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Tracking
        self.history = {
            'portfolio_value': [],
            'returns': [],
            'weights': [],
            'rewards': []
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step < self.lookback_window:
            # Use available history
            start_idx = 0
            end_idx = self.current_step + 1
        else:
            start_idx = self.current_step - self.lookback_window + 1
            end_idx = self.current_step + 1
        
        recent_returns = self.returns[start_idx:end_idx]
        
        # Features: returns, volatility, momentum
        features = []
        for asset_idx in range(self.n_assets):
            asset_returns = recent_returns[:, asset_idx]
            # Recent return (mean)
            mean_return = np.mean(asset_returns)
            # Volatility (std)
            volatility = np.std(asset_returns) if len(asset_returns) > 1 else 0.0
            # Momentum (latest return)
            momentum = asset_returns[-1] if len(asset_returns) > 0 else 0.0
            
            features.extend([mean_return, volatility, momentum])
        
        # Add current portfolio weights
        features.extend(self.weights.tolist())
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, portfolio_return: float) -> float:
        """Calculate reward based on portfolio return."""
        if self.reward_type == 'return':
            return portfolio_return
        elif self.reward_type == 'sharpe':
            # Sharpe ratio (simplified)
            if len(self.history['returns']) > 1:
                mean_return = np.mean(self.history['returns'])
                std_return = np.std(self.history['returns'])
                if std_return > 0:
                    return mean_return / std_return
            return portfolio_return
        elif self.reward_type == 'utility':
            # Utility: return - risk penalty
            risk_aversion = 2.0
            risk = portfolio_return ** 2  # Simplified risk measure
            return portfolio_return - (risk_aversion / 2) * risk
        else:
            return portfolio_return
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window  # Start after we have enough history
        self.portfolio_value = self.initial_portfolio_value
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.cash = 0.0
        
        self.history = {
            'portfolio_value': [self.portfolio_value],
            'returns': [],
            'weights': [self.weights.copy()],
            'rewards': []
        }
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in environment.
        
        Args:
            action: Portfolio weights (will be normalized)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0.0, 1.0)
        action = action / (np.sum(action) + 1e-8)
        
        # Calculate transaction cost
        weight_change = np.abs(action - self.weights)
        transaction_cost_value = self.transaction_cost * np.sum(weight_change) * self.portfolio_value
        
        # Update weights
        old_weights = self.weights.copy()
        self.weights = action
        
        # Get returns for current period
        if self.current_step >= self.n_periods - 1:
            # End of data
            return self._get_observation(), 0.0, True, False, {}
        
        asset_returns = self.returns[self.current_step]
        
        # Calculate portfolio return
        portfolio_return = np.dot(self.weights, asset_returns)
        
        # Subtract transaction costs
        net_return = portfolio_return - transaction_cost_value / self.portfolio_value
        
        # Update portfolio value
        self.portfolio_value *= (1 + net_return)
        
        # Calculate reward
        reward = self._calculate_reward(net_return)
        
        # Update history
        self.history['portfolio_value'].append(self.portfolio_value)
        self.history['returns'].append(net_return)
        self.history['weights'].append(self.weights.copy())
        self.history['rewards'].append(reward)
        
        # Move to next step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.n_periods - 1
        truncated = False
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': net_return,
            'weights': self.weights,
            'transaction_cost': transaction_cost_value
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment (print current state)."""
        if mode == 'human':
            print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Weights: {self.weights}")
            print(f"Last Return: {self.history['returns'][-1] if self.history['returns'] else 0:.4f}")

