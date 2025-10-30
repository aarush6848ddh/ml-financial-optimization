"""
Mean-Variance Portfolio Optimization using CVXPY and PyPortfolioOpt.

This module implements Markowitz mean-variance optimization using multiple libraries
for comparison and demonstration.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Try importing PyPortfolioOpt
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("Warning: PyPortfolioOpt not available. Some features disabled.")


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimization using CVXPY.
    
    Implements various portfolio optimization strategies:
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Risk Parity
    - Target Return
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize optimizer.
        
        Args:
            returns: DataFrame of asset returns (columns = assets, rows = time)
            risk_free_rate: Risk-free rate (annualized)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.n_assets = len(self.mean_returns)
        
        # Annualization factors
        self.freq_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
        
        # Detect frequency
        self.frequency = self._detect_frequency()
        self.annualization_factor = self.freq_map.get(self.frequency, 252)
    
    def _detect_frequency(self) -> str:
        """Detect return frequency from data."""
        if len(self.returns) < 2:
            return 'daily'
        
        avg_days = (self.returns.index[-1] - self.returns.index[0]).days / len(self.returns)
        
        if avg_days < 2:
            return 'daily'
        elif avg_days < 8:
            return 'weekly'
        elif avg_days < 32:
            return 'monthly'
        else:
            return 'quarterly'
    
    def optimize_max_sharpe(
        self,
        short_selling: bool = False,
        constraints: Optional[List] = None
    ) -> Dict:
        """
        Optimize for maximum Sharpe ratio using CVXPY.
        
        Args:
            short_selling: Allow short selling
            constraints: Additional CVXPY constraints
            
        Returns:
            Dictionary with weights, expected return, volatility, and Sharpe ratio
        """
        # Decision variable: portfolio weights
        w = cp.Variable(self.n_assets)
        
        # Annualized portfolio return and risk
        portfolio_return = self.mean_returns.T @ w * self.annualization_factor
        portfolio_risk = cp.quad_form(w, self.cov_matrix) * self.annualization_factor
        
        # Sharpe ratio (maximize = minimize negative)
        # Negative Sharpe for minimization
        sharpe = (portfolio_return - self.risk_free_rate) / cp.sqrt(portfolio_risk)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0 if not short_selling else w >= -1,  # Long-only or allow shorts
            w <= 1 if not short_selling else w <= 1  # No leverage or leverage limit
        ]
        
        if constraints:
            constraints_list.extend(constraints)
        
        # Objective: maximize Sharpe = minimize negative Sharpe
        problem = cp.Problem(cp.Maximize(sharpe), constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights = w.value
                
                # Calculate metrics
                expected_return = np.dot(weights, self.mean_returns) * self.annualization_factor
                volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(self.annualization_factor)
                sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                return {
                    'weights': weights,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'status': problem.status
                }
            else:
                return {
                    'weights': None,
                    'status': problem.status,
                    'error': 'Problem infeasible or unbounded'
                }
        except Exception as e:
            return {
                'weights': None,
                'status': 'error',
                'error': str(e)
            }
    
    def optimize_min_variance(
        self,
        short_selling: bool = False,
        constraints: Optional[List] = None
    ) -> Dict:
        """
        Optimize for minimum variance portfolio.
        
        """
        Optimize for minimum variance portfolio.
        
        Args:
            short_selling: Allow short selling
            constraints: Additional CVXPY constraints
            
        Returns:
            Dictionary with weights and metrics
        """
        w = cp.Variable(self.n_assets)
        
        # Portfolio variance (annualized)
        portfolio_risk = cp.quad_form(w, self.cov_matrix) * self.annualization_factor
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,
            w >= 0 if not short_selling else w >= -1
        ]
        
        if constraints:
            constraints_list.extend(constraints)
        
        # Objective: minimize variance
        problem = cp.Problem(cp.Minimize(portfolio_risk), constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights = w.value
                
                expected_return = np.dot(weights, self.mean_returns) * self.annualization_factor
                volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(self.annualization_factor)
                sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                return {
                    'weights': weights,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'status': problem.status
                }
            else:
                return {
                    'weights': None,
                    'status': problem.status,
                    'error': 'Problem infeasible or unbounded'
                }
        except Exception as e:
            return {
                'weights': None,
                'status': 'error',
                'error': str(e)
            }
    
    def optimize_target_return(
        self,
        target_return: float,
        short_selling: bool = False,
        constraints: Optional[List] = None
    ) -> Dict:
        """
        Optimize for minimum variance with target return constraint.
        
        Args:
            target_return: Target annual return
            short_selling: Allow short selling
            constraints: Additional CVXPY constraints
            
        Returns:
            Dictionary with weights and metrics
        """
        w = cp.Variable(self.n_assets)
        
        # Portfolio variance
        portfolio_risk = cp.quad_form(w, self.cov_matrix) * self.annualization_factor
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,
            self.mean_returns.T @ w * self.annualization_factor >= target_return,
            w >= 0 if not short_selling else w >= -1
        ]
        
        if constraints:
            constraints_list.extend(constraints)
        
        # Objective: minimize variance subject to return constraint
        problem = cp.Problem(cp.Minimize(portfolio_risk), constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights = w.value
                
                expected_return = np.dot(weights, self.mean_returns) * self.annualization_factor
                volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(self.annualization_factor)
                sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                return {
                    'weights': weights,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'status': problem.status
                }
            else:
                return {
                    'weights': None,
                    'status': problem.status,
                    'error': 'Problem infeasible or unbounded'
                }
        except Exception as e:
            return {
                'weights': None,
                'status': 'error',
                'error': str(e)
            }
    
    def optimize_pypfopt(
        self,
        method: str = 'max_sharpe',
        short_selling: bool = False
    ) -> Dict:
        """
        Optimize using PyPortfolioOpt library for comparison.
        
        Args:
            method: 'max_sharpe' or 'min_volatility'
            short_selling: Allow short selling
            
        Returns:
            Dictionary with weights and metrics
        """
        if not PYPFOPT_AVAILABLE:
            return {'error': 'PyPortfolioOpt not available'}
        
        try:
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(self.returns)
            S = risk_models.sample_cov(self.returns)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1) if not short_selling else (-1, 1))
            
            if method == 'max_sharpe':
                weights = ef.max_sharpe()
            elif method == 'min_volatility':
                weights = ef.min_volatility()
            else:
                return {'error': f'Unknown method: {method}'}
            
            # Get performance metrics
            performance = ef.portfolio_performance(verbose=False)
            
            return {
                'weights': np.array(list(weights.values())),
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def get_efficient_frontier(
        self,
        n_points: int = 50,
        short_selling: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate efficient frontier points.
        
        Args:
            n_points: Number of points on efficient frontier
            short_selling: Allow short selling
            
        Returns:
            Tuple of (returns, volatilities, sharpe_ratios)
        """
        # Find min and max returns
        min_return = self.mean_returns.min() * self.annualization_factor
        max_return = self.mean_returns.max() * self.annualization_factor
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_returns = []
        efficient_volatilities = []
        efficient_sharpes = []
        
        for target in target_returns:
            result = self.optimize_target_return(target, short_selling=short_selling)
            
            if result['weights'] is not None:
                efficient_returns.append(result['expected_return'])
                efficient_volatilities.append(result['volatility'])
                efficient_sharpes.append(result['sharpe_ratio'])
        
        return (
            np.array(efficient_returns),
            np.array(efficient_volatilities),
            np.array(efficient_sharpes)
        )


if __name__ == "__main__":
    # Test optimizer
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.DataFrame(
        np.random.randn(252, 5) * 0.01,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    )
    
    optimizer = MeanVarianceOptimizer(returns)
    
    # Test max Sharpe
    result = optimizer.optimize_max_sharpe()
    print("Max Sharpe Portfolio:")
    print(f"Weights: {result['weights']}")
    print(f"Expected Return: {result['expected_return']:.4f}")
    print(f"Volatility: {result['volatility']:.4f}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")

