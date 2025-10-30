"""
Inverse Optimization for Risk Aversion Estimation.

This implements inverse optimization from Essay 2 of the thesis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import cvxpy as cp


class InverseOptimization:
    """
    Inverse optimization to estimate investor risk aversion from portfolio choices.
    
    Based on: Liang, J. (2024) "Machine Learning in Asset Pricing and Portfolio Optimization"
    Chapter 4: Risk aversion and portfolio optimization for robo-advising
    """
    
    def __init__(
        self,
        n_assets: int,
        initial_risk_aversion: float = 2.0,
        learning_rate: float = 0.1,
        normal_state_risk_aversion: Optional[float] = None,
        disaster_state_risk_aversion: Optional[float] = None
    ):
        """
        Initialize inverse optimization.
        
        Args:
            n_assets: Number of assets in portfolio
            initial_risk_aversion: Initial risk aversion parameter
            learning_rate: Learning rate for online updates
            normal_state_risk_aversion: Initial normal state risk aversion
            disaster_state_risk_aversion: Initial disaster state risk aversion
        """
        self.n_assets = n_assets
        self.learning_rate = learning_rate
        
        # State-dependent risk aversion
        self.normal_state_risk_aversion = normal_state_risk_aversion or initial_risk_aversion
        self.disaster_state_risk_aversion = disaster_state_risk_aversion or (initial_risk_aversion * 2)
        
        # History for tracking
        self.risk_aversion_history = {
            'normal': [self.normal_state_risk_aversion],
            'disaster': [self.disaster_state_risk_aversion]
        }
    
    def forward_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        constraints: Optional[list] = None
    ) -> np.ndarray:
        """
        Forward optimization: maximize utility given risk aversion.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints (e.g., bounds)
            
        Returns:
            Optimal portfolio weights
        """
        # Mean-variance optimization: max w^T * mu - (gamma/2) * w^T * Sigma * w
        # Subject to: sum(w) = 1, w >= 0
        
        weights = cp.Variable(self.n_assets)
        
        # Objective: maximize expected return - risk penalty
        objective = cp.Maximize(
            expected_returns.T @ weights - (risk_aversion / 2) * cp.quad_form(weights, covariance_matrix)
        )
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        if constraints:
            constraints_list.extend(constraints)
        
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS)
        
        if problem.status == 'optimal':
            return weights.value
        else:
            # Fallback: equal weights
            return np.ones(self.n_assets) / self.n_assets
    
    def inverse_optimization(
        self,
        observed_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        state: str = 'normal'
    ) -> float:
        """
        Inverse optimization: estimate risk aversion from observed portfolio.
        
        Args:
            observed_weights: Observed portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            state: 'normal' or 'disaster'
            
        Returns:
            Estimated risk aversion parameter
        """
        def objective(gamma):
            """Minimize difference between observed and predicted weights."""
            predicted_weights = self.forward_optimization(expected_returns, covariance_matrix, gamma)
            if predicted_weights is None:
                return 1e10
            return np.sum((observed_weights - predicted_weights) ** 2)
        
        # Search for optimal risk aversion (bounded search)
        result = minimize(
            objective,
            x0=self.normal_state_risk_aversion if state == 'normal' else self.disaster_state_risk_aversion,
            method='L-BFGS-B',
            bounds=[(0.1, 20.0)]  # Reasonable bounds for risk aversion
        )
        
        if result.success:
            return result.x[0]
        else:
            # Return current estimate if optimization fails
            return self.normal_state_risk_aversion if state == 'normal' else self.disaster_state_risk_aversion
    
    def online_update(
        self,
        observed_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        state: str = 'normal'
    ) -> float:
        """
        Online update of risk aversion estimate.
        
        Args:
            observed_weights: Observed portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            state: 'normal' or 'disaster'
            
        Returns:
            Updated risk aversion parameter
        """
        # Get current risk aversion
        current_ra = self.normal_state_risk_aversion if state == 'normal' else self.disaster_state_risk_aversion
        
        # Estimate from inverse optimization
        estimated_ra = self.inverse_optimization(observed_weights, expected_returns, covariance_matrix, state)
        
        # Online update
        new_ra = (1 - self.learning_rate) * current_ra + self.learning_rate * estimated_ra
        
        # Update state-dependent risk aversion
        if state == 'normal':
            self.normal_state_risk_aversion = new_ra
            self.risk_aversion_history['normal'].append(new_ra)
        else:
            self.disaster_state_risk_aversion = new_ra
            self.risk_aversion_history['disaster'].append(new_ra)
        
        return new_ra
    
    def get_risk_aversion(self, state: str = 'normal') -> float:
        """
        Get current risk aversion estimate.
        
        Args:
            state: 'normal' or 'disaster'
            
        Returns:
            Risk aversion parameter
        """
        return self.normal_state_risk_aversion if state == 'normal' else self.disaster_state_risk_aversion

