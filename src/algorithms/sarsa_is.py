"""
SARSA with Importance Sampling for rare disaster scenarios.

This implements the SARSA-IS algorithm from Essay 1 of the thesis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict


class SarsaIS:
    """
    SARSA with Importance Sampling for portfolio optimization under rare disasters.
    
    Based on: Liang, J. (2024) "Machine Learning in Asset Pricing and Portfolio Optimization"
    Chapter 3: Robo-advising under rare disasters
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        disaster_probability: float = 0.05
    ):
        """
        Initialize SARSA-IS algorithm.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions (portfolio allocations)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Epsilon for epsilon-greedy policy
            disaster_probability: Initial disaster probability estimate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.disaster_probability = disaster_probability
        
        # Q-table: Q(s, a) - state-action value function
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
        # Importance sampling weights
        self.importance_weights = defaultdict(lambda: 1.0)
        
        # Estimated disaster probabilities per state
        self.disaster_probs = defaultdict(lambda: disaster_probability)
        
        # Tracking
        self.rewards_history = []
        self.value_history = []
        
    def select_action(self, state: int, use_importance: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with importance sampling.
        
        Args:
            state: Current state
            use_importance: Whether to use importance sampling weights
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Get Q-values for this state
        q_values = self.Q[state]
        
        # Apply importance sampling weights if enabled
        if use_importance:
            importance = self.importance_weights[state]
            q_values = q_values * importance
        
        # Select greedy action
        return np.argmax(q_values)
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        is_disaster: bool = False
    ):
        """
        Update Q-value using SARSA with importance sampling.
        
        Args:
            state: Current state
            action: Current action
            reward: Immediate reward
            next_state: Next state
            next_action: Next action (for SARSA)
            is_disaster: Whether this transition includes a disaster
        """
        # Current Q-value
        current_q = self.Q[state][action]
        
        # Next Q-value (SARSA uses next_action, not max)
        next_q = self.Q[next_state][next_action]
        
        # Importance sampling weight
        if is_disaster:
            # Increase importance of disaster transitions
            true_prob = self.disaster_probs[state]
            proposal_prob = max(true_prob, 0.1)  # Proposal should be >= true prob
            weight = true_prob / proposal_prob if proposal_prob > 0 else 1.0
            self.importance_weights[state] = weight
            
            # Update disaster probability estimate
            alpha_d = 0.1  # Learning rate for disaster probability
            self.disaster_probs[state] = (1 - alpha_d) * self.disaster_probs[state] + alpha_d * 1.0
        else:
            weight = 1.0
        
        # SARSA update with importance sampling
        td_error = reward + self.discount_factor * next_q - current_q
        self.Q[state][action] = current_q + self.learning_rate * weight * td_error
    
    def get_policy(self) -> Dict[int, int]:
        """
        Get greedy policy from Q-table.
        
        Returns:
            Dictionary mapping states to optimal actions
        """
        policy = {}
        for state in self.Q.keys():
            policy[state] = np.argmax(self.Q[state])
        return policy
    
    def get_value_function(self) -> Dict[int, float]:
        """
        Get value function V(s) = max_a Q(s, a).
        
        Returns:
            Dictionary mapping states to values
        """
        value_func = {}
        for state in self.Q.keys():
            value_func[state] = np.max(self.Q[state])
        return value_func
    
    def estimate_disaster_probability(self, state: int, history: list) -> float:
        """
        Estimate disaster probability for a state based on history.
        
        Args:
            state: State to estimate for
            history: History of (state, reward, is_disaster) tuples
            
        Returns:
            Estimated disaster probability
        """
        state_history = [h for h in history if h[0] == state]
        if not state_history:
            return self.disaster_probability
        
        disasters = sum(1 for h in state_history if h[2])
        return disasters / len(state_history) if len(state_history) > 0 else self.disaster_probability

