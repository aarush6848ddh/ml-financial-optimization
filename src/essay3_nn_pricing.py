"""
Essay 3: Nonlinear Pricing Kernels via Neural Networks - Using OpenBB Visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader as FinancialDataLoader
from src.utils.openbb_viz import create_openbb_essay3_visualizations


class PricingKernelNN(nn.Module):
    """Neural network for pricing kernel approximation."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], output_dim: int = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class PricingDataset(Dataset):
    def __init__(self, factors: np.ndarray, returns: np.ndarray):
        self.factors = torch.FloatTensor(factors)
        self.returns = torch.FloatTensor(returns)
    
    def __len__(self):
        return len(self.factors)
    
    def __getitem__(self, idx):
        return self.factors[idx], self.returns[idx]


def run_nn_pricing_experiment(
    train: bool = True,
    epochs: int = 100,
    factors: Optional[List[str]] = None,
    include_esg: bool = True,
    start_date: str = '2010-01-01',
    end_date: str = '2023-12-31',
    visualize: bool = False,
    save_plots: bool = False
) -> Dict:
    """Run Essay 3 experiment: Neural Network Pricing Kernels."""
    print("=" * 80)
    print("Essay 3: Nonlinear Pricing Kernels via Neural Networks")
    print("=" * 80)
    
    if factors is None:
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        if include_esg:
            factors.extend(['ESG', 'ENV', 'SOC', 'GOV'])
    
    loader = FinancialDataLoader()
    print("\nLoading factor data...")
    
    try:
        factor_data = loader.fetch_factors(start_date=start_date, end_date=end_date, factors=factors)
        market_data = loader.fetch_market_data(['SPY'], start_date=start_date, end_date=end_date)
        prices = market_data.loc[(slice(None), slice(None)), 'Close']
        if isinstance(prices.index, pd.MultiIndex):
            prices = prices.unstack(level=0).iloc[:, 0]
        asset_returns = loader.calculate_returns(prices)
        common_dates = factor_data.index.intersection(asset_returns.index)
        factor_data = factor_data.loc[common_dates]
        asset_returns = asset_returns.loc[common_dates]
    except Exception as e:
        print(f"Error loading data: {e}, using synthetic data...")
        np.random.seed(42)
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]
        factor_data = pd.DataFrame(
            np.random.randn(len(dates), len(factors)) * 0.01,
            index=dates,
            columns=factors
        )
        linear_component = factor_data.values @ np.random.randn(len(factors)) * 0.5
        nonlinear_component = 0.1 * (factor_data.values ** 2).sum(axis=1)
        asset_returns = pd.Series(
            linear_component + nonlinear_component + np.random.randn(len(dates)) * 0.01,
            index=dates
        )
    
    print(f"Data loaded: {len(factor_data)} periods, {len(factors)} factors")
    
    X = factor_data.values
    y = asset_returns.values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = PricingKernelNN(input_dim=len(factors), hidden_dims=[64, 32])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = PricingDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    pricing_errors = []
    
    if train:
        print("\nTraining neural network pricing kernel...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_factors, batch_returns in train_loader:
                optimizer.zero_grad()
                pricing_kernel = model(batch_factors)
                predicted_returns = pricing_kernel * batch_factors.mean(dim=1)
                loss = criterion(predicted_returns, batch_returns)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            pricing_errors.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        print("Training complete!")
    
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        pricing_kernels = model(X_test_tensor).numpy()
        test_pricing_error = np.mean((pricing_kernels - y_test) ** 2)
        print(f"Test Pricing Error: {test_pricing_error:.6f}")
    
    factor_importance = {}
    model.eval()
    X_mean = torch.FloatTensor(X_test.mean(axis=0, keepdims=True))
    X_mean.requires_grad = True
    output = model(X_mean)
    output.backward()
    gradients = X_mean.grad.numpy().flatten()
    for i, factor_name in enumerate(factors):
        factor_importance[factor_name] = abs(gradients[i])
    
    return {
        'model': model,
        'factor_data': factor_data,
        'pricing_errors': pricing_errors,
        'pricing_kernels': pricing_kernels,
        'factor_importance': factor_importance,
        'factors': factors,
        'test_pricing_error': test_pricing_error
    }


def visualize_essay3(results: Dict, save: bool = False):
    """Visualize Essay 3 using OpenBB native visualizations."""
    save_dir = Path(__file__).parent.parent.parent / "visualizations" / "essay3"
    save_dir.mkdir(parents=True, exist_ok=True)
    create_openbb_essay3_visualizations(results, save_dir)
    if save:
        print(f"\n✅ OpenBB visualizations saved to: {save_dir}")


if __name__ == "__main__":
    results = run_nn_pricing_experiment(train=True, epochs=50, visualize=True, save_plots=True)
    visualize_essay3(results, save=True)
