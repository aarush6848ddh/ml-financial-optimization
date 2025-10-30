"""
Comprehensive visualization utilities using OpenBB, Plotly, Matplotlib, and other libraries.

This module provides rich visualizations for all three essays:
- Portfolio performance charts with PyPortfolioOpt
- Risk analysis dashboards with cvxpy
- Factor analysis plots with qlib
- Training metrics visualization
- Interactive OpenBB charts
- Quantitative finance analytics with QuantLib
- Efficient frontier with cvxpy optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union
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
        OPENBB_AVAILABLE = True
    except ImportError:
        try:
            import openbb
            OPENBB_AVAILABLE = True
        except ImportError:
            OPENBB_AVAILABLE = False
            print("Warning: OpenBB not available. Some visualizations may be limited.")

# Try to import yfinance as a lightweight fallback data source
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Try pandas-datareader (stooq fallback)
try:
    from pandas_datareader import data as pdr
    PDR_AVAILABLE = True
except Exception:
    PDR_AVAILABLE = False

# Try to import cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not available. Some optimization visualizations may be limited.")

# Try to import QuantLib
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("Warning: QuantLib not available. Some quantitative finance features may be limited.")

# Try to import PyPortfolioOpt
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.plotting import plot_efficient_frontier
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("Warning: PyPortfolioOpt not available. Some portfolio optimization visualizations may be limited.")

# Try to import qlib
try:
    import qlib
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("Warning: qlib not available. Some machine learning features may be limited.")


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class QuantVisualizations:
    """Comprehensive visualization suite for quantitative finance research."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualization suite.
        
        Args:
            save_dir: Directory to save visualizations
        """
        if save_dir is None:
            self.save_dir = Path(__file__).parent.parent.parent / "visualizations"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.essay1_dir = self.save_dir / "essay1"
        self.essay2_dir = self.save_dir / "essay2"
        self.essay3_dir = self.save_dir / "essay3"
        
        for d in [self.essay1_dir, self.essay2_dir, self.essay3_dir]:
            d.mkdir(exist_ok=True)
    
    def create_portfolio_performance_plot(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Performance",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive portfolio performance plot using Plotly.
        
        Args:
            returns: Portfolio returns series
            benchmark: Benchmark returns (optional)
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(width=2, color='#1f77b4')
        ))
        
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            fig.add_trace(go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark.values,
                mode='lines',
                name='Benchmark',
                line=dict(width=2, color='#ff7f0e', dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_openbb_market_dashboard(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create OpenBB market dashboard for a symbol.
        
        Args:
            symbol: Stock/ETF symbol
            start_date: Start date
            end_date: End date
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure if OpenBB available, else None
        """
        try:
            data = None
            # Prefer OpenBB if available
            if OPENBB_AVAILABLE:
                try:
                    data = openbb.stocks.load(symbol, start_date=start_date, end_date=end_date)
                except Exception:
                    data = None
            # Fallback to yfinance if needed
            if (data is None or getattr(data, 'empty', True)) and YFINANCE_AVAILABLE:
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
                except Exception:
                    data = None
                if (data is None or getattr(data, 'empty', True)):
                    try:
                        # Fallback to history API with a period
                        hist = yf.Ticker(symbol).history(period='3y', auto_adjust=True)
                        data = hist
                    except Exception:
                        data = None
                if isinstance(data, pd.DataFrame) and 'Adj Close' in data.columns and 'Close' not in data.columns:
                    data = data.rename(columns={'Adj Close': 'Close'})
            # Final fallback: Stooq via pandas-datareader
            if (data is None or getattr(data, 'empty', True)) and PDR_AVAILABLE:
                try:
                    # Try raw symbol first
                    stooq = pdr.DataReader(symbol, 'stooq', start=start_date, end=end_date)
                except Exception:
                    stooq = None
                if (stooq is None or stooq.empty):
                    try:
                        stooq = pdr.DataReader(f"{symbol}.US", 'stooq', start=start_date, end=end_date)
                    except Exception:
                        stooq = None
                if isinstance(stooq, pd.DataFrame) and not stooq.empty:
                    stooq = stooq.sort_index()
                    data = stooq
            
            if data is None or data.empty:
                print(f"No data available for {symbol}")
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price', 'Volume', 'Returns'),
                row_heights=[0.5, 0.2, 0.3]
            )
            
            # Price chart
            if 'Close' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name='Close Price',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            if 'Volume' in data.columns:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )
            
            # Returns chart
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                fig.add_trace(
                    go.Scatter(
                        x=returns.index,
                        y=returns.values,
                        name='Returns',
                        line=dict(color='green'),
                        fill='tozeroy'
                    ),
                    row=3, col=1
                )
            
            fig.update_layout(
                title=f'{symbol} Market Dashboard',
                height=800,
                template='plotly_white'
            )
            
            if save_path:
                png_path = str(save_path).replace('.html', '.png') if '.html' in str(save_path) else str(save_path)
                if not png_path.endswith('.png'):
                    png_path += '.png'
                try:
                    fig.write_image(png_path, width=1200, height=800, scale=2)
                except:
                    print(f"Could not save PNG: {png_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error creating OpenBB dashboard: {e}")
            return None
    
    def create_risk_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Correlation Heatmap",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create correlation heatmap using Plotly.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=800,
            template='plotly_white'
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_efficient_frontier_plot(
        self,
        returns: pd.DataFrame,
        title: str = "Efficient Frontier",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create efficient frontier visualization.
        
        Args:
            returns: Returns DataFrame
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        from scipy.optimize import minimize
        
        # Calculate mean returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Generate random portfolios for visualization
        num_portfolios = 1000
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            
            portfolio_return = np.sum(weights * mean_returns) * 252  # Annualized
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            portfolio_sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = portfolio_sharpe
        
        fig = go.Figure()
        
        # Random portfolios
        fig.add_trace(go.Scatter(
            x=results[1, :],
            y=results[0, :],
            mode='markers',
            marker=dict(
                size=5,
                color=results[2, :],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Random Portfolios',
            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Risk (Std Dev)',
            yaxis_title='Expected Return',
            template='plotly_white',
            height=600
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_training_metrics_plot(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create training metrics visualization.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=list(metrics.keys())
        )
        
        for idx, (metric_name, values) in enumerate(metrics.items(), 1):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines',
                    name=metric_name,
                    line=dict(width=2)
                ),
                row=idx, col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(metrics),
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Episode", row=len(metrics), col=1)
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_factor_exposure_plot(
        self,
        factor_loadings: pd.DataFrame,
        title: str = "Factor Exposures",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create factor exposure bar chart.
        
        Args:
            factor_loadings: DataFrame with factor loadings
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for asset in factor_loadings.index:
            fig.add_trace(go.Bar(
                name=asset,
                x=factor_loadings.columns,
                y=factor_loadings.loc[asset].values
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Factors',
            yaxis_title='Factor Loading',
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_3d_surface_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xlabel: str = "X",
        ylabel: str = "Y",
        zlabel: str = "Z",
        title: str = "3D Surface",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create 3D surface plot (useful for pricing kernels).
        
        Args:
            x: X coordinates
            y: Y coordinates
            z: Z values (surface)
            xlabel: X axis label
            ylabel: Y axis label
            zlabel: Z axis label
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel
            ),
            template='plotly_white',
            height=700
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_cvxpy_efficient_frontier(
        self,
        returns: pd.DataFrame,
        num_portfolios: int = 100,
        title: str = "Efficient Frontier (cvxpy)",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create efficient frontier using cvxpy optimization.
        
        Args:
            returns: Returns DataFrame
            num_portfolios: Number of portfolios to optimize
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        if not CVXPY_AVAILABLE:
            print("cvxpy not available. Falling back to basic efficient frontier.")
            return self.create_efficient_frontier_plot(returns, title, save_path)
        
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(mean_returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        gamma = cp.Parameter(nonneg=True)
        
        # Portfolio return and risk
        portfolio_return = mean_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        # Efficient frontier points
        frontier_returns = []
        frontier_risks = []
        
        # Generate efficient frontier
        gamma_values = np.logspace(-3, 3, num=num_portfolios)
        
        for g in gamma_values:
            gamma.value = g
            problem = cp.Problem(
                cp.Maximize(portfolio_return - gamma * portfolio_risk),
                [cp.sum(weights) == 1, weights >= 0]
            )
            problem.solve(solver=cp.ECOS)
            
            if problem.status == 'optimal':
                w = weights.value
                if w is not None:
                    ret = mean_returns.T @ w * 252  # Annualized
                    risk = np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(252)  # Annualized
                    frontier_returns.append(ret)
                    frontier_risks.append(risk)
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_risks,
            y=frontier_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(width=3, color='blue')
        ))
        
        # Individual assets
        individual_returns = mean_returns * 252
        individual_risks = np.sqrt(np.diag(cov_matrix) * 252)
        
        fig.add_trace(go.Scatter(
            x=individual_risks,
            y=individual_returns,
            mode='markers+text',
            name='Assets',
            text=returns.columns,
            textposition="top center",
            marker=dict(size=10, color='red'),
            textfont=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Risk (Annualized Std Dev)',
            yaxis_title='Expected Return (Annualized)',
            template='plotly_white',
            height=600
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_pypfopt_optimization(
        self,
        returns: pd.DataFrame,
        title: str = "PyPortfolioOpt Optimization",
        save_path: Optional[str] = None
    ) -> Tuple[Optional[go.Figure], Dict]:
        """
        Create portfolio optimization visualization using PyPortfolioOpt.
        
        Args:
            returns: Returns DataFrame
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Tuple of (Plotly figure, optimization results)
        """
        if not PYPFOPT_AVAILABLE:
            print("PyPortfolioOpt not available. Skipping.")
            return None, {}
        
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.sample_cov(returns)
            
            # Optimize for maximum Sharpe ratio
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            
            # Get performance metrics
            performance = ef.portfolio_performance(verbose=False)
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Portfolio Weights',
                    'Risk-Return Comparison',
                    'Covariance Matrix',
                    'Performance Metrics'
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "heatmap"}, {"type": "table"}]]
            )
            
            # Portfolio weights
            assets = list(cleaned_weights.keys())
            weights = list(cleaned_weights.values())
            fig.add_trace(
                go.Bar(x=assets, y=weights, name='Weight'),
                row=1, col=1
            )
            
            # Risk-return scatter
            portfolio_ret = performance[0]
            portfolio_risk = performance[1]
            portfolio_sharpe = performance[2]
            
            fig.add_trace(
                go.Scatter(
                    x=[portfolio_risk],
                    y=[portfolio_ret],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='star'),
                    name='Optimal Portfolio'
                ),
                row=1, col=2
            )
            
            # Covariance matrix heatmap
            fig.add_trace(
                go.Heatmap(
                    z=S.values,
                    x=S.columns,
                    y=S.index,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Covariance")
                ),
                row=2, col=1
            )
            
            # Performance metrics table
            metrics_table = go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[
                    ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                    [f'{portfolio_ret:.2%}', f'{portfolio_risk:.2%}', f'{portfolio_sharpe:.2f}']
                ],
                fill_color='lavender',
                align='left')
            )
            fig.add_trace(metrics_table, row=2, col=2)
            
            fig.update_layout(
                title=title,
                height=800,
                template='plotly_white'
            )
            
            results = {
                'weights': cleaned_weights,
                'expected_return': portfolio_ret,
                'volatility': portfolio_risk,
                'sharpe_ratio': portfolio_sharpe
            }
            
            if save_path:
                png_path = str(save_path).replace('.html', '.png') if '.html' in str(save_path) else str(save_path)
                if not png_path.endswith('.png'):
                    png_path += '.png'
                try:
                    fig.write_image(png_path, width=1200, height=800, scale=2)
                except:
                    print(f"Could not save PNG: {png_path}")
            
            return fig, results
            
        except Exception as e:
            print(f"Error in PyPortfolioOpt optimization: {e}")
            return None, {}
    
    def create_quantlib_analytics(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        title: str = "QuantLib Analytics",
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create quantitative finance analytics using QuantLib.
        
        Args:
            symbol: Stock/ETF symbol
            start_date: Start date
            end_date: End date
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        if not QUANTLIB_AVAILABLE:
            print("QuantLib not available. Skipping.")
            return None
        
        try:
            # Fetch data (using simple approach - you'd integrate with your data source)
            # This is a placeholder - actual implementation would use QuantLib date/time utilities
            fig = go.Figure()
            
            # Example: Show QuantLib calendar and date functionality
            today = ql.Date.todaysDate()
            calendar = ql.UnitedStates()
            
            # Get business days in date range
            business_days = []
            current = ql.Date.from_date(pd.to_datetime(start_date).date())
            end = ql.Date.from_date(pd.to_datetime(end_date).date())
            
            count = 0
            while current <= end and count < 100:
                if calendar.isBusinessDay(current):
                    business_days.append(ql.to_date(current))
                current = calendar.advance(current, ql.Period(1, ql.Days))
                count += 1
            
            fig.add_trace(go.Scatter(
                x=business_days[:100],
                y=np.random.randn(len(business_days[:100])).cumsum(),
                mode='lines',
                name='Business Days Analysis',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f"{title} - {symbol}",
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_white',
                height=500
            )
            
            if save_path:
                png_path = str(save_path).replace('.html', '.png') if '.html' in str(save_path) else str(save_path)
                if not png_path.endswith('.png'):
                    png_path += '.png'
                try:
                    fig.write_image(png_path, width=1200, height=800, scale=2)
                except:
                    print(f"Could not save PNG: {png_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error in QuantLib analytics: {e}")
            return None
    
    def create_openbb_comprehensive_chart(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> Optional[Dict[str, go.Figure]]:
        """
        Create comprehensive OpenBB charts with multiple views.
        
        Args:
            symbol: Stock/ETF symbol
            start_date: Start date
            end_date: End date
            save_path: Path to save HTML files
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        try:
            # Fetch data: prefer OpenBB, fallback to yfinance
            data = None
            if OPENBB_AVAILABLE:
                try:
                    if hasattr(openbb, 'stocks'):
                        data = openbb.stocks.load(symbol, start_date=start_date, end_date=end_date)
                    elif hasattr(openbb, 'load'):
                        data = openbb.load(symbol, start_date=start_date, end_date=end_date)
                    else:
                        try:
                            from openbb import obb
                            data = obb.equity.price.historical(symbol, start_date=start_date, end_date=end_date).to_df()
                        except Exception:
                            data = None
                except Exception:
                    data = None
            if (data is None or getattr(data, 'empty', True)) and YFINANCE_AVAILABLE:
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
                except Exception:
                    data = None
                if (data is None or getattr(data, 'empty', True)):
                    try:
                        hist = yf.Ticker(symbol).history(period='3y', auto_adjust=True)
                        data = hist
                    except Exception:
                        data = None
                if isinstance(data, pd.DataFrame) and 'Adj Close' in data.columns and 'Close' not in data.columns:
                    data = data.rename(columns={'Adj Close': 'Close'})
            if (data is None or getattr(data, 'empty', True)) and PDR_AVAILABLE:
                try:
                    stooq = pdr.DataReader(symbol, 'stooq', start=start_date, end=end_date)
                except Exception:
                    stooq = None
                if (stooq is None or stooq.empty):
                    try:
                        stooq = pdr.DataReader(f"{symbol}.US", 'stooq', start=start_date, end=end_date)
                    except Exception:
                        stooq = None
                if isinstance(stooq, pd.DataFrame) and not stooq.empty:
                    stooq = stooq.sort_index()
                    data = stooq
            
            if data is None or data.empty:
                print(f"No data available for {symbol}")
                return None
            
            # Price chart with technical indicators
            fig_price = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Volume', 'Technical Indicators', 'Returns Distribution'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price
            if 'Close' in data.columns:
                fig_price.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Moving averages
                if len(data) > 20:
                    ma20 = data['Close'].rolling(20).mean()
                    ma50 = data['Close'].rolling(50).mean()
                    fig_price.add_trace(
                        go.Scatter(x=ma20.index, y=ma20.values, name='MA20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig_price.add_trace(
                        go.Scatter(x=ma50.index, y=ma50.values, name='MA50', line=dict(color='red')),
                        row=1, col=1
                    )
                
                # Volume
                if 'Volume' in data.columns:
                    fig_price.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name='Volume',
                            marker_color='lightblue'
                        ),
                        row=1, col=1
                    )
                
                # Technical indicators - RSI
                returns = data['Close'].pct_change()
                if len(returns) > 14:
                    delta = returns.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    fig_price.add_trace(
                        go.Scatter(x=rsi.index, y=rsi.values, name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig_price.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_price.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Returns distribution
                returns_clean = returns.dropna()
                fig_price.add_trace(
                    go.Histogram(
                        x=returns_clean.values,
                        name='Returns Distribution',
                        nbinsx=50,
                        marker_color='green'
                    ),
                    row=3, col=1
                )
            
            fig_price.update_layout(
                title=f'{symbol} Comprehensive Analysis',
                height=900,
                template='plotly_white'
            )
            figures['comprehensive'] = fig_price
            
            # Save if requested
            if save_path:
                for name, fig in figures.items():
                    path = str(Path(save_path).parent / f"{symbol}_{name}.png") if Path(save_path).is_dir() else save_path
                    if not path.endswith('.png'):
                        path = path.replace('.html', '.png') if '.html' in path else path + '.png'
                    try:
                        fig.write_image(path, width=1200, height=700, scale=2)
                    except:
                        print(f"Could not save PNG: {path}")
            
            return figures
            
        except Exception as e:
            print(f"Error creating OpenBB charts: {e}")
            return None
    
    def create_qlib_factor_analysis(
        self,
        factor_data: pd.DataFrame,
        returns: pd.Series,
        title: str = "QLib Factor Analysis",
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create factor analysis visualization using qlib concepts.
        
        Args:
            factor_data: DataFrame with factor values
            returns: Return series
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        try:
            # Factor correlation with returns
            correlations = {}
            for col in factor_data.columns:
                corr = factor_data[col].corr(returns)
                correlations[col] = corr
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Factor-Return Correlations',
                    'Factor Importance',
                    'Factor Distributions',
                    'Factor Time Series'
                )
            )
            
            # Correlation bar chart
            factors = list(correlations.keys())
            corr_values = list(correlations.values())
            fig.add_trace(
                go.Bar(x=factors, y=corr_values, name='Correlation', marker_color='steelblue'),
                row=1, col=1
            )
            
            # Factor importance (absolute correlation day sorted)
            sorted_factors = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors = [f[0] for f in sorted_factors[:10]]
            top_values = [abs(f[1]) for f in sorted_factors[:10]]
            
            fig.add_trace(
                go.Bar(x=top_factors, y=top_values, name='Importance', marker_color='coral'),
                row=1, col=2
            )
            
            # Factor distributions
            for i, factor in enumerate(factor_data.columns[:5]):  # Top 5 factors
                fig.add_trace(
                    go.Histogram(
                        x=factor_data[factor].values,
                        name=factor,
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=2, col=1
                )
            
            # Factor time series
            for factor in factor_data.columns[:5]:
                fig.add_trace(
                    go.Scatter(
                        x=factor_data.index,
                        y=factor_data[factor].values,
                        name=factor,
                        mode='lines',
                        opacity=0.7
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=title,
                height=800,
                template='plotly_white'
            )
            
            if save_path:
                png_path = str(save_path).replace('.html', '.png') if '.html' in str(save_path) else str(save_path)
                if not png_path.endswith('.png'):
                    png_path += '.png'
                try:
                    fig.write_image(png_path, width=1200, height=800, scale=2)
                except:
                    print(f"Could not save PNG: {png_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error in qlib factor analysis: {e}")
            return None
    
    def create_risk_metrics_dashboard(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Risk Metrics Dashboard",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive risk metrics dashboard.
        
        Args:
            returns: Portfolio returns
            benchmark: Benchmark returns (optional)
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        # Calculate risk metrics
        metrics = {}
        
        # Portfolio metrics
        metrics['Portfolio'] = {
            'Return': returns.mean() * 252,
            'Volatility': returns.std() * np.sqrt(252),
            'Sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'Max Drawdown': ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis()
        }
        
        if benchmark is not None:
            metrics['Benchmark'] = {
                'Return': benchmark.mean() * 252,
                'Volatility': benchmark.std() * np.sqrt(252),
                'Sharpe': (benchmark.mean() / benchmark.std()) * np.sqrt(252) if benchmark.std() > 0 else 0,
                'Max Drawdown': ((1 + benchmark).cumprod() / (1 + benchmark).cumprod().cummax() - 1).min(),
                'Skewness': benchmark.skew(),
                'Kurtosis': benchmark.kurtosis()
            }
        
        # Create subplots
        num_cols = len(metrics) + 1
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Risk-Return Comparison',
                'Rolling Sharpe Ratio',
                'Drawdown',
                'Return Distribution',
                'Rolling Volatility',
                'Metrics Table'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Risk-return scatter
        for name, vals in metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=[vals['Volatility']],
                    y=[vals['Return']],
                    mode='markers+text',
                    text=[name],
                    textposition="top center",
                    name=name,
                    marker=dict(size=15)
                ),
                row=1, col=1
            )
        
        # Rolling Sharpe
        rolling_window = min(60, len(returns) // 4)
        if rolling_window > 1:
            rolling_sharpe = returns.rolling(rolling_window).mean() / returns.rolling(rolling_window).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Portfolio', line=dict(color='blue')),
                row=1, col=2
            )
            
            if benchmark is not None:
                bench_sharpe = benchmark.rolling(rolling_window).mean() / benchmark.rolling(rolling_window).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(x=bench_sharpe.index, y=bench_sharpe.values, name='Benchmark', line=dict(color='orange')),
                    row=1, col=2
                )
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy', name='Drawdown', line=dict(color='red')),
            row=2, col=1
        )
        
        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns.values, name='Portfolio', nbinsx=50, opacity=0.7),
            row=2, col=2
        )
        if benchmark is not None:
            fig.add_trace(
                go.Histogram(x=benchmark.values, name='Benchmark', nbinsx=50, opacity=0.7),
                row=2, col=2
            )
        
        # Rolling volatility
        rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name='Portfolio Vol', line=dict(color='blue')),
            row=3, col=1
        )
        
        # Metrics table
        metric_names = ['Return', 'Volatility', 'Sharpe', 'Max Drawdown', 'Skewness', 'Kurtosis']
        table_values = [[f'{metrics["Portfolio"][m]:.4f}' for m in metric_names]]
        if 'Benchmark' in metrics:
            table_values.append([f'{metrics["Benchmark"][m]:.4f}' for m in metric_names])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric'] + metric_names, fill_color='paleturquoise', align='left'),
                cells=dict(values=[['Portfolio'] + (['Benchmark'] if 'Benchmark' in metrics else [])] + 
                          [[f'{m:.2%}' if i < 3 else f'{m:.4f}' for i, m in enumerate(metrics['Portfolio'][k] for k in metric_names)]] +
                          ([[f'{m:.2%}' if i < 3 else f'{m:.4f}' for i, m in enumerate(metrics['Benchmark'][k] for k in metric_names)]] if 'Benchmark' in metrics else []),
                          fill_color='lavender', align='left')
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title=title,
            height=1200,
            template='plotly_white'
        )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig
    
    def create_pricing_kernel_visualization(
        self,
        factor_data: pd.DataFrame,
        pricing_kernel: np.ndarray,
        title: str = "Pricing Kernel Visualization",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create 3D visualization of pricing kernel.
        
        Args:
            factor_data: Factor data
            pricing_kernel: Pricing kernel values
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        if len(factor_data.columns) >= 2:
            x = factor_data.iloc[:, 0].values
            y = factor_data.iloc[:, 1].values
            
            # Reshape pricing kernel if needed
            if pricing_kernel.ndim == 1:
                z = pricing_kernel
            else:
                z = pricing_kernel.flatten()[:len(x)]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Pricing Kernel")
                )
            )])
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=factor_data.columns[0],
                    yaxis_title=factor_data.columns[1],
                    zaxis_title="Pricing Kernel"
                ),
                template='plotly_white',
                height=700
            )
        else:
            # 2D plot if only one factor
            x = factor_data.iloc[:, 0].values
            y = pricing_kernel.flatten()[:len(x)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name='Pricing Kernel',
                line=dict(width=2, color='blue')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=factor_data.columns[0],
                yaxis_title="Pricing Kernel",
                template='plotly_white',
                height=500
            )
        
        # Save as PNG image only (no HTML)
        if save_path:
            # Convert HTML path to PNG
            png_path = str(save_path).replace('.html', '.png') if save_path.endswith('.html') else save_path
            if not png_path.endswith('.png'):
                png_path += '.png'
            try:
                fig.write_image(png_path, width=1200, height=700, scale=2)
            except Exception:
                # Fallback: Use matplotlib to convert
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates
                    
                    # Extract data and create matplotlib figure
                    fig_mpl, ax = plt.subplots(figsize=(14, 8))
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            ax.plot(trace.x, trace.y, label=trace.name, linewidth=2)
                    ax.set_title(fig.layout.title.text if hasattr(fig.layout, 'title') else 'Chart')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not save image: {e}")
        
        return fig


def create_comprehensive_dashboard(
    results: Dict[str, Any],
    save: bool = True,
    save_dir: Optional[Path] = None
) -> Dict[str, go.Figure]:
    """
    Create comprehensive dashboard with all visualizations.
    
    Args:
        results: Dictionary containing results from all essays
        save: Whether to save visualizations
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of figure objects
    """
    viz = QuantVisualizations(save_dir=save_dir)
    figures = {}
    
    # Essay 1 visualizations
    if 'essay1' in results:
        essay1 = results['essay1']
        
        if 'cumulative_returns' in essay1:
            fig = viz.create_portfolio_performance_plot(
                essay1['cumulative_returns'],
                title="Essay 1: SARSA-IS Portfolio Performance"
            )
            figures['essay1_performance'] = fig
            if save:
                png_path = str(viz.essay1_dir / "portfolio_performance.png")
                try:
                    fig.write_image(png_path, width=1200, height=700, scale=2)
                except:
                    import matplotlib.pyplot as plt
                    if hasattr(essay1['cumulative_returns'], 'plot'):
                        fig_mpl, ax = plt.subplots(figsize=(12, 6))
                        essay1['cumulative_returns'].plot(ax=ax, linewidth=2)
                        ax.set_title("Essay 1: SARSA-IS Portfolio Performance")
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(png_path, dpi=300, bbox_inches='tight')
                        plt.close()
        
        if 'training_metrics' in essay1:
            fig = viz.create_training_metrics_plot(
                essay1['training_metrics'],
                title="Essay 1: Training Metrics"
            )
            figures['essay1_training'] = fig
            if save:
                png_path = str(viz.essay1_dir / "training_metrics.png")
                try:
                    fig.write_image(png_path, width=1200, height=700, scale=2)
                except:
                    import matplotlib.pyplot as plt
                    fig_mpl, ax = plt.subplots(figsize=(12, 6))
                    for metric_name, values in essay1['training_metrics'].items():
                        ax.plot(values, label=metric_name, linewidth=2)
                    ax.set_title("Essay 1: Training Metrics")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
    
    # Essay 2 visualizations
    if 'essay2' in results:
        essay2 = results['essay2']
        
        if 'cumulative_returns' in essay2:
            fig = viz.create_portfolio_performance_plot(
                essay2['cumulative_returns'],
                title="Essay 2: Inverse RL Portfolio Performance"
            )
            figures['essay2_performance'] = fig
            if save:
                png_path = str(viz.essay2_dir / "portfolio_performance.png")
                try:
                    fig.write_image(png_path, width=1200, height=700, scale=2)
                except:
                    import matplotlib.pyplot as plt
                    fig_mpl, ax = plt.subplots(figsize=(12, 6))
                    essay2['cumulative_returns'].plot(ax=ax, linewidth=2)
                    ax.set_title("Essay 2: Inverse RL Portfolio Performance")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
        
        if 'risk_aversion' in essay2:
            # Create risk aversion convergence plot
            risk_aversion_data = essay2['risk_aversion']
            fig = go.Figure()
            for key, values in risk_aversion_data.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(values))),
                        y=values if isinstance(values, (list, np.ndarray)) else [values],
                        mode='lines',
                        name=key
                    ))
                else:
                    # Single value, just show as a point
                    fig.add_trace(go.Scatter(
                        x=[0],
                        y=[values] if not isinstance(values, (list, np.ndarray)) else values,
                        mode='markers',
                        name=key
                    ))
            fig.update_layout(
                title="Essay 2: Risk Aversion Convergence",
                xaxis_title="Iteration",
                yaxis_title="Risk Aversion",
                template='plotly_white'
            )
            figures['essay2_risk_aversion'] = fig
            if save:
                png_path = str(viz.essay2_dir / "risk_aversion.png")
                try:
                    fig.write_image(png_path, width=1200, height=700, scale=2)
                except:
                    import matplotlib.pyplot as plt
                    fig_mpl, ax = plt.subplots(figsize=(12, 6))
                    for key, values in risk_aversion_data.items():
                        if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                            ax.plot(values, label=key, linewidth=2, marker='o', markersize=3)
                    ax.set_title("Essay 2: Risk Aversion Convergence")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Risk Aversion")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
    
    # Essay 3 visualizations
    if 'essay3' in results:
        essay3 = results['essay3']
        
        if 'pricing_errors' in essay3:
            fig = viz.create_training_metrics_plot(
                {'Pricing Error': essay3['pricing_errors']},
                title="Essay 3: Pricing Error Convergence"
            )
            figures['essay3_pricing_error'] = fig
            if save:
                png_path = str(viz.essay3_dir / "pricing_error.png")
                try:
                    fig.write_image(png_path, width=1200, height=700, scale=2)
                except:
                    import matplotlib.pyplot as plt
                    fig_mpl, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(essay3['pricing_errors'], linewidth=2)
                    ax.set_title("Essay 3: Pricing Error Convergence")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Pricing Error")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
        
        if 'factor_importance' in essay3:
            # Convert dict to DataFrame if needed
            factor_imp = essay3['factor_importance']
            if isinstance(factor_imp, dict):
                factor_df = pd.DataFrame.from_dict(
                    factor_imp,
                    orient='index',
                    columns=['Importance']
                ).sort_values('Importance', ascending=False).T
            else:
                factor_df = factor_imp
            
            fig = viz.create_factor_exposure_plot(
                factor_df,
                title="Essay 3: Factor Importance"
            )
            figures['essay3_factors'] = fig
            if save:
                png_path = str(viz.essay3_dir / "factor_importance.png")
                try:
                    fig.write_image(png_path, width=1200, height=700, scale=2)
                except:
                    import matplotlib.pyplot as plt
                    fig_mpl, ax = plt.subplots(figsize=(12, 8))
                    factors = list(factor_imp.keys())
                    importance_values = [abs(v) for v in factor_imp.values()]
                    sorted_data = sorted(zip(factors, importance_values), key=lambda x: x[1], reverse=True)
                    sorted_factors = [x[0] for x in sorted_data]
                    sorted_values = [x[1] for x in sorted_data]
                    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_factors)))
                    ax.barh(sorted_factors, sorted_values, color=colors)
                    ax.set_title("Essay 3: Factor Importance")
                    ax.set_xlabel("Importance")
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
    
    return figures

