import os
from pathlib import Path
from typing import Optional, Dict
from datetime import date, timedelta
import streamlit as st


st.set_page_config(page_title="ML Financial Optimization", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Main Findings", "OpenBB Market Analysis"])


def _write_intro_long_form():
    st.title("ML Financial Optimization — Full Findings")
    st.markdown(
        """
        **GitHub Repository**: [ml-financial-optimization](https://github.com/aarush6848ddh/ml-financial-optimization)
        
        This page is a single, comprehensive report of the project: why it was built, the hypotheses tested, how it was implemented,
        and what the results imply for quantitative portfolio construction and asset pricing.

        Source inspiration: replication and extension of Jiawen Liang's PhD thesis, "Machine Learning in Asset Pricing and Portfolio Optimization" (University of Glasgow, 2024).
        Link: https://theses.gla.ac.uk/84858/

        Why this project?
        - Robo‑advisors and many ML strategies underperform exactly when it matters most — during rare disasters (COVID‑19, GFC‑style episodes). Conventional training pipelines are dominated by “average regimes”, so policies under‑react in the tails.
        - Risk preferences are state‑dependent in reality; survey‑based profiling is static and often inconsistent with revealed behavior.
        - Linear pricing kernels can’t span nonlinear payoffs and are brittle in the factor‑zoo era; we need a compact, interpretable nonlinear alternative.

        What I built (three integrated essays):
        - Essay 1 — SARSA‑IS: an on‑policy RL allocator that deliberately over‑samples disaster‑like transitions with importance sampling, improving tail awareness without destabilizing updates.
        - Essay 2 — Inverse Optimization + Deep RL: recover investor risk aversion separately for calm vs. disaster states from observed allocations, then regularize a deep RL policy toward those economically consistent priors.
        - Essay 3 — Neural Pricing Kernel: a shallow neural network SDF that captures nonlinearities with controlled capacity to avoid overfitting, yielding lower quadratic pricing errors and interpretable factor importance.

        Data and tooling:
        - Market data via OpenBB/yfinance/Stooq; results organized under visualizations/essay1‑3 as PNG artifacts.
        - RL environments and optimization utilities in src/ (portfolio_env, mean‑variance, etc.).
        - Visual analytics with Plotly/Matplotlib; this page renders the exported PNGs for reliable presentation.

        How to read this page:
        - Each essay starts with objective and economic intuition, then design decisions, validation logic, and how to read the figures. Key takeaways summarize practical implications.
        - The galleries are curated but comprehensive; they mirror the pipeline outputs I used to verify the claims.
        """
    )
    # Pull contextual sections from README for full motivation and scope
    try:
        readme_path = Path(__file__).parent / "README.md"
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_text = f.read()
            with st.expander("Project context from README (Historical Context, Purpose, Essays Overview, Conclusions)", expanded=False):
                st.markdown(readme_text)
    except Exception:
        pass


def render_gallery_for_dir(dir_path: Path, title: str, captions_map: Optional[Dict[str, str]] = None):
    st.subheader(title)
    if not dir_path.exists():
        st.info(f"No figures found at {dir_path}")
        return
    images = sorted([p for p in dir_path.glob("*.png")])
    if not images:
        st.info("No PNG images found.")
        return
    cols = st.columns(2)
    for idx, img_path in enumerate(images):
        col = cols[idx % 2]
        caption = captions_map.get(img_path.name, img_path.stem.replace("_", " ").title()) if captions_map else img_path.stem.replace("_", " ").title()
        with col:
            st.image(str(img_path), caption=caption, use_column_width=True)


def page_findings():
    _write_intro_long_form()

    base = Path(__file__).parent / "visualizations"

    # Helper to read and show code with a title
    def show_code(title: str, path: Path, language: str = "python", max_lines: int = 300):
        st.markdown(f"#### {title}")
        try:
            text = path.read_text(encoding="utf-8")
            if max_lines:
                lines = text.splitlines()
                if len(lines) > max_lines:
                    text = "\n".join(lines[:max_lines]) + "\n# ... (truncated)"
            st.code(text, language=language)
        except Exception as e:
            st.info(f"Could not load code from {path}: {e}")

    st.markdown("### Essay 1 — SARSA-IS: Rare Disaster-Aware Allocation")
    st.markdown(
        """
        - Objective: Learn a policy that adapts exposure under rare disasters without overreacting in normal regimes.
        - Design:
          • SARSA (on‑policy) so targets match the data‑collection policy; fewer off‑policy instabilities.
          • Importance sampling to tilt learning toward crisis transitions; reduces variance where rewards matter most.
          • Features include drawdown and regime indicators; reward shaped to penalize extreme downside and instability.
        - Validation: Compare cumulative returns and drawdowns vs. benchmark; inspect RSI/volume to confirm regime shifts; verify training metrics converge smoothly.
        - Result: Better crisis‑time drawdown control and competitive long‑run compounding.
        """
    )
    essay1_dir = base / "essay1"
    essay1_captions = {
        "portfolio_performance.png": "Cumulative Returns — Policy vs Benchmark",
        "training_metrics.png": "Training Metrics — Stability and Convergence",
        "SPY_price.png": "SPY Price",
        "SPY_rsi.png": "SPY RSI",
        "SPY_volume.png": "SPY Volume",
        "GLD_price.png": "GLD Price",
        "GLD_rsi.png": "GLD RSI",
        "GLD_volume.png": "GLD Volume",
        "TLT_price.png": "TLT Price",
        "TLT_rsi.png": "TLT RSI",
        "TLT_volume.png": "TLT Volume",
        "*_returns_dist.png": "Returns Distribution",
    }
    render_gallery_for_dir(essay1_dir, "Essay 1 Figures", essay1_captions)

    # Essay 1 code walkthrough
    st.markdown("#### Essay 1 — Code Walkthrough")
    st.markdown(
        """
        Key components:
        - `src/algorithms/sarsa_is.py`: SARSA update with importance sampling; maintains on‑policy targets while reweighting tails.
        - `src/environments/portfolio_env.py`: episodic portfolio environment with returns, transaction frictions (if enabled), and risk‑aware features.
        - `src/essay1_sarsa_is.py`: training loop, hyperparameters, logging, and figure export.
        """
    )
    show_code("SARSA‑IS (core algorithm)", Path(__file__).parent / "src" / "algorithms" / "sarsa_is.py")
    show_code("Essay 1 driver", Path(__file__).parent / "src" / "essay1_sarsa_is.py")

    st.markdown("### Essay 2 — Inverse Optimization + Deep RL: State-Dependent Risk Aversion")
    st.markdown(
        """
        - Objective: Infer state‑dependent risk aversion from revealed preferences and inject it into the RL loss.
        - Design:
          • Solve inverse mean‑variance problems to estimate risk aversion for calm vs. disaster states from observed allocations.
          • Regularize the actor‑critic objective to keep actions close to those implied by the estimated preferences.
        - Read the figures: risk‑aversion trajectories should converge; performance curves should stabilize with lower variance.
        - Result: More explainable behavior and clearer separation across aggressive/moderate/conservative profiles.
        """
    )
    essay2_dir = base / "essay2"
    essay2_captions = {
        "portfolio_performance.png": "Cumulative Returns — Inverse-RL Portfolio",
        "risk_aversion.png": "Risk Aversion Convergence",
    }
    render_gallery_for_dir(essay2_dir, "Essay 2 Figures", essay2_captions)

    # Essay 2 code walkthrough
    st.markdown("#### Essay 2 — Code Walkthrough")
    st.markdown(
        """
        Key components:
        - `src/algorithms/inverse_opt.py`: solves inverse mean‑variance problems to infer state‑dependent risk aversion.
        - `src/essay2_inverse_rl.py`: integrates the inferred preferences into the actor‑critic objective as a regularizer.
        - Validation relies on the convergence of the recovered parameters and stabilized out‑of‑sample curves.
        """
    )
    show_code("Inverse optimization (risk aversion)", Path(__file__).parent / "src" / "algorithms" / "inverse_opt.py")
    show_code("Essay 2 driver", Path(__file__).parent / "src" / "essay2_inverse_rl.py")

    st.markdown("### Essay 3 — Neural Pricing Kernel: Nonlinear Cross-Sectional Pricing")
    st.markdown(
        """
        - Objective: Learn a nonlinear SDF capable of pricing option‑like payoffs without an unwieldy factor expansion.
        - Design:
          • Shallow NN with explicit regularization and early stopping; compact to preserve interpretability.
          • Evaluate via quadratic pricing error and factor importance ranking.
        - Result: Lower pricing errors and plausible drivers (including ESG proxies) without overfitting.
        """
    )
    essay3_dir = base / "essay3"
    essay3_captions = {
        "pricing_error.png": "Pricing Error Convergence",
        "factor_importance.png": "Factor Importance Ranking",
    }
    render_gallery_for_dir(essay3_dir, "Essay 3 Figures", essay3_captions)

    # Essay 3 code walkthrough
    st.markdown("#### Essay 3 — Code Walkthrough")
    st.markdown(
        """
        Key components:
        - `src/essay3_nn_pricing.py`: defines the shallow neural network SDF, loss construction, training loop, and exports.
        - Factor importance is computed from trained weights/sensitivities; pricing error curves validate convergence.
        """
    )
    show_code("Neural pricing kernel (driver)", Path(__file__).parent / "src" / "essay3_nn_pricing.py")

    # Essay 3 schematic image
    st.markdown("#### Essay 3 — Neural Network SDF Schematic")
    try:
        nn_img = Path(__file__).parent / "essay3_nn.png"
        if nn_img.exists():
            st.image(str(nn_img), caption="Neural Network SDF Schematic", use_column_width=True)
        else:
            st.info("Schematic image 'essay3_nn.png' not found in project root.")
    except Exception as e:
        st.info(f"Could not load schematic image: {e}")


def page_openbb_analysis():
    st.title("OpenBB Comprehensive Market Workbench")
    st.markdown("---")
    
    st.markdown(
        """
        ## Overview
        
        This section presents a comprehensive market analysis workbench built using the **OpenBB Platform**, 
        an open-source financial data infrastructure. The notebook explores six major ETFs across different 
        asset classes and market segments, providing deep insights into price action, technical indicators, 
        and risk metrics from EU 2015 to present.
        
        ### Purpose
        The OpenBB workbench serves as a foundational data exploration tool for the ML financial optimization 
        project. It provides the market data context, technical indicators, and risk metrics that inform the 
        three essays on portfolio optimization and asset pricing.
        
        ### Assets Analyzed
        The analysis covers six key ETFs representing different market exposures:
        - **SPY** (S&P 500 ETF) — Large-cap U.S. equities
        - **QQQ** (Nasdaq 100 ETF) — Large-cap U.S. tech stocks
        - **IWM** (Russell 2000 ETF) — Small-cap U.S. equities
        - **EFA** (iShares MSCI EAFE ETF) — International developed markets (ex-North America)
        - **GLD** (SPDR Gold Trust) — Gold commodity exposure
        - **TLT** (iShares 20+ Year Treasury Bond ETF) — Long-duration U.S. government bonds
        
        ### Methodology
        1. **Data Acquisition**: Historical price and volume data fetched via OpenBB Platform API (2015-01-01 to present)
        2. **Data Normalization**: Robust column name normalization for consistent processing across different data sources
        3. **Technical Analysis**: Computation of moving averages, MACD, Bollinger Bands, RSI, and ATR
        4. **Risk Metrics**: Rolling Sharpe ratios, volatility measures, correlation analysis, and drawdown calculations
        5. **Visualization**: Interactive Plotly charts with dark theme styling for professional presentation
        """
    )
    
    st.markdown("---")
    
    # Section 1: Individual Asset Dashboards
    st.markdown("## Individual Asset Dashboards")
    st.markdown(
        """
        Each ETF dashboard combines three key metrics in a multi-panel view:
        
        1. **Price Chart**: Closing prices over time showing long-term trends and major market events
        2. **Volume**: Trading volume bars indicating liquidity and market participation
        3. **Returns & RSI**: Daily percentage returns with Relative Strength Index (14-period) overlay
        
        The RSI helps identify overbought (>70) and oversold (<30) conditions, providing context for 
        the returns distribution and potential reversal points.
        """
    )
    
    viz_dir = Path(__file__).parent / "notebooks" / "Visualizations"
    
    # SPY Dashboard
    st.markdown("### SPY (S&P 500 ETF)")
    spy_img = viz_dir / "SPYPrice.png"
    if spy_img.exists():
        st.image(str(spy_img), caption="SPY Dashboard: Price, Volume, and Returns with RSI", use_container_width=True)
    st.markdown(
        """
        The S&P 500 represents the broad U.S. large-cap market. Notable periods visible:
        - **2018 Q4**: Volatility spike and correction
        - **2020 Q1-Q2**: COVID-19 crash and recovery
        - **2022**: Fed tightening cycle and bear market
        - **2023-2024**: Recovery and AI-driven tech rally
        """
    )
    
    # QQQ Dashboard
    st.markdown("### QQQ (Nasdaq 100 ETF)")
    qqq_img = viz_dir / "QQQPrice.png"
    if qqq_img.exists():
        st.image(str(qqq_img), caption="QQQ Dashboard: Price, Volume, and Returns with RSI", use_container_width=True)
    st.markdown(
        """
        QQQ tracks tech-heavy Nasdaq 100. Key characteristics:
        - Higher volatility than SPY due to tech concentration
        - Strong performance during low-rate environments
        - More pronounced reactions to growth/momentum shifts
        """
    )
    
    # IWM Dashboard
    st.markdown("### IWM (Russell 2000 ETF)")
    iwm_img = viz_dir / "IWMPrice.png"
    if iwm_img.exists():
        st.image(str(iwm_img), caption="IWM Dashboard: Price, Volume, and Returns with RSI", use_container_width=True)
    st.markdown(
        """
        Small-cap stocks (IWM) show:
        - Higher sensitivity to economic cycles
        - Lower correlation with large-cap during certain regimes
        - Potential for diversification benefits in multi-asset portfolios
        """
    )
    
    # EFA Dashboard
    st.markdown("### EFA (International Developed Markets)")
    efa_img = viz_dir / "EFAPrice.png"
    if efa_img.exists():
        st.image(str(efa_img), caption="EFA Dashboard: Price, Volume, and Returns with RSI", use_container_width=True)
    st.markdown(
        """
        International diversification through EFA:
        - Currency risk exposure (USD vs. foreign currencies)
        - Different economic cycles vs. U.S.
        - Lower correlation with domestic equities provides portfolio benefits
        """
    )
    
    # GLD Dashboard
    st.markdown("### GLD (Gold ETF)")
    gld_img = viz_dir / "GLDDashboard.png"
    if gld_img.exists():
        st.image(str(gld_img), caption="GLD Dashboard: Price, Volume, and Returns with RSI", use_container_width=True)
    st.markdown(
        """
        Gold (GLD) serves as:
        - **Inflation hedge**: Preserves purchasing power during monetary debasement
        - **Crisis asset**: Safe haven during market stress (visible during 2020 crash)
        - **Portfolio diversifier**: Low/negative correlation with equities
        - **Interest rate sensitivity**: Underperforms during rising rate environments (2022)
        """
    )
    
    # TLT Dashboard
    st.markdown("### TLT (20+ Year Treasury Bond ETF)")
    tlt_img = viz_dir / "TLTDashboard.png"
    if tlt_img.exists():
        st.image(str(tlt_img), caption="TLT Dashboard: Price, Volume, and Returns with RSI", use_container_width=True)
    st.markdown(
        """
        Long-duration bonds (TLT) exhibit:
        - **Interest rate sensitivity**: High duration means large price moves with rate changes
        - **Inverse relationship with equities**: Often rises during equity selloffs (flight to quality)
        - **2022 performance**: Worst year on record as Fed aggressively raised rates
        - **Portfolio role**: Provides income and capital appreciation when rates fall
        """
    )
    
    st.markdown("---")
    
    # Section 2: Technical Indicators
    st.markdown("## Technical Indicators Analysis")
    st.markdown(
        """
        Technical indicators help identify trends, momentum, and potential reversal points. The notebook 
        computes four key indicators for each asset:
        
        ### Moving Averages (MA20, MA50)
        - **MA20**: 20-day simple moving average — short-term trend
        - **MA50**: 50-day simple moving average — medium-term trend
        - **Interpretation**: Price above MAs suggests uptrend; crossovers (Golden/Death Cross) signal trend changes
        
        ### MACD (Moving Average Convergence Divergence)
        - **Components**: MACD line (12-EMA minus 26-EMA) and Signal line (9-EMA of MACD)
        - **Trading signals**: MACD crossing above/below signal line indicates momentum shifts
        - **Divergence**: When price makes new highs but MACD doesn't, potential reversal warning
        
        ### Bollinger Bands
        - **Construction**: 20-day MA ± 2 standard deviations
        - **Mean reversion**: Price touching upper/lower bands suggests overextension
        - **Volatility measure**: Band width indicates volatility regime (wide = high vol, narrow = low vol)
        
        ### ATR (Average True Range)
        - **Purpose**: Measures volatility independent of direction
        - **Use cases**: Position sizing, stop-loss placement, volatility regime identification
        - **Interpretation**: Higher ATR = more volatile = larger potential moves
        """
    )
    
    spy_indicators = viz_dir / "SPYIndicators.png"
    if spy_indicators.exists():
        st.image(str(spy_indicators), caption="SPY Technical Indicators: Moving Averages, MACD, Bollinger Bands, and ATR", use_container_width=True)
    st.markdown(
        """
        **Analysis for SPY**:
        - Bollinger Bands show clear volatility regimes (tight bands 2017, wide bands 2020)
        - MACD captures momentum shifts before major moves
        - ATR spikes align with major market events (COVID crash, 2022 bear market)
        - Moving average crossovers often precede significant trend changes
        """
    )
    
    st.markdown("---")
    
    # Section 3: Candlestick Charts
    st.markdown("## Candlestick Charts with Overlays")
    st.markdown(
        """
        Candlestick charts provide granular price action visualization with Open-High-Low-Close (OHLC) data. 
        Combined with technical overlays, they reveal:
        
        - **Support/Resistance levels**: Where price repeatedly bounces or stalls
        - **Trend strength**: Strong trends show consistent candlestick patterns in one direction
        - **Reversal patterns**: Doji, hammer, engulfing patterns at key levels
        - **Bollinger Band interactions**: Price reactions at upper/lower bands
        """
    )
    
    spy_candles = viz_dir / "SPYCandles.png"
    if spy_candles.exists():
        st.image(str(spy_candles), caption="SPY Candlestick Chart with Moving Averages and Bollinger Bands", use_container_width=True)
    
    qqq_candles = viz_dir / "QQQCandles.png"
    if qqq_candles.exists():
        st.image(str(qqq_candles), caption="QQQ Candlestick Chart (Full View)", use_container_width=True)
    
    qqq_zoom = viz_dir / "QQQCandlesZoom.png"
    if qqq_zoom.exists():
        st.image(str(qqq_zoom), caption="QQQ Candlestick Chart (Zoomed — Last 30 Days)", use_container_width=True)
    st.markdown(
        """
        **Key Observations**:
        - Green candles (close > open) show bullish momentum
        - Red candles (close < open) show bearish pressure
        - Long wicks indicate rejection at those price levels
        - Price bouncing off Bollinger Bands suggests mean reversion opportunities
        - Zoomed view (30-day) reveals recent price action and short-term patterns
        """
    )
    
    st.markdown("---")
    
    # Section 4: Multi-Asset Comparison
    st.markdown("## Multi-Asset Risk & Performance Analysis")
    st.markdown(
        """
        Comparing assets side-by-side reveals diversification opportunities, risk characteristics, 
        and regime-dependent relationships. This section presents four key analyses.
        """
    )
    
    # Cumulative Returns
    cum_returns = viz_dir / "CumulativeReturns.png"
    if cum_returns.exists():
        st.markdown("### Cumulative Returns Comparison")
        st.image(str(cum_returns), caption="Normalized Cumulative Returns (Base = 100 at start date)", use_container_width=True)
    st.markdown(
        """
        **Interpretation**:
        - All lines start at 100 (normalized base)
        - Steeper slopes indicate higher returns over the period
        - **QQQ** shows strongest performance (tech growth bias)
        - **TLT** underperformed during the rate-hiking cycle (2022-2023)
        - **GLD** provides diversification but lower absolute returns
        - **IWM** (small-cap) underperformed large-cap (SPY) during most of the period
        """
    )
    
    # Correlation Heatmap
    corr_heatmap = viz_dir / "CorrelationHeatmap.png"
    if corr_heatmap.exists():
        st.markdown("### Correlation Heatmap")
        st.image(str(corr_heatmap), caption="Return Correlation Matrix Across Assets", use_container_width=True)
    st.markdown(
        """
        **Key Insights**:
        - **High correlations (red)**: SPY, QQQ, IWM all highly correlated (~0.8-0.9) — limited diversification within U.S. equities
        - **Negative correlations (blue)**: TLT vs. equities often negative during stress periods
        - **Low correlations**: GLD shows lower correlation with equities, providing diversification
        - **EFA correlation**: Moderate correlation with U.S. stocks (~0.6-0.7) — some international diversification benefit
        
        **Portfolio Construction Implications**:
        - Combining SPY/QQQ/IWM doesn't add much diversification (they move together)
        - Adding TLT or GLD reduces portfolio volatility
        - EFA adds geographic diversification beyond pure correlation benefits
        """
    )
    
    # Rolling Sharpe & Volatility
    rolling_sharpe = viz_dir / "RollingSharpeVol.png"
    if rolling_sharpe.exists():
        st.markdown("### Rolling Risk-Adjusted Returns & Volatility")
        st.image(str(rolling_sharpe), caption="60-Day Rolling Sharpe Ratio (Top) and Annualized Volatility (Bottom)", use_container_width=True)
    st.markdown(
        """
        **Rolling Sharpe Ratio** (60-day window):
        - Measures risk-adjusted returns: (mean return / volatility) × √252
        - Values >1 indicate good risk-adjusted performance
        - Values <0 indicate losses relative to risk-free rate
        - **Observations**:
          * QQQ shows highest Sharpe during bull markets but extreme drawdowns during corrections
          * TLT Sharpe highly variable (negative during 2022 rate hikes, positive during equity stress)
          * GLD Sharpe varies with inflation expectations and crisis periods
        
        **Rolling Volatility** (60-day, annualized):
        - Annualized standard deviation of returns: std(returns) × √252
        - **Ranking (typically)**: QQQ > IWM > SPY > EFA > GLD > TLT
        - Volatility clustering: High vol periods (2020, 2022) persist for months
        - **Regime changes**: Sudden vol spikes indicate regime shifts (COVID crash, Fed policy changes)
        """
    )
    
    # Drawdowns
    drawdowns = viz_dir / "Drawdowns.png"
    if drawdowns.exists():
        st.markdown("### Drawdown Analysis")
        st.image(str(drawdowns), caption="Maximum Drawdown from Peak (Peak-to-Trough Decline)", use_container_width=True)
    st.markdown(
        """
        **What is Drawdown?**
        - Peak-to-trough decline: % decline from the highest point to the lowest point before recovering
        - Negative values indicate drawdown (0% = at peak, -20% = 20% below peak)
        
        **Key Observations**:
        - **2020 COVID crash**: All equity ETFs (SPY, QQQ, IWM, EFA) dropped ~30-35%
        - **2022 bear market**: QQQ worst hit (~35% drawdown), SPY ~25%, TLT ~50% (worst ever)
        - **GLD resilience**: Smaller drawdowns during equity stress (2020: ~15%, 2022: ~20%)
        - **TLT**: Massive drawdown in 2022 as rates rose, but recovered as rates stabilized
        
        **Risk Management Implications**:
        - Maximum expected drawdown helps set position sizes
        - TLT's 2022 drawdown shows bonds aren't always "safe" assets
        - Diversification reduces portfolio-level drawdowns (GLD helped during 2020)
        """
    )
    
    st.markdown("---")
    
    # Section 5: Technical Implementation
    st.markdown("## Technical Implementation Details")
    st.markdown(
        """
        ### Data Pipeline
        1. **OpenBB Platform SDK**: Unified API for financial data access
           ```python
           from openbb import obb
           df = obb.equity.price.historical(symbol, start_date="2015-01-01")
           ```
        
        2. **Data Normalization**: Robust column name matching handles variations across data sources
           - Price columns: "close", "Close", "adj close", "price", etc.
           - Volume columns: "volume", "Volume", "vol", etc.
        
        3. **Indicator Computation**: Custom functions compute technical indicators from price data
           - Moving averages: Simple rolling means
           - MACD: Exponential moving average differences
           - Bollinger Bands: MA ± 2 standard deviations
           - RSI: Relative strength index using gain/loss ratios
        
        4. **Risk Metrics**:
           - Returns: `prices.pct_change()`
           - Cumulative: `(1 + returns).cumprod()`
           - Correlation: `returns.corr()`
           - Rolling Sharpe: `(mean / std) * sqrt(252)`
           - Drawdowns: `(current / running_max) - 1`
        
        ### Visualization Stack
        - **Plotly**: Interactive charts with zoom, pan, and hover tooltips
        - **Dark theme**: Professional styling consistent with financial dashboards
        - **Subplot layouts**: Multi-panel views for comprehensive analysis
        
        ### Key Features
        - **Robust error handling**: Graceful fallbacks when OHLC data unavailable
        - **Date alignment**: Handles missing data and aligns across assets
        - **Export functionality**: PNG exports for presentations and reports
        """
    )
    
    st.markdown("---")
    
    # Section 6: Connection to ML Essays
    st.markdown("## Connection to ML Financial Optimization Essays")
    st.markdown(
        """
        This OpenBB workbench directly supports the three essays in this project:
        
        ### Essay 1: SARSA-IS Portfolio Optimization
        - **Data foundation**: Returns, volume, and RSI feed into the portfolio environment
        - **Regime detection**: Technical indicators help identify disaster vs. normal regimes
        - **Risk metrics**: Drawdown analysis validates the importance sampling approach
        
        ### Essay 2: Inverse Optimization + Deep RL
        - **State-dependent features**: Rolling volatility and Sharpe ratios inform state definitions
        - **Risk aversion inference**: Correlation patterns reveal investor preferences
        - **Multi-asset context**: EFA, GLD, TLT provide diversification options for inverse optimization
        
        ### Essay 3: Neural Pricing Kernel
        - **Factor selection**: Technical indicators (MACD, Bollinger Bands) as potential pricing factors
        - **Nonlinear relationships**: Correlation heatmap shows complex relationships a linear model might miss
        - **Asset universe**: Six ETFs provide cross-sectional experiences for pricing kernel estimation
        
        ### Data Quality & Validation
        The OpenBB analysis ensures:
        - Data consistency across assets
        - Proper conversion and normalization
        - Visual validation of data quality (no obvious errors in charts)
        - Sufficient history for rolling metrics and regime identification
        """
    )
    
    st.markdown("---")
    
    # Notebook Link
    st.markdown("## Access the Notebook")
    notebook_path = Path(__file__).parent / "notebooks" / "OpenBB_Comprehensive_Workbench.ipynb"
    if notebook_path.exists():
        st.markdown(
            f"""
            The full interactive notebook is available at:
            `notebooks/OpenBB_Comprehensive_Workbench.ipynb`
            
            **To run the notebook**:
            1. Install OpenBB Platform: `pip install "openbb>=4"`
            2. Open in Jupyter: `jupyter notebook notebooks/OpenBB_Comprehensive_Workbench.ipynb`
            3. Run cells sequentially to regenerate all visualizations
            
            The notebook includes live data fetching, so charts update with current market data.
            """
        )


def main():
    # Multi-page navigation
    if page == "Main Findings":
        page_findings()
    elif page == "OpenBB Market Analysis":
        page_openbb_analysis()


if __name__ == "__main__":
    main()


