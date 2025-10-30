import os
from pathlib import Path
from typing import Optional, Dict
from datetime import date, timedelta
import streamlit as st


st.set_page_config(page_title="ML Financial Optimization — Findings", layout="wide")


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


# Single-page app — no additional sections


def main():
    # Single-page: render findings only
    page_findings()


if __name__ == "__main__":
    main()


