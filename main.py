#!/usr/bin/env python3
"""
Main entry point for the ML Financial Optimization Project.

This script runs the three essays from the PhD thesis replication project.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.essay1_sarsa_is import run_sarsa_is_experiment, visualize_essay1
from src.essay2_inverse_rl import run_inverse_rl_experiment, visualize_essay2
from src.essay3_nn_pricing import run_nn_pricing_experiment, visualize_essay3
from src.utils.visualizations import create_comprehensive_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning in Asset Pricing and Portfolio Optimization - PhD Thesis Replication"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three essays"
    )
    
    parser.add_argument(
        "--essay1",
        action="store_true",
        help="Run Essay 1: Robo-advising under rare disasters"
    )
    
    parser.add_argument(
        "--essay2",
        action="store_true",
        help="Run Essay 2: Risk aversion and portfolio optimization"
    )
    
    parser.add_argument(
        "--essay3",
        action="store_true",
        help="Run Essay 3: Nonlinear pricing kernels via neural networks"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to visualizations/ directory"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (Essay 1)"
    )
    
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=100,
        help="Number of training epochs (Essays 2 & 3)"
    )
    
    parser.add_argument(
        "--openbb-dashboard",
        action="store_true",
        help="Launch OpenBB visualization dashboard"
    )
    
    args = parser.parse_args()
    
    # If no specific essay selected, run all
    if not any([args.essay1, args.essay2, args.essay3]):
        args.all = True
    
    print("=" * 80)
    print("Machine Learning in Asset Pricing and Portfolio Optimization")
    print("PhD Thesis Replication Project")
    print("=" * 80)
    print()
    
    results = {}
    
    # Essay 1: SARSA-IS
    if args.all or args.essay1:
        print("\n" + "=" * 80)
        print("ESSAY 1: Robo-advising under Rare Disasters")
        print("SARSA with Importance Sampling")
        print("=" * 80)
        
        essay1_results = run_sarsa_is_experiment(
            episodes=args.episodes,
            visualize=args.visualize,
            save_plots=args.save_plots
        )
        results['essay1'] = essay1_results
        
        if args.visualize:
            visualize_essay1(essay1_results, save=args.save_plots)
    
    # Essay 2: Inverse RL + Deep RL
    if args.all or args.essay2:
        print("\n" + "=" * 80)
        print("ESSAY 2: Risk Aversion and Portfolio Optimization")
        print("Inverse Optimization + Deep RL (A2C)")
        print("=" * 80)
        
        essay2_results = run_inverse_rl_experiment(
            train=args.train_epochs > 0,
            epochs=args.train_epochs,
            visualize=args.visualize,
            save_plots=args.save_plots
        )
        results['essay2'] = essay2_results
        
        if args.visualize:
            visualize_essay2(essay2_results, save=args.save_plots)
    
    # Essay 3: Neural Network Pricing Kernels
    if args.all or args.essay3:
        print("\n" + "=" * 80)
        print("ESSAY 3: Nonlinear Pricing Kernels via Neural Networks")
        print("Neural Network Pricing Kernels with ESG Factors")
        print("=" * 80)
        
        essay3_results = run_nn_pricing_experiment(
            train=args.train_epochs > 0,
            epochs=args.train_epochs,
            visualize=args.visualize,
            save_plots=args.save_plots
        )
        results['essay3'] = essay3_results
        
        if args.visualize:
            visualize_essay3(essay3_results, save=args.save_plots)
    
    # Comprehensive Dashboard
    if args.visualize and (args.all or len(results) > 1):
        print("\n" + "=" * 80)
        print("Creating Comprehensive Dashboard...")
        print("=" * 80)
        
        create_comprehensive_dashboard(results, save=args.save_plots)
    
    # OpenBB Dashboard
    if args.openbb_dashboard:
        print("\n" + "=" * 80)
        print("Launching OpenBB Dashboard...")
        print("=" * 80)
        
        # Visualizations are handled by essay-specific functions using src/utils/openbb_viz.py
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()

