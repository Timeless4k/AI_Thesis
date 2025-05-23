# main.py - Complete Thesis Implementation Runner
"""
Main script to run the complete thesis experiment for short-term volatility forecasting.
This implements the methodology described in the thesis document.
"""

import sys
import os
import warnings
import argparse
import platform
from datetime import datetime

warnings.filterwarnings('ignore')

# Import all components
from risk_pipeline import RiskPipeline
from models import (
    BaselineModels, TraditionalModels, DeepLearningModels, 
    StockMixer, ModelEvaluator, run_model_comparison
)
from experiment_runner import ExperimentRunner, InterpretabilityAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description="Run the short-term volatility forecasting experiment.")
    parser.add_argument('--full', action='store_true', help="Run the full thesis experiment")
    parser.add_argument('--output', type=str, default='./thesis_results', help="Directory to save results")
    parser.add_argument('--headless', action='store_true', help="Run in headless mode (no GUI backend)")
    return parser.parse_args()

def check_dependencies():
    """Check if all required packages are installed and print their versions"""
    required_packages = [
        'numpy', 'pandas', 'yfinance', 'sklearn', 
        'statsmodels', 'xgboost', 'tensorflow', 'shap', 'matplotlib', 'seaborn'
    ]
    print(f"\nPython version: {platform.python_version()}")
    missing = []
    for package in required_packages:
        try:
            pkg = __import__(package)
            version = getattr(pkg, '__version__', 'unknown')
            print(f"{package} version: {version}")
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install using: pip install -r requirements.txt")
        return False
    return True

def run_quick_demo():
    """Run a quick demonstration with limited data"""
    print("\n" + "="*80)
    print("QUICK DEMONSTRATION - VOLATILITY FORECASTING")
    print("="*80)
    
    # Initialize pipeline with shorter date range for demo
    pipeline = RiskPipeline(start_date='2023-01-01', end_date='2024-03-31')
    pipeline.run_pipeline()
    
    # Run comparison for one asset
    if 'AAPL' in pipeline.features:
        # Regression task
        print("\n--- Regression Task Demo ---")
        results = run_model_comparison(pipeline, 'AAPL', 'regression')
        
        # Classification task
        print("\n--- Classification Task Demo ---")
        results = run_model_comparison(pipeline, 'AAPL', 'classification')
    
    print("\nDemo completed! Run with --full for complete thesis experiment.")

def run_full_experiment(output_dir='./thesis_results', log_stdout=True):
    """Run the complete thesis experiment and log output"""
    print("\n" + "="*80)
    print("FULL THESIS EXPERIMENT - SHORT-TERM VOLATILITY FORECASTING")
    print("="*80)
    print("This will take considerable time to complete...")
    print("="*80)
    runner = ExperimentRunner(output_dir=output_dir)
    results = None
    if log_stdout:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(log_path, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            try:
                results = runner.run_complete_experiment()
            finally:
                sys.stdout = old_stdout
        print(f"\nExperiment completed!\nResults saved in: {output_dir}\nLog saved to: {log_path}")
    else:
        results = runner.run_complete_experiment()
        print(f"\nExperiment completed!\nResults saved in: {output_dir}")
    return results

def main():
    """Main entry point"""
    args = parse_args()
    if args.headless:
        import matplotlib
        matplotlib.use('Agg')
    # Print help if no args
    if len(sys.argv) == 1:
        print("\nRun a demo with: python main.py")
        print("Run full experiment: python main.py --full")
        print("For more options: python main.py --help")
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    print("\n" + "="*80)
    print("THESIS IMPLEMENTATION: SHORT-TERM VOLATILITY FORECASTING")
    print("Using ML/DL Models for AU and US Financial Markets")
    print("="*80)
    if args.full:
        run_full_experiment(output_dir=args.output, log_stdout=True)
    else:
        run_quick_demo()
        print("\nTo run the full experiment, use: python main.py --full")

if __name__ == "__main__":
    main()

