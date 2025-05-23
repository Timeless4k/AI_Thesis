# main.py - Complete Thesis Implementation Runner
"""
Main script to run the complete thesis experiment for short-term volatility forecasting.
This implements the methodology described in the thesis document.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import all components
from risk_pipeline import RiskPipeline
from models import (
    BaselineModels, TraditionalModels, DeepLearningModels, 
    StockMixer, ModelEvaluator, run_model_comparison
)
from experiment_runner import ExperimentRunner, InterpretabilityAnalysis

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'yfinance', 'sklearn', 
        'statsmodels', 'xgboost', 'tensorflow', 'shap'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
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

def run_full_experiment():
    """Run the complete thesis experiment"""
    print("\n" + "="*80)
    print("FULL THESIS EXPERIMENT - SHORT-TERM VOLATILITY FORECASTING")
    print("="*80)
    print("This will take considerable time to complete...")
    print("="*80)
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir='./thesis_results')
    
    # Run complete experiment
    results = runner.run_complete_experiment()
    
    print("\nExperiment completed!")
    print("Results saved in: ./thesis_results/")
    
    return results

def main():
    """Main entry point"""
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n" + "="*80)
    print("THESIS IMPLEMENTATION: SHORT-TERM VOLATILITY FORECASTING")
    print("Using ML/DL Models for AU and US Financial Markets")
    print("="*80)
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Run full experiment
        run_full_experiment()
    else:
        # Run quick demo
        run_quick_demo()
        print("\nTo run the full experiment, use: python main.py --full")

if __name__ == "__main__":
    main()

# setup.py - Installation helper
"""
Setup script to prepare the environment for thesis experiments
"""

import subprocess
import sys

def setup_environment():
    """Set up the Python environment"""
    
    print("Setting up thesis experiment environment...")
    
    # Install required packages
    print("\nInstalling required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create necessary directories
    import os
    directories = ['./results', './thesis_results', './logs', './models']
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        print(f"Created directory: {dir}")
    
    print("\nEnvironment setup complete!")
    print("You can now run the experiments using: python main.py")

if __name__ == "__main__":
    setup_environment()