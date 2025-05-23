# Thesis Implementation: Short-term Volatility Forecasting

This implementation corresponds to the thesis on "Short-term Volatility Forecasting Using ML/DL Models for Australian and US Financial Markets".

## Setup Instructions

1. **Install Python 3.8 or higher**

2. **Clone/Download the code files:**
   - risk_pipeline.py (Core data pipeline)
   - models.py (Model implementations)
   - experiment_runner.py (Experiment orchestration and SHAP)
   - main.py (Main entry point)
   - requirements.txt (Dependencies)
   - setup.py (Setup helper)

3. **Set up the environment:**
   ```bash
   python setup.py
   ```

4. **Run a quick demo:**
   ```bash
   python main.py
   ```

5. **Run the full thesis experiment:**
   ```bash
   python main.py --full
   ```

## Expected Outputs

The experiment will generate:
- Model performance metrics (RMSE, MAE, R², F1, etc.)
- Walk-forward cross-validation results
- SHAP interpretability plots
- Regime-based analysis
- Comprehensive summary reports

All results are saved in the `./thesis_results/` directory.

## Computational Requirements

- **Demo mode**: ~5-10 minutes on a standard laptop
- **Full experiment**: ~2-4 hours depending on hardware
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: ~1GB for data and results

## Key Features Implemented

1. **Data Pipeline (RiskPipeline)**
   - Fetches historical data for 6 assets
   - Engineers technical and correlation features
   - Implements walk-forward cross-validation

2. **Models**
   - Traditional: ARIMA
   - Deep Learning: LSTM, StockMixer
   - Ensemble: XGBoost
   - Baselines for comparison

3. **Interpretability**
   - SHAP global feature importance
   - SHAP local explanations
   - Regime-based performance analysis

4. **Evaluation**
   - Multiple metrics for regression and classification
   - Cross-market comparison (US vs AU)
   - Comprehensive reporting

## Troubleshooting

- **Memory issues**: Reduce the date range or number of walk-forward splits
- **Missing data**: Some tickers may not have complete historical data
- **GPU usage**: TensorFlow will use GPU if available, but CPU is sufficient

## Results Interpretation

The experiment will identify:
- Which models perform best for volatility forecasting
- Most important features via SHAP analysis
- Performance differences across market regimes
- Potential for cross-market model transfer

Refer to the thesis document for detailed interpretation of results.
"""