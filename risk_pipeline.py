import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class RiskPipeline:
    """
    Core pipeline for volatility forecasting as described in the thesis.
    Handles data fetching, feature engineering, model training, and evaluation.
    """
    
    def __init__(self, start_date='2017-01-01', end_date='2024-03-31'):
        self.start_date = start_date
        self.end_date = end_date
        
        # Define assets as per thesis
        self.assets = {
            'US': ['^GSPC', 'AAPL', 'MSFT'],
            'AU': ['IOZ.AX', 'CBA.AX', 'BHP.AX']
        }
        
        # Feature configuration
        self.feature_config = {
            'lag_days': [1, 2, 3],
            'ma_windows': [10, 50],
            'rolling_windows': [5, 30],
            'correlation_pairs': [
                ('AAPL', '^GSPC'),
                ('IOZ.AX', 'CBA.AX'),
                ('BHP.AX', 'IOZ.AX')
            ]
        }
        
        self.data = {}
        self.features = {}
        self.targets = {}
        
    def fetch_data(self):
        """Fetch historical data for all assets including VIX"""
        print("Fetching historical data...")
        
        # Fetch asset data
        all_tickers = self.assets['US'] + self.assets['AU'] + ['^VIX']
        
        for ticker in all_tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if len(df) > 0:
                    self.data[ticker] = df[['Adj Close', 'Volume']]
                    print(f"✓ Fetched {ticker}: {len(df)} days")
                else:
                    print(f"✗ No data for {ticker}")
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
                
    def calculate_returns_and_volatility(self):
        """Calculate log returns and rolling volatility"""
        print("\nCalculating returns and volatility...")
        
        for ticker in self.data:
            df = self.data[ticker].copy()
            
            # Log returns
            df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            
            # 5-day rolling volatility (annualized)
            df['Volatility5D'] = df['log_return'].rolling(5).std() * np.sqrt(252)
            
            # Volatility regime classification (quantile-based)
            df['Volatility5D_clean'] = df['Volatility5D'].dropna()
            if len(df['Volatility5D_clean']) > 0:
                quantiles = df['Volatility5D_clean'].quantile([0.33, 0.67])
                df['VolRegime'] = pd.cut(df['Volatility5D'], 
                                        bins=[-np.inf, quantiles[0.33], quantiles[0.67], np.inf],
                                        labels=['Low', 'Medium', 'High'])
            
            self.data[ticker] = df
            
    def engineer_features(self):
        """Engineer features as specified in thesis Section 2.3"""
        print("\nEngineering features...")
        
        for ticker in [t for t in self.data if t not in ['^VIX']]:
            df = self.data[ticker].copy()
            features = pd.DataFrame(index=df.index)
            
            # 1. Lagged returns
            for lag in self.feature_config['lag_days']:
                features[f'Lag{lag}'] = df['log_return'].shift(lag)
            
            # 2. Rate of Change (5-day)
            features['ROC5'] = (df['Adj Close'] / df['Adj Close'].shift(5) - 1) * 100
            
            # 3. Moving averages
            features['MA10'] = df['Adj Close'].rolling(10).mean()
            features['MA50'] = df['Adj Close'].rolling(50).mean()
            features['MA_ratio'] = features['MA10'] / features['MA50']
            
            # 4. Rolling standard deviation
            features['RollingStd5'] = df['log_return'].rolling(5).std()
            
            # 5. Volume indicators
            features['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # 6. VIX features (if available)
            if '^VIX' in self.data:
                vix_df = self.data['^VIX'].copy()
                features['VIX'] = vix_df['Adj Close']
                features['VIX_change'] = vix_df['Adj Close'].pct_change()
            
            # Store features and targets
            self.features[ticker] = features
            self.targets[ticker] = {
                'regression': df['Volatility5D'].shift(-1),  # Predict next period
                'classification': df['VolRegime'].shift(-1)
            }
            
    def add_correlation_features(self):
        """Add inter-asset correlation features"""
        print("\nCalculating inter-asset correlations...")
        
        for pair in self.feature_config['correlation_pairs']:
            ticker1, ticker2 = pair
            
            if ticker1 in self.data and ticker2 in self.data:
                # Get returns for both assets
                returns1 = self.data[ticker1]['log_return']
                returns2 = self.data[ticker2]['log_return']
                
                # Calculate rolling correlation
                corr_series = returns1.rolling(30).corr(returns2)
                
                # Add to features of first ticker
                if ticker1 in self.features:
                    self.features[ticker1][f'Corr_{ticker1}_{ticker2}'] = corr_series
                    
    def prepare_ml_data(self, ticker, task='regression'):
        """Prepare data for ML models with proper preprocessing"""
        
        # Get features and target
        X = self.features[ticker].copy()
        y = self.targets[ticker][task].copy()
        
        # Remove NaN values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        return X_scaled, y, scaler
    
    def walk_forward_split(self, X, y, n_splits=5):
        """
        Implement walk-forward cross-validation as per thesis Section 2.5
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(X)//10)
        
        splits = []
        for train_idx, test_idx in tscv.split(X):
            # Get train/test data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Determine market regime for test period
            if hasattr(y_test, 'index'):
                test_returns = self.data[X.index.name if hasattr(X.index, 'name') else 'Unknown']['log_return'].loc[y_test.index]
                regime = self._classify_regime(test_returns)
            else:
                regime = 'Unknown'
                
            splits.append({
                'train': (X_train, y_train),
                'test': (X_test, y_test),
                'regime': regime
            })
            
        return splits
    
    def _classify_regime(self, returns):
        """Classify market regime based on return slope"""
        if len(returns) < 20:
            return 'Unknown'
            
        # Calculate trend using linear regression
        x = np.arange(len(returns))
        slope = np.polyfit(x, returns.fillna(0), 1)[0]
        
        # Classify based on slope and volatility
        volatility = returns.std()
        
        if slope > 0.001 and volatility < 0.02:
            return 'Bull'
        elif slope < -0.001 and volatility < 0.02:
            return 'Bear'
        else:
            return 'Sideways'
            
    def run_pipeline(self):
        """Execute the complete pipeline"""
        print("="*60)
        print("RISK PIPELINE EXECUTION")
        print("="*60)
        
        # Step 1: Fetch data
        self.fetch_data()
        
        # Step 2: Calculate returns and volatility
        self.calculate_returns_and_volatility()
        
        # Step 3: Engineer features
        self.engineer_features()
        
        # Step 4: Add correlation features
        self.add_correlation_features()
        
        print("\nPipeline execution complete!")
        print(f"Assets processed: {len(self.features)}")
        
        # Display sample features for first US asset
        if 'AAPL' in self.features:
            print(f"\nSample features for AAPL:")
            print(self.features['AAPL'].tail())
            
        return self

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RiskPipeline()
    
    # Run the pipeline
    pipeline.run_pipeline()
    
    # Prepare data for a specific asset
    if 'AAPL' in pipeline.features:
        X, y, scaler = pipeline.prepare_ml_data('AAPL', task='regression')
        print(f"\nPrepared data shape: X={X.shape}, y={y.shape}")
        
        # Get walk-forward splits
        splits = pipeline.walk_forward_split(X, y)
        print(f"Number of walk-forward splits: {len(splits)}")
        
        for i, split in enumerate(splits):
            X_train, y_train = split['train']
            X_test, y_test = split['test']
            print(f"\nSplit {i+1}: Train={len(X_train)}, Test={len(X_test)}, Regime={split['regime']}")