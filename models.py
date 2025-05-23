import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """Baseline models for benchmarking"""
    
    @staticmethod
    def naive_moving_average(X_train, y_train, X_test):
        """Simple moving average baseline for regression"""
        # Use last known volatility as prediction
        last_vol = y_train.iloc[-1] if hasattr(y_train, 'iloc') else y_train[-1]
        predictions = np.full(len(X_test), last_vol)
        return predictions
    
    @staticmethod
    def random_classifier(X_train, y_train, X_test):
        """Random classifier baseline for classification"""
        # Get class distribution from training data
        if hasattr(y_train, 'value_counts'):
            class_probs = y_train.value_counts(normalize=True)
            classes = class_probs.index
            probs = class_probs.values
        else:
            classes, counts = np.unique(y_train, return_counts=True)
            probs = counts / counts.sum()
            
        # Random predictions based on class distribution
        predictions = np.random.choice(classes, size=len(X_test), p=probs)
        return predictions

class TraditionalModels:
    """Traditional statistical models"""
    
    @staticmethod
    def fit_arima(X_train, y_train, X_test, order=(5,1,0)):
        """ARIMA model for volatility forecasting"""
        try:
            # ARIMA uses only the target variable (univariate)
            model = ARIMA(y_train, order=order)
            fitted = model.fit()
            
            # Multi-step forecast
            forecast = fitted.forecast(steps=len(X_test))
            return forecast
            
        except Exception as e:
            print(f"ARIMA error: {e}")
            # Fallback to naive forecast
            return BaselineModels.naive_moving_average(X_train, y_train, X_test)

class DeepLearningModels:
    """Deep learning implementations"""
    
    @staticmethod
    def build_lstm_regressor(input_shape, units=50):
        """LSTM for regression task"""
        model = models.Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(units//2),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def build_lstm_classifier(input_shape, num_classes=3, units=50):
        """LSTM for classification task"""
        model = models.Sequential([
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(units//2),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    @staticmethod
    def prepare_lstm_data(X, y, lookback=10):
        """Prepare data for LSTM (create sequences)"""
        X_seq, y_seq = [], []
        
        for i in range(lookback, len(X)):
            X_seq.append(X.iloc[i-lookback:i].values)
            y_seq.append(y.iloc[i])
            
        return np.array(X_seq), np.array(y_seq)

class StockMixer:
    """
    StockMixer implementation based on the paper:
    'StockMixer: A Simple yet Strong MLP-based Architecture'
    """
    
    def __init__(self, n_features, n_stocks=1, d_model=64, n_heads=4):
        self.n_features = n_features
        self.n_stocks = n_stocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.model = None
        
    def build_model(self, task='regression'):
        """Build StockMixer architecture"""
        
        # Input layer
        inputs = layers.Input(shape=(self.n_features,))
        
        # Time Mixing Layer
        time_mixed = layers.Dense(self.d_model, activation='relu')(inputs)
        time_mixed = layers.LayerNormalization()(time_mixed)
        
        # Indicator Mixing Layer
        indicator_mixed = layers.Dense(self.d_model)(time_mixed)
        indicator_mixed = layers.Activation('relu')(indicator_mixed)
        indicator_mixed = layers.Dropout(0.2)(indicator_mixed)
        indicator_mixed = layers.LayerNormalization()(indicator_mixed)
        
        # Stock Mixing Layer (simplified for single stock)
        stock_mixed = layers.Dense(self.d_model)(indicator_mixed)
        stock_mixed = layers.Activation('relu')(stock_mixed)
        stock_mixed = layers.Dropout(0.2)(stock_mixed)
        
        # Output layer
        if task == 'regression':
            outputs = layers.Dense(1)(stock_mixed)
            loss = 'mse'
            metrics = ['mae']
        else:
            outputs = layers.Dense(3, activation='softmax')(stock_mixed)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
            
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss=loss, metrics=metrics)
        
        return self.model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the model"""
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

class ModelEvaluator:
    """Unified model evaluation framework"""
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """Evaluate regression performance"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def evaluate_classification(y_true, y_pred):
        """Evaluate classification performance"""
        # Convert string labels to numeric if needed
        if isinstance(y_true[0], str):
            label_map = {'Low': 0, 'Medium': 1, 'High': 2}
            y_true = [label_map.get(y, y) for y in y_true]
            y_pred = [label_map.get(y, y) for y in y_pred]
            
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred, average='weighted'),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted')
        }
        return metrics

def run_model_comparison(pipeline, ticker='AAPL', task='regression'):
    """Run comprehensive model comparison"""
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON - {ticker} ({task})")
    print(f"{'='*60}")
    
    # Prepare data
    X, y, scaler = pipeline.prepare_ml_data(ticker, task=task)
    
    # Get walk-forward splits
    splits = pipeline.walk_forward_split(X, y, n_splits=3)  # Reduced for demo
    
    # Store results
    results = {}
    
    for model_name in ['Baseline', 'ARIMA', 'XGBoost', 'LSTM', 'StockMixer']:
        print(f"\nTraining {model_name}...")
        model_results = []
        
        for i, split in enumerate(splits):
            X_train, y_train = split['train']
            X_test, y_test = split['test']
            
            try:
                if task == 'regression':
                    # Regression models
                    if model_name == 'Baseline':
                        y_pred = BaselineModels.naive_moving_average(X_train, y_train, X_test)
                    elif model_name == 'ARIMA':
                        y_pred = TraditionalModels.fit_arima(X_train, y_train, X_test)
                    elif model_name == 'XGBoost':
                        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                        xgb_model.fit(X_train, y_train)
                        y_pred = xgb_model.predict(X_test)
                    elif model_name == 'LSTM':
                        # Prepare sequences
                        X_train_seq, y_train_seq = DeepLearningModels.prepare_lstm_data(X_train, y_train)
                        X_test_seq, y_test_seq = DeepLearningModels.prepare_lstm_data(
                            pd.concat([X_train.tail(10), X_test]), 
                            pd.concat([y_train.tail(10), y_test])
                        )
                        X_test_seq = X_test_seq[-len(X_test):]
                        y_test_seq = y_test_seq[-len(X_test):]
                        
                        # Build and train
                        lstm_model = DeepLearningModels.build_lstm_regressor((10, X.shape[1]))
                        lstm_model.fit(X_train_seq, y_train_seq, epochs=20, verbose=0)
                        y_pred = lstm_model.predict(X_test_seq).flatten()
                        y_test = y_test_seq
                    elif model_name == 'StockMixer':
                        sm = StockMixer(n_features=X_train.shape[1])
                        sm.build_model(task='regression')
                        sm.fit(X_train, y_train, epochs=30)
                        y_pred = sm.predict(X_test).flatten()
                    
                    # Evaluate
                    metrics = ModelEvaluator.evaluate_regression(y_test, y_pred)
                    
                else:
                    # Classification models
                    if model_name == 'Baseline':
                        y_pred = BaselineModels.random_classifier(X_train, y_train, X_test)
                    elif model_name == 'XGBoost':
                        # Convert labels to numeric
                        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
                        y_train_num = y_train.map(label_map)
                        
                        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                        xgb_model.fit(X_train, y_train_num)
                        y_pred_num = xgb_model.predict(X_test)
                        
                        # Convert back to labels
                        reverse_map = {v: k for k, v in label_map.items()}
                        y_pred = [reverse_map[p] for p in y_pred_num]
                    elif model_name == 'LSTM':
                        # Similar to regression but with classifier
                        pass  # Skip for brevity
                    elif model_name == 'StockMixer':
                        # Convert labels
                        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
                        y_train_num = y_train.map(label_map)
                        
                        sm = StockMixer(n_features=X_train.shape[1])
                        sm.build_model(task='classification')
                        sm.fit(X_train, y_train_num, epochs=30)
                        y_pred_num = np.argmax(sm.predict(X_test), axis=1)
                        
                        # Convert back
                        reverse_map = {v: k for k, v in label_map.items()}
                        y_pred = [reverse_map[p] for p in y_pred_num]
                    
                    # Evaluate
                    metrics = ModelEvaluator.evaluate_classification(y_test, y_pred)
                
                metrics['regime'] = split['regime']
                model_results.append(metrics)
                
            except Exception as e:
                print(f"  Error in fold {i+1}: {e}")
                continue
        
        if model_results:
            results[model_name] = model_results
            
            # Print average performance
            avg_metrics = {}
            for key in model_results[0].keys():
                if key != 'regime':
                    values = [r[key] for r in model_results if key in r]
                    if values:
                        avg_metrics[key] = np.mean(values)
            
            print(f"  Average performance: {avg_metrics}")
    
    return results

# Example usage
if __name__ == "__main__":
    # This would be run with the pipeline from the previous artifact
    print("Model implementations ready for use with RiskPipeline")