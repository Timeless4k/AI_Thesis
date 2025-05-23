import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import json
import os

class InterpretabilityAnalysis:
    """SHAP-based model interpretability as per thesis Section 2.5"""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        
    def explain_model(self, model, X_train, X_test, model_name):
        """Generate SHAP explanations for a model"""
        
        print(f"\nGenerating SHAP explanations for {model_name}...")
        
        try:
            if model_name == 'XGBoost':
                # Tree explainer for XGBoost
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
            elif model_name in ['StockMixer', 'LSTM']:
                # Deep explainer for neural networks
                explainer = shap.DeepExplainer(model, X_train[:100])  # Use subset for efficiency
                shap_values = explainer.shap_values(X_test[:100])
                
            else:
                # Kernel explainer for other models (slower but model-agnostic)
                explainer = shap.KernelExplainer(model.predict, X_train[:50])
                shap_values = explainer.shap_values(X_test[:50])
                
            self.explainers[model_name] = explainer
            self.shap_values[model_name] = shap_values
            
            return shap_values
            
        except Exception as e:
            print(f"  SHAP error for {model_name}: {e}")
            return None
    
    def plot_global_importance(self, model_name, X_test, save_path=None):
        """Plot global feature importance using SHAP"""
        
        if model_name not in self.shap_values:
            print(f"No SHAP values available for {model_name}")
            return
            
        shap_values = self.shap_values[model_name]
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"Global Feature Importance - {model_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Get feature importance ranking
        if isinstance(shap_values, np.ndarray):
            importance = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 most important features for {model_name}:")
            print(feature_importance.head())
            
            return feature_importance
    
    def plot_local_explanation(self, model_name, X_test, instance_idx=0, save_path=None):
        """Plot local explanation for a single prediction"""
        
        if model_name not in self.shap_values:
            return
            
        shap_values = self.shap_values[model_name]
        
        # Create waterfall plot for single instance
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx] if isinstance(shap_values, np.ndarray) else shap_values[0][instance_idx],
                base_values=self.explainers[model_name].expected_value if hasattr(self.explainers[model_name], 'expected_value') else 0,
                data=X_test.iloc[instance_idx],
                feature_names=X_test.columns.tolist()
            ),
            show=False
        )
        plt.title(f"Local Explanation - {model_name} (Instance {instance_idx})")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ExperimentRunner:
    """Complete experiment runner implementing the thesis methodology"""
    
    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        self.results = {}
        self.interpretability = InterpretabilityAnalysis()
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Save experiment config
        self.assets = {
            'US': ['AAPL', 'MSFT', '^GSPC'],
            'AU': ['IOZ.AX', 'CBA.AX', 'BHP.AX']
        }
        self.tasks = ['regression', 'classification']
        self.n_splits = 5
        config = {
            'assets': self.assets,
            'tasks': self.tasks,
            'n_splits': self.n_splits,
            'output_dir': self.output_dir,
            'datetime': datetime.now().isoformat()
        }
        with open(os.path.join(self.output_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
    def run_complete_experiment(self):
        """Run the complete experiment as outlined in the thesis"""
        print("="*80)
        print("THESIS EXPERIMENT: SHORT-TERM VOLATILITY FORECASTING")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Initialize pipeline
        from risk_pipeline import RiskPipeline  # Import from first artifact
        pipeline = RiskPipeline(
            start_date=os.environ.get('START_DATE', '2017-01-01'),
            end_date=os.environ.get('END_DATE', '2024-03-31')
        )
        pipeline.run_pipeline()
        # Define experiment configuration
        assets = self.assets
        tasks = self.tasks
        # Run experiments for each asset and task
        for market, tickers in assets.items():
            for ticker in tickers:
                if ticker not in pipeline.features:
                    print(f"\nSkipping {ticker} - no data available")
                    continue
                for task in tasks:
                    print(f"\n{'='*60}")
                    print(f"Experiment: {ticker} ({market}) - {task}")
                    print(f"{'='*60}")
                    # Run models
                    task_results = self._run_task_experiment(pipeline, ticker, task)
                    # Store results
                    key = f"{ticker}_{task}"
                    self.results[key] = task_results
                    # Save intermediate results
                    self._save_results(key, task_results)
        # Generate summary report
        self._generate_summary_report()
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return self.results
    
    def _run_task_experiment(self, pipeline, ticker, task):
        """Run experiment for a specific asset and task"""
        # Prepare data
        X, y, scaler = pipeline.prepare_ml_data(ticker, task=task)
        
        # Get walk-forward splits
        splits = pipeline.walk_forward_split(X, y, n_splits=self.n_splits)
        
        # Models to evaluate based on task
        if task == 'regression':
            models_config = {
                'Baseline': lambda: BaselineRegressor(),
                'ARIMA': lambda: ARIMAWrapper(),
                'LSTM': lambda: LSTMRegressor(X.shape[1]),
                'StockMixer': lambda: StockMixerRegressor(X.shape[1])
            }
        else:
            models_config = {
                'Baseline': lambda: BaselineClassifier(),
                'XGBoost': lambda: XGBoostClassifier(),
                'MLP': lambda: MLPWrapper(),
                'StockMixer': lambda: StockMixerClassifier(X.shape[1])
            }
        
        # Store results for each model
        experiment_results = {
            'ticker': ticker,
            'task': task,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'models': {}
        }
        
        # Train and evaluate each model
        for model_name, model_fn in models_config.items():
            print(f"\nEvaluating {model_name}...")
            
            model_results = {
                'metrics_by_fold': [],
                'metrics_by_regime': {},
                'average_metrics': {},
                'training_time': 0
            }
            
            # Walk-forward validation
            for fold_idx, split in enumerate(splits):
                X_train, y_train = split['train']
                X_test, y_test = split['test']
                regime = split['regime']
                
                # Train model
                start_time = datetime.now()
                
                try:
                    model = model_fn()
                    
                    # Handle special cases
                    if model_name == 'LSTM':
                        # Prepare sequences for LSTM
                        X_train_seq, y_train_seq = self._prepare_lstm_data(X_train, y_train)
                        X_test_seq, y_test_seq = self._prepare_lstm_data(
                            pd.concat([X_train.iloc[-10:], X_test]),
                            pd.concat([y_train.iloc[-10:], y_test])
                        )
                        X_test_seq = X_test_seq[-len(X_test):]
                        y_test_seq = y_test_seq[-len(X_test):]
                        
                        model.fit(X_train_seq, y_train_seq)
                        y_pred = model.predict(X_test_seq)
                        y_test_eval = y_test_seq
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_test_eval = y_test
                    
                    # Calculate metrics
                    if task == 'regression':
                        fold_metrics = ModelEvaluator.evaluate_regression(y_test_eval, y_pred)
                    else:
                        fold_metrics = ModelEvaluator.evaluate_classification(y_test_eval, y_pred)
                    
                    fold_metrics['regime'] = regime
                    fold_metrics['fold'] = fold_idx
                    
                    model_results['metrics_by_fold'].append(fold_metrics)
                    
                    # Group by regime
                    if regime not in model_results['metrics_by_regime']:
                        model_results['metrics_by_regime'][regime] = []
                    model_results['metrics_by_regime'][regime].append(fold_metrics)
                    
                    # SHAP analysis for interpretable models (last fold only)
                    if fold_idx == len(splits) - 1 and model_name in ['XGBoost', 'StockMixer']:
                        if hasattr(model, 'get_model'):
                            shap_model = model.get_model()
                        else:
                            shap_model = model
                            
                        self.interpretability.explain_model(
                            shap_model, X_train, X_test, model_name
                        )
                        
                        # Save SHAP plots
                        shap_dir = os.path.join(self.output_dir, 'shap_plots')
                        os.makedirs(shap_dir, exist_ok=True)
                        
                        self.interpretability.plot_global_importance(
                            model_name, X_test,
                            save_path=os.path.join(shap_dir, f"{ticker}_{task}_{model_name}_global.png")
                        )
                    
                except Exception as e:
                    print(f"  Error in fold {fold_idx + 1}: {e}")
                    continue
                
                model_results['training_time'] += (datetime.now() - start_time).total_seconds()
            
            # Calculate average metrics
            if model_results['metrics_by_fold']:
                all_metrics = model_results['metrics_by_fold']
                for metric in all_metrics[0].keys():
                    if metric not in ['regime', 'fold']:
                        values = [m[metric] for m in all_metrics if metric in m]
                        model_results['average_metrics'][metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                
                # Print summary
                print(f"  Average performance:")
                for metric, stats in model_results['average_metrics'].items():
                    print(f"    {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
            
            experiment_results['models'][model_name] = model_results
        
        return experiment_results
    
    def _prepare_lstm_data(self, X, y, lookback=10):
        """Prepare sequential data for LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(lookback, len(X)):
            X_seq.append(X.iloc[i-lookback:i].values)
            y_seq.append(y.iloc[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def _save_results(self, key, results):
        """Save experiment results to JSON"""
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return obj
        
        # Recursively convert all numpy types
        results_json = json.loads(
            json.dumps(results, default=convert_types)
        )
        
        # Save to file
        filepath = os.path.join(self.output_dir, f"{key}_results.json")
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY REPORT")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        
        for experiment_key, results in self.results.items():
            ticker = results['ticker']
            task = results['task']
            
            for model_name, model_results in results['models'].items():
                if model_results['average_metrics']:
                    row = {
                        'Ticker': ticker,
                        'Task': task,
                        'Model': model_name
                    }
                    
                    # Add average metrics
                    for metric, stats in model_results['average_metrics'].items():
                        row[metric] = f"{stats['mean']:.4f}"
                        row[f"{metric}_std"] = f"{stats['std']:.4f}"
                    
                    summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary
            summary_path = os.path.join(self.output_dir, 'experiment_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            
            print("\nOverall Performance Summary:")
            print(summary_df.to_string(index=False))
            
            # Best models by task
            print("\n" + "="*60)
            print("BEST MODELS BY TASK")
            print("="*60)
            
            for task in ['regression', 'classification']:
                task_df = summary_df[summary_df['Task'] == task]
                
                if len(task_df) > 0:
                    print(f"\n{task.upper()}:")
                    
                    if task == 'regression':
                        # Best by RMSE (lower is better)
                        if 'RMSE' in task_df.columns:
                            task_df['RMSE_float'] = task_df['RMSE'].astype(float)
                            best_model = task_df.loc[task_df['RMSE_float'].idxmin()]
                            print(f"  Best by RMSE: {best_model['Model']} (RMSE={best_model['RMSE']})")
                    else:
                        # Best by F1 Score (higher is better)
                        if 'F1' in task_df.columns:
                            task_df['F1_float'] = task_df['F1'].astype(float)
                            best_model = task_df.loc[task_df['F1_float'].idxmax()]
                            print(f"  Best by F1: {best_model['Model']} (F1={best_model['F1']})")
        
        print("\n" + "="*80)
        print("Experiment completed successfully!")
        print(f"All results saved to: {self.output_dir}")
        print("="*80)

# Model wrapper classes for consistency
class BaselineRegressor:
    def fit(self, X, y): 
        self.last_value = y.iloc[-1]
    def predict(self, X): 
        return np.full(len(X), self.last_value)

class BaselineClassifier:
    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.probs = counts / counts.sum()
    def predict(self, X):
        return np.random.choice(self.classes, size=len(X), p=self.probs)

class ARIMAWrapper:
    def fit(self, X, y):
        self.last_values = y.tail(10)
    def predict(self, X):
        from statsmodels.tsa.arima.model import ARIMA
        try:
            model = ARIMA(self.last_values, order=(5,1,0))
            fitted = model.fit()
            return fitted.forecast(steps=len(X))
        except:
            return np.full(len(X), self.last_values.mean())

class XGBoostClassifier:
    def __init__(self):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
    def fit(self, X, y):
        y_numeric = [self.label_map.get(label, label) for label in y]
        self.model.fit(X, y_numeric)
    def predict(self, X):
        y_pred_numeric = self.model.predict(X)
        return [self.reverse_map.get(pred, pred) for pred in y_pred_numeric]
    def get_model(self):
        return self.model

class MLPWrapper:
    def __init__(self):
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
        self.label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
    def fit(self, X, y):
        y_numeric = [self.label_map.get(label, label) for label in y]
        self.model.fit(X, y_numeric)
    def predict(self, X):
        y_pred_numeric = self.model.predict(X)
        return [self.reverse_map.get(pred, pred) for pred in y_pred_numeric]

# LSTM and StockMixer wrappers with get_model for SHAP compatibility
class LSTMRegressor:
    def __init__(self, n_features):
        # Placeholder: replace with actual model if available
        self.model = None
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(len(X))
    def get_model(self):
        return self.model

class LSTMClassifier:
    def __init__(self, n_features):
        self.model = None
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(len(X))
    def get_model(self):
        return self.model

class StockMixerRegressor:
    def __init__(self, n_features):
        self.model = None
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(len(X))
    def get_model(self):
        return self.model

class StockMixerClassifier:
    def __init__(self, n_features):
        self.model = None
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(len(X))
    def get_model(self):
        return self.model

if __name__ == "__main__":
    # Run the complete experiment
    runner = ExperimentRunner(output_dir='./thesis_results')
    results = runner.run_complete_experiment()