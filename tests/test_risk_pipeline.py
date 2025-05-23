import unittest
import pandas as pd
import numpy as np
from risk_pipeline import RiskPipeline

class TestRiskPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = RiskPipeline(start_date='2023-01-01', end_date='2023-03-31')
        self.pipeline.fetch_data()
        self.pipeline.calculate_returns_and_volatility()
        self.pipeline.engineer_features()
        self.pipeline.add_correlation_features()

    def test_feature_engineering(self):
        # Check features exist for at least one asset
        self.assertTrue(len(self.pipeline.features) > 0)
        for ticker, features in self.pipeline.features.items():
            self.assertIsInstance(features, pd.DataFrame)
            self.assertGreater(features.shape[1], 0)

    def test_walk_forward_split(self):
        # Use first available asset
        ticker = next(iter(self.pipeline.features))
        X, y, _ = self.pipeline.prepare_ml_data(ticker, task='regression')
        splits = self.pipeline.walk_forward_split(X, y, n_splits=3)
        self.assertEqual(len(splits), 3)
        for split in splits:
            X_train, y_train = split['train']
            X_test, y_test = split['test']
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_test), 0)

if __name__ == '__main__':
    unittest.main()
