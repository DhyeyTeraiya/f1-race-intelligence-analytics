"""
Unit and Integration Tests for Real Formula 1 Race Intelligence Pipeline
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
sys.path.append(SRC_DIR)

from data_loader import load_raw_data
from feature_engineering import build_engineered_features, get_modeling_data
from strategy_simulator import StrategySimulator
import joblib


class TestRealF1Pipeline(unittest.TestCase):

    def test_01_real_dataset_integrity(self):
        df_races, df_telemetry = load_raw_data()
        self.assertFalse(df_races.empty, "Races dataset should not be empty")
        self.assertGreater(len(df_races), 2000, "Should have over 2000 official race entries")
        self.assertIn("circuit_name", df_races.columns)
        self.assertIn("podium_finish", df_races.columns)
        self.assertIn("grid_position", df_races.columns)

        # Check that modern seasons exist
        seasons = df_races["season"].unique()
        self.assertIn(2021, seasons)
        self.assertIn(2024, seasons)
        self.assertIn(2025, seasons)
        self.assertIn(2026, seasons)

    def test_02_feature_engineering_pipeline(self):
        data = get_modeling_data()
        df = data["df_full"]
        self.assertIn("driver_rolling_points", df.columns)
        self.assertIn("team_rolling_pts", df.columns)
        self.assertIn("grid_circuit_difficulty_interaction", df.columns)
        self.assertIn("is_front_row", df.columns)

    def test_03_chronological_splits(self):
        data = get_modeling_data()
        # Train on <=2024, test on >=2025
        self.assertGreater(len(data["X_train"]), 1000)
        self.assertGreater(len(data["X_test"]), 500)
        self.assertEqual(len(data["X_train"]), len(data["y_train_podium"]))
        self.assertEqual(len(data["X_test"]), len(data["y_test_podium"]))

    def test_04_strategy_undercut_logic(self):
        sim = StrategySimulator()
        res = sim.simulate_undercut(gap_before_pit_sec=1.5, chaser_pit_lap=20, leader_pit_lap=22)
        self.assertIn("success", res)
        self.assertIn("net_margin_sec", res)
        self.assertIn("recommendation", res)

    def test_05_trained_model_inference(self):
        model_path = os.path.join(OUTPUTS_DIR, "podium_model.joblib")
        self.assertTrue(os.path.exists(model_path), "Model artifact should exist")
        artifact = joblib.load(model_path)
        pipeline = artifact["pipeline"]
        features = artifact["features"]

        sample_input = pd.DataFrame([{
            "grid_position": 1,
            "overtake_difficulty": 3,
            "is_street_circuit": 0,
            "team_tier": 1,
            "driver_rolling_points": 20.0,
            "driver_rolling_finish": 1.5,
            "team_rolling_pts": 40.0,
            "grid_circuit_difficulty_interaction": 3,
            "is_front_row": 1,
            "is_top_5_grid": 1,
            "is_top_10_grid": 1,
            "pit_stops_count": 1
        }])

        proba = pipeline.predict_proba(sample_input[features])[0][1]
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)


if __name__ == "__main__":
    unittest.main()
