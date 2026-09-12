"""
Unit and Integration Tests for F1 Race Intelligence Analytics Pipeline
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
sys.path.append(SRC_DIR)

from data_loader import generate_race_dataset, generate_lap_telemetry_dataset
from feature_engineering import build_engineered_features, get_modeling_data
from strategy_simulator import StrategySimulator
import joblib


class TestF1Pipeline(unittest.TestCase):

    def test_01_data_generation(self):
        df_races = generate_race_dataset()
        df_laps = generate_lap_telemetry_dataset()

        self.assertFalse(df_races.empty, "Races dataset should not be empty")
        self.assertFalse(df_laps.empty, "Telemetry dataset should not be empty")
        self.assertIn("circuit_name", df_races.columns)
        self.assertIn("podium_finish", df_races.columns)
        self.assertIn("lap_time_sec", df_laps.columns)

    def test_02_feature_engineering(self):
        df_races, _ = get_modeling_data()["df_full"], None
        self.assertIn("driver_rolling_points", df_races.columns)
        self.assertIn("grid_circuit_difficulty_interaction", df_races.columns)
        self.assertIn("is_front_row", df_races.columns)

    def test_03_modeling_split(self):
        data = get_modeling_data()
        self.assertGreater(len(data["X_train"]), 0)
        self.assertGreater(len(data["X_test"]), 0)
        self.assertEqual(len(data["X_train"]), len(data["y_train_podium"]))
        self.assertEqual(len(data["X_test"]), len(data["y_test_podium"]))

    def test_04_strategy_simulator(self):
        sim = StrategySimulator()
        res = sim.simulate_undercut(gap_before_pit_sec=1.5, chaser_pit_lap=20, leader_pit_lap=22)
        self.assertIn("success", res)
        self.assertIn("net_margin_sec", res)
        self.assertIn("recommendation", res)

    def test_05_saved_model_inference(self):
        model_path = os.path.join(OUTPUTS_DIR, "podium_model.joblib")
        self.assertTrue(os.path.exists(model_path), "Model artifact should exist")
        artifact = joblib.load(model_path)
        pipeline = artifact["pipeline"]
        features = artifact["features"]

        sample_input = pd.DataFrame([{
            "grid_position": 1,
            "qualifying_delta": 0.0,
            "team_tier": 1,
            "driver_skill": 97,
            "overtake_difficulty": 3,
            "downforce_numeric": 2,
            "weather_numeric": 0,
            "track_temp_c": 32.0,
            "driver_rolling_points": 20.0,
            "driver_rolling_finish": 1.5,
            "team_rolling_pts": 40.0,
            "grid_circuit_difficulty_interaction": 3,
            "is_front_row": 1,
            "is_top_5_grid": 1
        }])

        proba = pipeline.predict_proba(sample_input[features])[0][1]
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)


if __name__ == "__main__":
    unittest.main()
