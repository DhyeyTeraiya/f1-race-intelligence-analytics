"""Regression tests for the F1 analytics and live-data pipeline."""
import os
import sys
import unittest

import joblib
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
sys.path.append(SRC_DIR)

from data_loader import load_raw_data
from feature_engineering import get_modeling_data
from live_data import results_to_dataframe
from advanced_analytics import build_prediction_sensitivity, build_strategy_robustness_grid
from strategy_simulator import StrategySimulator


class TestF1Pipeline(unittest.TestCase):
    def test_real_dataset_integrity(self):
        races, telemetry = load_raw_data()
        self.assertFalse(races.empty)
        self.assertGreater(len(races), 2000)
        self.assertTrue({2021, 2024, 2025, 2026}.issubset(set(races["season"].unique())))
        self.assertGreater(len(telemetry), 1000)

    def test_feature_engineering_and_chronological_split(self):
        data = get_modeling_data()
        self.assertGreater(len(data["X_train"]), 1000)
        self.assertGreater(len(data["X_test"]), 500)
        self.assertEqual(len(data["X_train"]), len(data["y_train_podium"]))
        for feature in ("driver_rolling_points", "team_rolling_pts", "is_front_row"):
            self.assertIn(feature, data["df_full"].columns)

    def test_live_result_parser_normalizes_nested_api_data(self):
        payload = {"MRData": {"url": "https://api.jolpi.ca/ergast/f1/2026/results.json", "RaceTable": {"Races": [{
            "season": "2026", "round": "1", "raceName": "Test Grand Prix", "date": "2026-03-08",
            "Circuit": {"circuitName": "Test Circuit", "Location": {"locality": "Test City", "country": "Testland"}},
            "Results": [{"position": "1", "positionText": "1", "grid": "2", "points": "25", "laps": "58", "status": "Finished",
                          "Driver": {"givenName": "Test", "familyName": "Driver", "code": "TST"},
                          "Constructor": {"name": "Test Racing"}, "FastestLap": {"rank": "1"}}]
        }]}}}
        frame, meta = results_to_dataframe(payload)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["driver_name"], "Test Driver")
        self.assertEqual(frame.iloc[0]["constructor"], "Test Racing")
        self.assertEqual(int(frame.iloc[0]["podium_finish"]), 1)
        self.assertEqual(meta["source"], "Jolpica F1 API")

    def test_strategy_undercut_logic(self):
        result = StrategySimulator().simulate_undercut(1.5, 20, 22)
        self.assertIn("success", result)
        self.assertIn("net_margin_sec", result)
        self.assertIn("recommendation", result)

    def test_trained_model_inference(self):
        artifact = joblib.load(os.path.join(OUTPUTS_DIR, "podium_model.joblib"))
        sample = pd.DataFrame([{
            "grid_position": 1, "overtake_difficulty": 3, "is_street_circuit": 0, "team_tier": 1,
            "driver_rolling_points": 20.0, "driver_rolling_finish": 1.5, "team_rolling_pts": 40.0,
            "grid_circuit_difficulty_interaction": 3, "is_front_row": 1, "is_top_5_grid": 1,
            "is_top_10_grid": 1, "pit_stops_count": 1,
        }])
        probability = artifact["pipeline"].predict_proba(sample[artifact["features"]])[0][1]
        self.assertTrue(0.0 <= probability <= 1.0)

    def test_prediction_sensitivity_grid_is_complete(self):
        artifact = joblib.load(os.path.join(OUTPUTS_DIR, "podium_model.joblib"))
        sample = pd.DataFrame([{
            "grid_position": 2, "overtake_difficulty": 3, "is_street_circuit": 0, "team_tier": 1,
            "driver_rolling_points": 20.0, "driver_rolling_finish": 1.5, "team_rolling_pts": 40.0,
            "grid_circuit_difficulty_interaction": 6, "is_front_row": 1, "is_top_5_grid": 1,
            "is_top_10_grid": 1, "pit_stops_count": 1,
        }])
        result = build_prediction_sensitivity(sample, artifact, grid_positions=[1, 5, 10], pit_stops=[1, 2])
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result["grid_position"]), {1, 5, 10})
        self.assertTrue(result["podium_probability"].between(0, 100).all())

    def test_strategy_robustness_grid_is_complete(self):
        result = build_strategy_robustness_grid(
            gap_values=[1.0, 2.0], tire_ages=[10, 30], chaser_pit_lap=20, leader_pit_lap=22
        )
        self.assertEqual(len(result), 4)
        self.assertTrue({"net_margin_sec", "recommendation", "success"}.issubset(result.columns))
        self.assertTrue(result["net_margin_sec"].notna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
