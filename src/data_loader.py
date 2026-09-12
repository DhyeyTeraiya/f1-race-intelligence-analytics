"""
Formula 1 Data Generator & ETL Ingestion Pipeline
Generates high-fidelity modern era (2021-2024) F1 race results, qualifying metrics,
weather conditions, pit stops, and lap-by-lap tire telemetry.
"""

import os
import numpy as np
import pandas as pd

# Set deterministic seed for reproducible data science experiments
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

CIRCUITS = [
    {"name": "Bahrain International Circuit", "country": "Bahrain", "downforce": "Medium", "overtake_diff": 2, "base_lap": 91.5, "laps": 57},
    {"name": "Jeddah Corniche Circuit", "country": "Saudi Arabia", "downforce": "Low", "overtake_diff": 3, "base_lap": 88.0, "laps": 50},
    {"name": "Albert Park Circuit", "country": "Australia", "downforce": "Medium", "overtake_diff": 3, "base_lap": 78.5, "laps": 58},
    {"name": "Baku City Circuit", "country": "Azerbaijan", "downforce": "Low", "overtake_diff": 2, "base_lap": 103.0, "laps": 51},
    {"name": "Circuit de Monaco", "country": "Monaco", "downforce": "High", "overtake_diff": 5, "base_lap": 72.0, "laps": 78},
    {"name": "Circuit de Barcelona-Catalunya", "country": "Spain", "downforce": "High", "overtake_diff": 4, "base_lap": 74.0, "laps": 66},
    {"name": "Circuit Gilles Villeneuve", "country": "Canada", "downforce": "Low", "overtake_diff": 3, "base_lap": 73.0, "laps": 70},
    {"name": "Red Bull Ring", "country": "Austria", "downforce": "Medium", "overtake_diff": 2, "base_lap": 65.5, "laps": 71},
    {"name": "Silverstone Circuit", "country": "United Kingdom", "downforce": "High", "overtake_diff": 2, "base_lap": 88.5, "laps": 52},
    {"name": "Hungaroring", "country": "Hungary", "downforce": "High", "overtake_diff": 4, "base_lap": 77.0, "laps": 70},
    {"name": "Circuit de Spa-Francorchamps", "country": "Belgium", "downforce": "Medium", "overtake_diff": 2, "base_lap": 105.0, "laps": 44},
    {"name": "Circuit Zandvoort", "country": "Netherlands", "downforce": "High", "overtake_diff": 4, "base_lap": 71.0, "laps": 72},
    {"name": "Autodromo Nazionale Monza", "country": "Italy", "downforce": "Low", "overtake_diff": 2, "base_lap": 81.5, "laps": 53},
    {"name": "Marina Bay Street Circuit", "country": "Singapore", "downforce": "High", "overtake_diff": 5, "base_lap": 96.0, "laps": 62},
    {"name": "Suzuka International Racing Course", "country": "Japan", "downforce": "High", "overtake_diff": 3, "base_lap": 89.0, "laps": 53},
    {"name": "Circuit of the Americas", "country": "USA", "downforce": "Medium", "overtake_diff": 3, "base_lap": 96.5, "laps": 56},
    {"name": "Autódromo Hermanos Rodríguez", "country": "Mexico", "downforce": "High", "overtake_diff": 3, "base_lap": 78.0, "laps": 71},
    {"name": "Autódromo José Carlos Pace", "country": "Brazil", "downforce": "Medium", "overtake_diff": 2, "base_lap": 71.5, "laps": 71},
    {"name": "Las Vegas Strip Circuit", "country": "USA", "downforce": "Low", "overtake_diff": 2, "base_lap": 94.0, "laps": 50},
    {"name": "Yas Marina Circuit", "country": "Abu Dhabi", "downforce": "Medium", "overtake_diff": 3, "base_lap": 84.5, "laps": 58},
]

DRIVERS = [
    {"name": "Max Verstappen", "code": "VER", "constructor": "Red Bull Racing", "skill_rating": 97, "team_tier": 1},
    {"name": "Lewis Hamilton", "code": "HAM", "constructor": "Mercedes", "skill_rating": 95, "team_tier": 1},
    {"name": "Charles Leclerc", "code": "LEC", "constructor": "Ferrari", "skill_rating": 94, "team_tier": 1},
    {"name": "Lando Norris", "code": "NOR", "constructor": "McLaren", "skill_rating": 93, "team_tier": 1},
    {"name": "George Russell", "code": "RUS", "constructor": "Mercedes", "skill_rating": 91, "team_tier": 1},
    {"name": "Carlos Sainz", "code": "SAI", "constructor": "Ferrari", "skill_rating": 91, "team_tier": 1},
    {"name": "Oscar Piastri", "code": "PIA", "constructor": "McLaren", "skill_rating": 90, "team_tier": 1},
    {"name": "Sergio Perez", "code": "PER", "constructor": "Red Bull Racing", "skill_rating": 88, "team_tier": 1},
    {"name": "Fernando Alonso", "code": "ALO", "constructor": "Aston Martin", "skill_rating": 92, "team_tier": 2},
    {"name": "Lance Stroll", "code": "STR", "constructor": "Aston Martin", "skill_rating": 81, "team_tier": 2},
    {"name": "Pierre Gasly", "code": "GAS", "constructor": "Alpine", "skill_rating": 85, "team_tier": 2},
    {"name": "Esteban Ocon", "code": "OCO", "constructor": "Alpine", "skill_rating": 84, "team_tier": 2},
    {"name": "Alexander Albon", "code": "ALB", "constructor": "Williams", "skill_rating": 86, "team_tier": 3},
    {"name": "Yuki Tsunoda", "code": "TSU", "constructor": "RB", "skill_rating": 83, "team_tier": 3},
    {"name": "Daniel Ricciardo", "code": "RIC", "constructor": "RB", "skill_rating": 84, "team_tier": 3},
    {"name": "Nico Hulkenberg", "code": "HUL", "constructor": "Haas", "skill_rating": 84, "team_tier": 3},
    {"name": "Kevin Magnussen", "code": "MAG", "constructor": "Haas", "skill_rating": 82, "team_tier": 3},
    {"name": "Valtteri Bottas", "code": "BOT", "constructor": "Kick Sauber", "skill_rating": 83, "team_tier": 4},
    {"name": "Zhou Guanyu", "code": "ZHO", "constructor": "Kick Sauber", "skill_rating": 79, "team_tier": 4},
    {"name": "Logan Sargeant", "code": "SAR", "constructor": "Williams", "skill_rating": 76, "team_tier": 4},
]

SEASONS = [2021, 2022, 2023, 2024]
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def generate_race_dataset():
    records = []
    race_id = 1

    for season in SEASONS:
        for round_num, circuit in enumerate(CIRCUITS, 1):
            weather = np.random.choice(["Dry", "Dry", "Dry", "Wet", "Mixed"], p=[0.75, 0.1, 0.05, 0.05, 0.05])
            track_temp = np.random.uniform(25, 48) if weather == "Dry" else np.random.uniform(18, 26)
            air_temp = track_temp - np.random.uniform(5, 12)
            safety_car_laps = int(np.random.exponential(scale=2.5))
            if circuit["overtake_diff"] >= 4:
                safety_car_laps += np.random.choice([0, 3, 5])

            # Generate qualifying performance based on car tier + driver rating + noise
            driver_scores = []
            for d in DRIVERS:
                car_factor = (5 - d["team_tier"]) * 1.8
                driver_factor = (d["skill_rating"] - 75) * 0.25
                noise = np.random.normal(0, 0.35)
                score = car_factor + driver_factor + noise
                driver_scores.append((score, d))

            driver_scores.sort(key=lambda x: x[0], reverse=True)

            # Assign grid positions and qualifying times
            pole_time = circuit["base_lap"]
            race_entries = []

            for grid_pos, (score, driver) in enumerate(driver_scores, 1):
                gap_to_pole = (grid_pos - 1) * np.random.uniform(0.06, 0.14)
                if grid_pos == 1:
                    gap_to_pole = 0.000
                quali_time = round(pole_time + gap_to_pole, 3)

                # Race pace score
                race_pace = score * 0.65 + np.random.normal(0, 0.45)
                # DNF chance (mechanical, crash)
                dnf_prob = 0.04 + (0.03 if weather != "Dry" else 0)
                is_dnf = np.random.random() < dnf_prob

                race_entries.append({
                    "race_id": race_id,
                    "season": season,
                    "round": round_num,
                    "circuit_name": circuit["name"],
                    "country": circuit["country"],
                    "downforce_level": circuit["downforce"],
                    "overtake_difficulty": circuit["overtake_diff"],
                    "laps_total": circuit["laps"],
                    "weather": weather,
                    "track_temp_c": round(track_temp, 1),
                    "air_temp_c": round(air_temp, 1),
                    "safety_car_laps": safety_car_laps,
                    "driver_name": driver["name"],
                    "driver_code": driver["code"],
                    "constructor": driver["constructor"],
                    "team_tier": driver["team_tier"],
                    "driver_skill": driver["skill_rating"],
                    "grid_position": grid_pos,
                    "qualifying_time_sec": quali_time,
                    "qualifying_delta": round(gap_to_pole, 3),
                    "race_pace": race_pace,
                    "is_dnf": is_dnf
                })

            # Determine race finish order based on grid + pace + circuit overtake difficulty
            finished = [e for e in race_entries if not e["is_dnf"]]
            dnfs = [e for e in race_entries if e["is_dnf"]]

            # Higher overtake difficulty weights grid position more heavily
            grid_weight = 0.35 + (circuit["overtake_diff"] * 0.08)
            pace_weight = 1.0 - grid_weight

            for e in finished:
                e["final_score"] = (21 - e["grid_position"]) * grid_weight + (e["race_pace"] * 4.5) * pace_weight

            finished.sort(key=lambda x: x["final_score"], reverse=True)

            for finish_pos, e in enumerate(finished, 1):
                e["finish_position"] = finish_pos
                e["points"] = POINTS_MAP.get(finish_pos, 0)
                e["podium_finish"] = 1 if finish_pos <= 3 else 0
                e["race_winner"] = 1 if finish_pos == 1 else 0
                e["status"] = "Finished"

                # Pit stops
                stops = np.random.choice([1, 2, 3], p=[0.45, 0.48, 0.07])
                if weather != "Dry":
                    stops += 1
                e["pit_stops_count"] = stops
                e["avg_pit_stop_sec"] = round(np.random.uniform(2.1, 3.4), 2)
                e["fastest_lap"] = 1 if finish_pos <= 5 and np.random.random() < 0.25 else 0
                if e["fastest_lap"] and finish_pos <= 10:
                    e["points"] += 1

            for idx, e in enumerate(dnfs):
                e["finish_position"] = 20 - idx
                e["points"] = 0
                e["podium_finish"] = 0
                e["race_winner"] = 0
                e["status"] = np.random.choice(["Collision", "Power Unit", "Hydraulics", "Suspension"])
                e["pit_stops_count"] = np.random.choice([0, 1])
                e["avg_pit_stop_sec"] = round(np.random.uniform(2.4, 4.0), 2)
                e["fastest_lap"] = 0

            records.extend(finished)
            records.extend(dnfs)
            race_id += 1

    df = pd.DataFrame(records)
    # Clean up internal fields
    if "final_score" in df.columns:
        df.drop(columns=["final_score"], inplace=True)
    if "is_dnf" in df.columns:
        df.drop(columns=["is_dnf"], inplace=True)

    csv_path = os.path.join(RAW_DIR, "f1_races.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} race entries across {len(SEASONS)} seasons at {csv_path}")
    return df


def generate_lap_telemetry_dataset():
    """
    Generates detailed lap-by-lap telemetry for tire degradation analysis
    and stint modeling across Soft, Medium, and Hard compounds.
    """
    laps_data = []
    compounds = {
        "SOFT": {"deg_rate": 0.085, "base_offset": -0.75, "cliff_lap": 18},
        "MEDIUM": {"deg_rate": 0.045, "base_offset": 0.00, "cliff_lap": 28},
        "HARD": {"deg_rate": 0.024, "base_offset": 0.65, "cliff_lap": 42},
    }

    base_lap = 85.0
    for driver_code, driver_skill in [("VER", 97), ("HAM", 95), ("LEC", 94), ("NOR", 93), ("RUS", 91), ("SAI", 91)]:
        for stint, compound_name in enumerate(["MEDIUM", "HARD"]):
            comp_props = compounds[compound_name]
            stint_length = 25 if compound_name == "MEDIUM" else 30

            for lap_in_stint in range(1, stint_length + 1):
                # Fuel burn effect (~0.033 sec/lap faster as fuel is consumed)
                fuel_effect = -0.033 * (lap_in_stint + (stint * 25))

                # Tire degradation
                if lap_in_stint <= comp_props["cliff_lap"]:
                    tire_deg = comp_props["deg_rate"] * (lap_in_stint ** 1.15)
                else:
                    # Exponential cliff
                    excess = lap_in_stint - comp_props["cliff_lap"]
                    tire_deg = (comp_props["deg_rate"] * comp_props["cliff_lap"]) + (excess ** 1.8 * 0.22)

                skill_advantage = -((driver_skill - 90) * 0.06)
                noise = np.random.normal(0, 0.12)
                lap_time = round(base_lap + comp_props["base_offset"] + fuel_effect + tire_deg + skill_advantage + noise, 3)

                laps_data.append({
                    "driver_code": driver_code,
                    "stint": stint + 1,
                    "compound": compound_name,
                    "lap_in_stint": lap_in_stint,
                    "total_lap": lap_in_stint if stint == 0 else lap_in_stint + 25,
                    "fuel_load_kg": round(105 - (lap_in_stint * 1.8), 1),
                    "tire_age_laps": lap_in_stint,
                    "lap_time_sec": lap_time,
                    "speed_trap_kmh": round(315 + np.random.uniform(-4, 6), 1),
                    "throttle_pct": round(68 + np.random.uniform(-2, 3), 1),
                    "brake_wear_pct": round(min(100, lap_in_stint * 2.2), 1),
                })

    df_laps = pd.DataFrame(laps_data)
    csv_path = os.path.join(RAW_DIR, "f1_lap_telemetry.csv")
    df_laps.to_csv(csv_path, index=False)
    print(f"Generated {len(df_laps)} lap telemetry data points at {csv_path}")
    return df_laps


if __name__ == "__main__":
    generate_race_dataset()
    generate_lap_telemetry_dataset()
