"""Live Formula 1 data client.

Uses the public Jolpica F1 API (Ergast-compatible) for current race results,
schedules, and championship standings. Responses are cached on disk so the
Streamlit app remains usable when the API is temporarily unavailable.

Jolpica API docs: https://api.jolpi.ca/ergast/f1/
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_TTL_SECONDS = 30 * 60


def _cache_path(resource: str) -> Path:
    safe = resource.replace("/", "_").replace("?", "_").replace("=", "-")
    return CACHE_DIR / f"{safe}.json"


def _request_json(path: str, *, ttl_seconds: int = CACHE_TTL_SECONDS) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(path)
    now = time.time()
    if cache_file.exists() and now - cache_file.stat().st_mtime < ttl_seconds:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    request = Request(
        f"{BASE_URL}/{path.lstrip('/')}",
        headers={"User-Agent": "f1-race-intelligence-analytics/2.0"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise RuntimeError(f"Jolpica API unavailable and no cached data exists: {exc}") from exc


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    mr = payload.get("MRData", {})
    return {
        "source": "Jolpica F1 API",
        "source_url": mr.get("url", BASE_URL),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _position(value: Any, fallback: int = 99) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def results_to_dataframe(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize Ergast/Jolpica nested race results to dashboard columns."""
    records: list[dict[str, Any]] = []
    for race in payload.get("MRData", {}).get("RaceTable", {}).get("Races", []):
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            finish = _position(result.get("position"))
            grid = _position(result.get("grid"))
            records.append({
                "race_id": f"{race.get('season')}-{race.get('round')}",
                "season": _position(race.get("season"), 0),
                "round": _position(race.get("round"), 0),
                "race_name": race.get("raceName", "Unknown Grand Prix"),
                "race_date": race.get("date"),
                "circuit_name": circuit.get("circuitName", "Unknown Circuit"),
                "place_name": location.get("locality", ""),
                "country": location.get("country", ""),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "driver_code": driver.get("code", driver.get("driverId", "UNK")).upper(),
                "constructor": constructor.get("name", "Unknown"),
                "grid_position": grid,
                "finish_position": finish,
                "position_text": result.get("positionText", str(finish)),
                "points": _float(result.get("points")),
                "podium_finish": int(finish <= 3),
                "race_winner": int(finish == 1),
                "is_top_5": int(finish <= 5),
                "is_points_finish": int(_float(result.get("points")) > 0),
                "status": result.get("status", "Unknown"),
                "is_classified": int(result.get("status", "") in {"Finished", "Lapped"}),
                "laps_completed": _position(result.get("laps"), 0),
                "total_laps": _position(result.get("laps"), 0),
                "fastest_lap": int(bool(result.get("FastestLap"))),
                "is_street_circuit": 0,
                "overtake_difficulty": 2,
                "source": "Jolpica F1 API",
            })
    return pd.DataFrame(records), _metadata(payload)


def fetch_results(season: str | int = "current", *, force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch all available race results for a season, with a cache fallback."""
    path = f"{season}/results.json?limit=2000"
    payload = _request_json(path, ttl_seconds=0 if force_refresh else CACHE_TTL_SECONDS)
    return results_to_dataframe(payload)


def fetch_schedule(season: str | int = "current", *, force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = _request_json(
        f"{season}.json?limit=2000",
        ttl_seconds=0 if force_refresh else CACHE_TTL_SECONDS,
    )
    records = []
    for race in payload.get("MRData", {}).get("RaceTable", {}).get("Races", []):
        circuit = race.get("Circuit", {})
        records.append({
            "season": _position(race.get("season"), 0),
            "round": _position(race.get("round"), 0),
            "race_name": race.get("raceName"),
            "date": race.get("date"),
            "time": race.get("time"),
            "circuit_name": circuit.get("circuitName"),
            "locality": circuit.get("Location", {}).get("locality"),
            "country": circuit.get("Location", {}).get("country"),
        })
    return pd.DataFrame(records), _metadata(payload)


def fetch_standings(season: str | int = "current", *, constructors: bool = False,
                    force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    category = "constructorstandings" if constructors else "driverstandings"
    payload = _request_json(
        f"{season}/{category}.json?limit=100",
        ttl_seconds=0 if force_refresh else CACHE_TTL_SECONDS,
    )
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    rows = lists[0].get("ConstructorStandings" if constructors else "DriverStandings", []) if lists else []
    records = []
    for row in rows:
        if constructors:
            entity = row.get("Constructor", {})
            records.append({"position": _position(row.get("position")), "name": entity.get("name"), "points": _float(row.get("points")), "wins": _position(row.get("wins"), 0)})
        else:
            entity = row.get("Driver", {})
            records.append({"position": _position(row.get("position")), "name": f"{entity.get('givenName', '')} {entity.get('familyName', '')}".strip(), "code": entity.get("code", ""), "points": _float(row.get("points")), "wins": _position(row.get("wins"), 0)})
    return pd.DataFrame(records), _metadata(payload)


def load_live_season(season: int | str = "current", *, force_refresh: bool = False) -> dict[str, Any]:
    """Convenience loader used by the dashboard."""
    results, meta = fetch_results(season, force_refresh=force_refresh)
    schedule, _ = fetch_schedule(season, force_refresh=force_refresh)
    drivers, _ = fetch_standings(season, force_refresh=force_refresh)
    constructors, _ = fetch_standings(season, constructors=True, force_refresh=force_refresh)
    return {"results": results, "schedule": schedule, "drivers": drivers, "constructors": constructors, "metadata": meta}


if __name__ == "__main__":
    live = load_live_season()
    print(live["metadata"])
    print(live["drivers"].head(10).to_string(index=False))


# OpenF1 live timing layer. OpenF1 documents real-time data as a subscription
# feature; the adapter is intentionally optional and degrades gracefully to
# Jolpica when live timing is unavailable.
OPENF1_BASE_URL = "https://api.openf1.org/v1"


def _openf1_json(path: str, *, ttl_seconds: int = 15) -> list[dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path("openf1_" + path)
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl_seconds:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    request = Request(
        f"{OPENF1_BASE_URL}/{path.lstrip('/')}",
        headers={"User-Agent": "f1-race-intelligence-analytics/2.1"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return []


def fetch_openf1_live(session_key: str | int = "latest") -> dict[str, Any]:
    """Return the latest session state and normalized timing tables.

    The endpoint works for live or recently completed sessions. During a race,
    the leaderboard can be refreshed every 10–20 seconds without restarting
    Streamlit; during practice/qualifying, lap and position data remain useful
    even when race intervals are not published.
    """
    sessions = _openf1_json(f"sessions?session_key={session_key}")
    session = sessions[-1] if sessions else {}
    actual_key = session.get("session_key", session_key)
    key = str(actual_key)
    drivers = _openf1_json(f"drivers?session_key={key}")
    positions = _openf1_json(f"position?session_key={key}")
    laps = _openf1_json(f"laps?session_key={key}")
    intervals = _openf1_json(f"intervals?session_key={key}")
    weather = _openf1_json(f"weather?session_key={key}")
    race_control = _openf1_json(f"race_control?session_key={key}")

    driver_map = {int(row.get("driver_number")): row for row in drivers if row.get("driver_number") is not None}
    latest_position: dict[int, dict[str, Any]] = {}
    for row in positions:
        if row.get("driver_number") is not None:
            latest_position[int(row["driver_number"])] = row
    latest_lap: dict[int, dict[str, Any]] = {}
    for row in laps:
        if row.get("driver_number") is not None:
            latest_lap[int(row["driver_number"])] = row
    latest_interval: dict[int, dict[str, Any]] = {}
    for row in intervals:
        if row.get("driver_number") is not None:
            latest_interval[int(row["driver_number"])] = row

    leaderboard = []
    driver_numbers = set(driver_map) | set(latest_position) | set(latest_lap) | set(latest_interval)
    for number in driver_numbers:
        driver = driver_map.get(number, {})
        pos = latest_position.get(number, {})
        lap = latest_lap.get(number, {})
        interval = latest_interval.get(number, {})
        leaderboard.append({
            "position": pos.get("position"),
            "driver_number": number,
            "driver": driver.get("name_acronym", str(number)),
            "full_name": driver.get("full_name", "Unknown"),
            "team": driver.get("team_name", "Unknown"),
            "team_colour": driver.get("team_colour", "E10600"),
            "lap": lap.get("lap_number"),
            "lap_time_sec": lap.get("lap_duration"),
            "sector_1_sec": lap.get("duration_sector_1"),
            "sector_2_sec": lap.get("duration_sector_2"),
            "sector_3_sec": lap.get("duration_sector_3"),
            "gap_to_leader_sec": interval.get("gap_to_leader"),
            "interval_sec": interval.get("interval"),
            "is_pit_out_lap": lap.get("is_pit_out_lap"),
        })
    leaderboard_df = pd.DataFrame(leaderboard)
    if not leaderboard_df.empty:
        leaderboard_df.sort_values(["position", "driver"], na_position="last", inplace=True)
        leaderboard_df.reset_index(drop=True, inplace=True)

    last_weather = weather[-1] if weather else {}
    last_event = race_control[-1] if race_control else {}
    return {
        "session": session,
        "leaderboard": leaderboard_df,
        "weather": last_weather,
        "race_control": pd.DataFrame(race_control),
        "latest_event": last_event,
        "metadata": {
            "source": "OpenF1 API",
            "session_key": actual_key,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "live_timing_available": bool(intervals or laps or positions),
        },
    }
