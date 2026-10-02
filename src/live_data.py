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
