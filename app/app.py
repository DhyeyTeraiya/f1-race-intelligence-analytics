"""F1 Race Intelligence dashboard with live Jolpica data and cached fallback."""
from __future__ import annotations

import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
sys.path.append(SRC_DIR)
from live_data import load_live_season  # noqa: E402
from strategy_simulator import StrategySimulator  # noqa: E402

st.set_page_config(page_title="F1 Race Intelligence", page_icon="🏎️", layout="wide")
st.markdown("""
<style>
.main { background: #0E1117; color: #FAFAFA; }
.f1-header { color: #E10600; font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
.f1-sub { color: #A0AEC0; margin: 0 0 18px 0; }
.metric-card { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px; text-align: center; }
.metric-value { color: #E10600; font-size: 1.7rem; font-weight: 800; }
.metric-label { color: #8B949E; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def get_live_data(force_refresh: bool = False):
    return load_live_season("current", force_refresh=force_refresh)


@st.cache_data
def get_bundle_data():
    races = pd.read_csv(os.path.join(DATA_DIR, "f1_races.csv"))
    telemetry_path = os.path.join(DATA_DIR, "f1_lap_telemetry.csv")
    telemetry = pd.read_csv(telemetry_path) if os.path.exists(telemetry_path) else pd.DataFrame()
    return races, telemetry


@st.cache_resource
def load_ml_model():
    path = os.path.join(OUTPUTS_DIR, "podium_model.joblib")
    return joblib.load(path) if os.path.exists(path) else None


with st.sidebar:
    st.header("Data controls")
    force_refresh = st.button("Refresh live data now", type="primary")
    st.caption("Live results and standings use the Jolpica F1 API. Cached responses are used if the API is temporarily unavailable.")

try:
    live = get_live_data(force_refresh)
    live_results = live["results"]
    live_drivers = live["drivers"]
    live_constructors = live["constructors"]
    schedule = live["schedule"]
    metadata = live["metadata"]
    live_ok = not live_results.empty
except Exception as exc:
    live = None
    live_results = pd.DataFrame()
    live_drivers = pd.DataFrame()
    live_constructors = pd.DataFrame()
    schedule = pd.DataFrame()
    metadata = {"source": "Bundled repository snapshot", "error": str(exc)}
    live_ok = False

bundle_races, telemetry = get_bundle_data()
df_races = bundle_races.copy()
if live_ok:
    current_season = int(live_results["season"].max())
    df_races = pd.concat([bundle_races[bundle_races["season"] != current_season], live_results], ignore_index=True)
else:
    current_season = int(df_races["season"].max()) if not df_races.empty else datetime.now().year

model_artifact = load_ml_model()
st.markdown('<div class="f1-header">🏎️ F1 RACE INTELLIGENCE & PREDICTIVE ANALYTICS</div>', unsafe_allow_html=True)
st.markdown(f'<div class="f1-sub">Live championship intelligence • {current_season} season • historical model window 2021–{current_season}</div>', unsafe_allow_html=True)

latest_date = pd.to_datetime(live_results.get("race_date"), errors="coerce").max() if live_ok else None
latest_date_text = latest_date.strftime("%d %b %Y") if pd.notna(latest_date) else "Bundled snapshot"
latest_race = live_results.iloc[-1]["race_name"] if live_ok and not live_results.empty else "Historical dataset"
metrics = [
    (f"{len(df_races):,}", "Race entries"),
    (str(current_season), "Live season"),
    (str(live_results["round"].nunique()) if live_ok else "—", "Rounds loaded"),
    (latest_date_text, "Latest result"),
    ("Jolpica" if live_ok else "Fallback", "Data source"),
]
cols = st.columns(5)
for col, (value, label) in zip(cols, metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
st.caption(f"{latest_race} • {metadata.get('source', 'Unknown source')} • cache TTL: 30 minutes")
if not live_ok:
    st.warning("Live API data was not available. The dashboard is showing the versioned repository snapshot instead.")

tab_live, tab_predict, tab_strategy, tab_history, tab_quality = st.tabs([
    "🟢 Live Championship", "🎯 Podium Prediction", "⏱️ Strategy Lab", "📈 Historical Analytics", "🔬 Data Quality"
])

with tab_live:
    st.subheader(f"{current_season} live championship monitor")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Drivers")
        if not live_drivers.empty:
            st.dataframe(live_drivers.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No live standings returned.")
    with right:
        st.markdown("#### Constructors")
        if not live_constructors.empty:
            st.dataframe(live_constructors.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No live constructor standings returned.")
    st.markdown("#### Latest race results")
    if live_ok and not live_results.empty:
        race_options = live_results[["round", "race_name"]].drop_duplicates().sort_values("round")
        selected_race = st.selectbox("Race", race_options["race_name"].tolist(), index=len(race_options) - 1)
        st.dataframe(live_results[live_results["race_name"] == selected_race][["finish_position", "driver_name", "constructor", "grid_position", "points", "status"]].sort_values("finish_position"), use_container_width=True, hide_index=True)
    if not schedule.empty:
        st.markdown("#### Season schedule")
        st.dataframe(schedule, use_container_width=True, hide_index=True)

with tab_predict:
    st.subheader("Data-driven podium probability")
    st.write("The production model uses the repository's historical feature set; driver form and team strength are refreshed from the latest available results.")
    if df_races.empty or model_artifact is None:
        st.error("Prediction assets are unavailable.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            circuits = sorted(df_races["circuit_name"].dropna().unique())
            drivers = sorted(df_races["driver_name"].dropna().unique())
            circuit = st.selectbox("Circuit", circuits, index=0)
            default_driver = "Max Verstappen" if "Max Verstappen" in drivers else drivers[0]
            driver = st.selectbox("Driver", drivers, index=drivers.index(default_driver))
            grid = st.slider("Starting grid", 1, 22, 2)
            stops = st.slider("Planned pit stops", 1, 3, 1)
        with c2:
            driver_rows = df_races[df_races["driver_name"] == driver].sort_values(["season", "round"])
            circuit_rows = df_races[df_races["circuit_name"] == circuit]
            row = driver_rows.iloc[-1] if not driver_rows.empty else df_races.iloc[-1]
            circuit_row = circuit_rows.iloc[-1] if not circuit_rows.empty else df_races.iloc[-1]
            team = row.get("constructor", "Unknown")
            team_rows = df_races[df_races["constructor"] == team].sort_values(["season", "round"])
            team_points = float(team_rows["points"].tail(8).sum()) if not team_rows.empty else 0.0
            input_df = pd.DataFrame([{
                "grid_position": grid,
                "overtake_difficulty": int(circuit_row.get("overtake_difficulty", 2)),
                "is_street_circuit": int(circuit_row.get("is_street_circuit", 0)),
                "team_tier": 1 if team_points >= 50 else 2 if team_points >= 25 else 3,
                "driver_rolling_points": float(driver_rows["points"].tail(4).mean()) if not driver_rows.empty else 0,
                "driver_rolling_finish": float(driver_rows["finish_position"].tail(4).mean()) if not driver_rows.empty else 12,
                "team_rolling_pts": team_points,
                "grid_circuit_difficulty_interaction": grid * int(circuit_row.get("overtake_difficulty", 2)),
                "is_front_row": int(grid <= 2), "is_top_5_grid": int(grid <= 5), "is_top_10_grid": int(grid <= 10), "pit_stops_count": stops,
            }])
            features = model_artifact["features"]
            probability = float(model_artifact["pipeline"].predict_proba(input_df[features])[0][1] * 100)
            fig = go.Figure(go.Indicator(mode="gauge+number", value=probability, number={"suffix": "%"}, title={"text": f"{driver} podium probability"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E10600"}}))
            fig.update_layout(height=300, paper_bgcolor="#0E1117", font={"color": "#FAFAFA"})
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"{driver} / {team} • recent average finish: {input_df.loc[0, 'driver_rolling_finish']:.1f} • recent team points: {team_points:.1f}")

with tab_strategy:
    st.subheader("Strategy Lab")
    a, b = st.columns([1, 2])
    with a:
        gap = st.slider("Gap before pit (s)", 0.5, 5.0, 1.8, 0.1)
        chaser_lap = st.number_input("Chaser pit lap", 10, 70, 22)
        leader_lap = st.number_input("Leader pit lap", 11, 70, 24)
        compound = st.selectbox("Fresh compound", ["SOFT", "MEDIUM", "HARD"], index=2)
        old_laps = st.slider("Leader tire age", 10, 45, 24)
    with b:
        result = StrategySimulator().simulate_undercut(gap, chaser_lap, leader_lap, compound, laps_on_leader_tire=old_laps)
        if result.get("success"):
            st.success(f"EXECUTE UNDERCUT • net margin +{result['net_margin_sec']}s")
        else:
            st.warning(f"DEFEND / EXTEND • net margin {result.get('net_margin_sec', 0)}s")
        if result.get("lap_breakdown"):
            breakdown = pd.DataFrame(result["lap_breakdown"])
            fig = px.line(breakdown, x="lap_offset", y="cumulative_delta", markers=True, title="Cumulative undercut gain")
            fig.add_hline(y=gap, line_dash="dash", annotation_text="Initial gap")
            st.plotly_chart(fig, use_container_width=True)

with tab_history:
    st.subheader("Historical performance and pace")
    d1, d2 = st.columns(2)
    with d1:
        summary = df_races.groupby("driver_name").agg(Starts=("race_id", "count"), Wins=("race_winner", "sum"), Podiums=("podium_finish", "sum"), Points=("points", "sum"), Avg_Finish=("finish_position", "mean")).sort_values("Points", ascending=False).head(15)
        st.dataframe(summary.style.format({"Points": "{:.1f}", "Avg_Finish": "{:.1f}"}), use_container_width=True)
    with d2:
        season_team = df_races.groupby(["season", "constructor"], as_index=False)["points"].sum()
        fig = px.line(season_team, x="season", y="points", color="constructor", markers=True, title="Constructor points trajectory")
        st.plotly_chart(fig, use_container_width=True)
    chart = os.path.join(OUTPUTS_DIR, "car_development_pace_gap.png")
    if os.path.exists(chart):
        st.image(chart, caption="Historical qualifying pace and race-day position delta")

with tab_quality:
    st.subheader("Data quality and provenance")
    duplicate_count = int(df_races.duplicated(["race_id", "driver_code"]).sum())
    missing_grid = int(df_races["grid_position"].isna().sum())
    quality = pd.DataFrame([
        {"Check": "Rows loaded", "Value": f"{len(df_races):,}", "Status": "PASS" if len(df_races) else "FAIL"},
        {"Check": "Duplicate driver-race rows", "Value": duplicate_count, "Status": "PASS" if duplicate_count == 0 else "REVIEW"},
        {"Check": "Missing grid positions", "Value": missing_grid, "Status": "PASS" if missing_grid == 0 else "REVIEW"},
        {"Check": "Live API rows", "Value": len(live_results), "Status": "PASS" if live_ok else "FALLBACK"},
        {"Check": "Historical telemetry", "Value": len(telemetry), "Status": "BUNDLED DERIVED DATA"},
    ])
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.markdown("**Sources:** live race results, standings, and schedule are fetched from the public [Jolpica F1 API](https://api.jolpi.ca/ergast/f1/). The repository snapshot remains available as an offline fallback. Historical lap telemetry in this repository is a derived modeling dataset and is labeled accordingly.")
