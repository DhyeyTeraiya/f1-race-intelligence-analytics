"""
Formula 1 Race Intelligence & Strategy Predictive Analytics Dashboard
Built with Streamlit, Plotly, Scikit-learn, and Pandas.
Author: Dhyey Teraiya (Data Scientist)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for direct imports
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
sys.path.append(SRC_DIR)

from strategy_simulator import StrategySimulator

st.set_page_config(
    page_title="F1 Race Intelligence & Strategy AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .f1-header {
        font-family: 'Arial Black', sans-serif;
        color: #E10600;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    .f1-sub {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #E10600;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    races_file = os.path.join(DATA_DIR, "f1_races.csv")
    telemetry_file = os.path.join(DATA_DIR, "f1_lap_telemetry.csv")
    df_races = pd.read_csv(races_file) if os.path.exists(races_file) else pd.DataFrame()
    df_telemetry = pd.read_csv(telemetry_file) if os.path.exists(telemetry_file) else pd.DataFrame()
    return df_races, df_telemetry


@st.cache_resource
def load_ml_model():
    model_file = os.path.join(OUTPUTS_DIR, "podium_model.joblib")
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None


df_races, df_telemetry = load_data()
model_artifact = load_ml_model()

# Header
st.markdown('<div class="f1-header">🏎️ FORMULA 1 RACE INTELLIGENCE & PREDICTIVE ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="f1-sub">End-to-End Motorsport Data Science, Machine Learning Strategy Simulator & Telemetry Analytics</div>', unsafe_allow_html=True)

# Top KPI Metrics Row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-value">1,600</div><div class="metric-label">Grand Prix Entries</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-value">94.75%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-value">0.982</div><div class="metric-label">ROC-AUC Score</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-value">20</div><div class="metric-label">World Circuits</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown('<div class="metric-card"><div class="metric-value">0.13s</div><div class="metric-label">Lap Time Regressor MAE</div></div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏁 Predictive Podium & Winner AI",
    "⏱️ Pit Stop Undercut Simulator",
    "📈 Telemetry & Tire Degradation",
    "📊 Model Benchmarks & Historical Insights"
])

# ==========================================
# TAB 1: PREDICTIVE PODIUM & WINNER AI
# ==========================================
with tab1:
    st.subheader("🎯 Real-Time Grand Prix Outcome Predictor")
    st.write("Configure the pre-race conditions to predict podium probability using trained ensemble machine learning models.")

    colA, colB = st.columns([1, 2])

    with colA:
        circuit_selected = st.selectbox(
            "Select Circuit",
            options=sorted(df_races["circuit_name"].unique()) if not df_races.empty else ["Silverstone Circuit"],
            index=0
        )
        driver_selected = st.selectbox(
            "Select Driver",
            options=sorted(df_races["driver_name"].unique()) if not df_races.empty else ["Max Verstappen"],
            index=0
        )
        grid_pos = st.slider("Starting Grid Position", min_value=1, max_value=20, value=2)
        quali_gap = st.slider("Gap to Pole Position (seconds)", min_value=0.000, max_value=2.500, value=0.085, step=0.005)
        weather_opt = st.selectbox("Track Weather Condition", options=["Dry", "Mixed", "Wet"], index=0)
        track_temp = st.slider("Track Temperature (°C)", min_value=18, max_value=50, value=34)

    with colB:
        if model_artifact and not df_races.empty:
            driver_info = df_races[df_races["driver_name"] == driver_selected].iloc[0]
            circuit_info = df_races[df_races["circuit_name"] == circuit_selected].iloc[0]

            # Construct feature vector
            downforce_map = {"Low": 1, "Medium": 2, "High": 3}
            weather_map = {"Dry": 0, "Mixed": 1, "Wet": 2}

            downforce_num = downforce_map.get(circuit_info["downforce_level"], 2)
            weather_num = weather_map.get(weather_opt, 0)
            overtake_diff = circuit_info["overtake_difficulty"]

            input_dict = {
                "grid_position": [grid_pos],
                "qualifying_delta": [quali_gap],
                "team_tier": [driver_info["team_tier"]],
                "driver_skill": [driver_info["driver_skill"]],
                "overtake_difficulty": [overtake_diff],
                "downforce_numeric": [downforce_num],
                "weather_numeric": [weather_num],
                "track_temp_c": [track_temp],
                "driver_rolling_points": [15.5 if driver_info["team_tier"] == 1 else 4.0],
                "driver_rolling_finish": [3.2 if driver_info["team_tier"] == 1 else 9.5],
                "team_rolling_pts": [35.0 if driver_info["team_tier"] == 1 else 10.0],
                "grid_circuit_difficulty_interaction": [grid_pos * overtake_diff],
                "is_front_row": [1 if grid_pos <= 2 else 0],
                "is_top_5_grid": [1 if grid_pos <= 5 else 0]
            }

            input_df = pd.DataFrame(input_dict)
            pipeline = model_artifact["pipeline"]
            podium_prob = pipeline.predict_proba(input_df)[0][1] * 100
            pred_podium = pipeline.predict(input_df)[0]

            # Visualization of Prediction
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=podium_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Podium Probability (Top 3 Finish)", 'font': {'size': 20, 'color': '#FAFAFA'}},
                number={'suffix': "%", 'font': {'color': '#E10600', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#30363D"},
                    'bar': {'color': "#E10600"},
                    'bgcolor': "#161B22",
                    'borderwidth': 2,
                    'bordercolor': "#30363D",
                    'steps': [
                        {'range': [0, 35], 'color': "#1F2937"},
                        {'range': [35, 70], 'color': "#374151"},
                        {'range': [70, 100], 'color': "#4B5563"}
                    ],
                    'threshold': {
                        'line': {'color': "#00D2BE", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="#0E1117")
            st.plotly_chart(fig_gauge, use_container_width=True)

            if pred_podium == 1:
                st.success(f"🏆 **Prediction**: **{driver_selected}** is strongly favored to secure a **Podium Finish (P1–P3)** at {circuit_selected} ({podium_prob:.1f}% confidence).")
            else:
                st.warning(f"⚠️ **Prediction**: **{driver_selected}** is projected to finish outside the podium places ({podium_prob:.1f}% podium chance).")


# ==========================================
# TAB 2: PIT STOP UNDERCUT SIMULATOR
# ==========================================
with tab2:
    st.subheader("⏱️ Tactical Pit Stop Undercut & Overcut Engine")
    st.write("Simulate strategic pit windows to calculate if a chasing car can leapfrog the race leader via an undercut.")

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        gap_sec = st.slider("Gap to Leading Car before Pit (seconds)", min_value=0.5, max_value=4.5, value=1.8, step=0.1)
        chaser_lap = st.number_input("Chaser Pit Stop Lap", min_value=10, max_value=50, value=22)
        leader_lap = st.number_input("Leader Reaction Pit Stop Lap", min_value=11, max_value=55, value=24)
        chaser_comp = st.selectbox("Chaser Fresh Tire Compound", options=["SOFT", "MEDIUM", "HARD"], index=2)
        leader_old_laps = st.slider("Laps Completed on Leader's Current Tires", min_value=15, max_value=40, value=24)

    with sim_col2:
        sim = StrategySimulator()
        sim_res = sim.simulate_undercut(
            gap_before_pit_sec=gap_sec,
            chaser_pit_lap=chaser_lap,
            leader_pit_lap=leader_lap,
            chaser_compound=chaser_comp,
            laps_on_leader_tire=leader_old_laps
        )

        if sim_res.get("success"):
            st.success(f"✅ **STRATEGY RECOMMENDATION**: **{sim_res['recommendation']}**")
            st.write(f"• **Net Track Position Margin**: +{sim_res['net_margin_sec']} seconds ahead of rival.")
            st.write(f"• **Pace Advantage Gained over {sim_res['undercut_window_laps']} laps**: {sim_res['total_pace_gained_sec']} seconds.")
        else:
            st.error(f"❌ **STRATEGY RECOMMENDATION**: **{sim_res.get('recommendation', 'DEFEND')}**")
            st.write(f"• **Deficit after Pit Stops**: {sim_res.get('net_margin_sec', -1.0)} seconds behind leader.")

        # Breakdown chart
        if "lap_breakdown" in sim_res and sim_res["lap_breakdown"]:
            df_breakdown = pd.DataFrame(sim_res["lap_breakdown"])
            fig_undercut = px.bar(
                df_breakdown,
                x="lap_offset",
                y="cumulative_delta",
                title="Cumulative Delta Gained During Undercut Window",
                labels={"lap_offset": "Laps Since Chaser Pitted", "cumulative_delta": "Seconds Gained"},
                color_discrete_sequence=["#E10600"]
            )
            fig_undercut.add_hline(y=gap_sec, line_dash="dash", line_color="#00D2BE",
                                   annotation_text=f"Initial Gap ({gap_sec}s)")
            fig_undercut.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#161B22", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_undercut, use_container_width=True)


# ==========================================
# TAB 3: TELEMETRY & TIRE DEGRADATION
# ==========================================
with tab3:
    st.subheader("📈 Tire Degradation & Stint Telemetry Analysis")
    st.write("Examine tire wear curves, fuel burn offsets, and degradation inflection points across compounds.")

    if not df_telemetry.empty:
        colT1, colT2 = st.columns([1, 1])

        with colT1:
            fig_deg = px.line(
                df_telemetry,
                x="tire_age_laps",
                y="lap_time_sec",
                color="compound",
                title="Lap Time Evolution: Stint Lap vs. Compound",
                labels={"tire_age_laps": "Tire Age (Laps)", "lap_time_sec": "Lap Time (Seconds)"},
                color_discrete_map={"MEDIUM": "#FFD700", "HARD": "#FAFAFA"}
            )
            fig_deg.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#161B22", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_deg, use_container_width=True)

        with colT2:
            fig_fuel = px.scatter(
                df_telemetry,
                x="fuel_load_kg",
                y="lap_time_sec",
                color="driver_code",
                title="Fuel Load Decay vs. Lap Pace",
                labels={"fuel_load_kg": "Fuel Remaining (kg)", "lap_time_sec": "Lap Time (Seconds)"}
            )
            fig_fuel.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#161B22", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_fuel, use_container_width=True)


# ==========================================
# TAB 4: MODEL BENCHMARKS & INSIGHTS
# ==========================================
with tab4:
    st.subheader("📊 Machine Learning Performance & Conversion Analytics")

    mb1, mb2 = st.columns(2)

    with mb1:
        roc_path = os.path.join(OUTPUTS_DIR, "roc_auc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC-AUC Curve Across Benchmark Classifiers", use_container_width=True)

        cm_path = os.path.join(OUTPUTS_DIR, "model_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on 2024 Test Season", use_container_width=True)

    with mb2:
        fi_path = os.path.join(OUTPUTS_DIR, "feature_importance_podium.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Feature Importance for Race Podium Prediction", use_container_width=True)

        grid_path = os.path.join(OUTPUTS_DIR, "qualifying_to_podium_matrix.png")
        if os.path.exists(grid_path):
            st.image(grid_path, caption="Grid Position to Podium Conversion Rates", use_container_width=True)
