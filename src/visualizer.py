"""
Formula 1 Visual Analytics & Comparative Performance Engine
Generates publication-quality charts for:
1. Driver Wins & Podiums Evolution (2021-2026)
2. Constructor Dominance & Championship Share Shift (2021-2026)
3. Car Development Trajectory & Pace Gap Analysis
4. Driver Efficiency & Grid-to-Finish Conversion Profiles
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10
})

TEAM_COLORS = {
    "Red Bull": "#1E41FF",
    "Mercedes": "#00D2BE",
    "Ferrari": "#E10600",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "Haas": "#B6BABD",
    "Sauber": "#52E252",
    "Kick Sauber": "#52E252",
    "AlphaTauri": "#5E8FAA",
    "RB": "#6692FF"
}


def load_f1_data():
    races_csv = os.path.join(RAW_DIR, "f1_races.csv")
    df = pd.read_csv(races_csv)
    # Clean driver names if necessary
    df["driver_clean"] = df["driver_name"].str.replace("", "e")
    return df


def plot_driver_wins_and_podiums(df: pd.DataFrame):
    """
    Plots real driver wins and podiums across the 2021-2026 era.
    """
    driver_stats = df.groupby("driver_clean").agg(
        Wins=("race_winner", "sum"),
        Podiums=("podium_finish", "sum"),
        Points=("points", "sum")
    ).sort_values(by="Wins", ascending=False).head(8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Plot 1: Total Wins
    bars1 = ax1.barh(driver_stats.index[::-1], driver_stats["Wins"][::-1], color="#E10600", alpha=0.88, edgecolor="#8B0000")
    ax1.set_title("Total Grand Prix Victories (2021-2026)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Race Wins", fontsize=11)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=10, fontweight="bold")

    # Plot 2: Total Podiums
    podium_stats = df.groupby("driver_clean")["podium_finish"].sum().sort_values(ascending=False).head(8)
    bars2 = ax2.barh(podium_stats.index[::-1], podium_stats.values[::-1], color="#1E41FF", alpha=0.88, edgecolor="#00008B")
    ax2.set_title("Total Podium Finishes (2021-2026)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Podiums (P1 - P3)", fontsize=11)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "driver_wins_podiums_evolution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SUCCESS] Exported Driver Wins & Podiums chart to {out_path}")


def plot_constructor_dominance_shift(df: pd.DataFrame):
    """
    Plots real constructor points and win evolution across seasons 2021 through 2026.
    Shows the major shift from Red Bull dominance to McLaren and Mercedes.
    """
    top_teams = ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Aston Martin"]
    df_top = df[df["constructor"].isin(top_teams)]

    yearly_pts = df_top.groupby(["season", "constructor"])["points"].sum().unstack(fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Plot 1: Points Trend Lines
    for team in top_teams:
        if team in yearly_pts.columns:
            ax1.plot(yearly_pts.index, yearly_pts[team], marker="o", lw=3, label=team,
                     color=TEAM_COLORS.get(team, "#333333"), markersize=7)

    ax1.set_title("Constructor Championship Points Evolution (2021-2026)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Championship Season", fontsize=11)
    ax1.set_ylabel("Championship Points", fontsize=11)
    ax1.set_xticks(yearly_pts.index)
    ax1.legend(frameon=True, fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Team Win Share by Season
    yearly_wins = df.groupby(["season", "constructor"])["race_winner"].sum().unstack(fill_value=0)
    yearly_win_pct = yearly_wins.div(yearly_wins.sum(axis=1), axis=0) * 100

    # Stacked bar of win shares
    bottom = np.zeros(len(yearly_win_pct))
    for team in ["Red Bull", "Mercedes", "McLaren", "Ferrari", "Alpine"]:
        if team in yearly_win_pct.columns:
            vals = yearly_win_pct[team].values
            ax2.bar(yearly_win_pct.index, vals, bottom=bottom, label=team,
                    color=TEAM_COLORS.get(team, "#777777"), alpha=0.88, edgecolor="white", width=0.6)
            bottom += vals

    ax2.set_title("Grand Prix Win Share % by Constructor (2021-2026)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Championship Season", fontsize=11)
    ax2.set_ylabel("Share of Total Wins (%)", fontsize=11)
    ax2.set_xticks(yearly_win_pct.index)
    ax2.legend(frameon=True, fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "constructor_dominance_shift.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SUCCESS] Exported Constructor Dominance Shift chart to {out_path}")


def plot_car_development_pace_gap(df: pd.DataFrame):
    """
    Evaluates 'who has the better car/development' by analyzing average starting grid
    rank per constructor across seasons (lower grid position = faster car).
    """
    top_teams = ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Aston Martin"]
    df_top = df[df["constructor"].isin(top_teams)]

    # Lower average grid position indicates higher aerodynamic & engine performance
    pace_index = df_top.groupby(["season", "constructor"])["grid_position"].mean().unstack(fill_value=15.0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for team in top_teams:
        if team in pace_index.columns:
            ax.plot(pace_index.index, pace_index[team], marker="s", lw=2.8, label=team,
                    color=TEAM_COLORS.get(team, "#333333"), markersize=7)

    # Invert Y-axis so P1 is at the top!
    ax.invert_yaxis()
    ax.set_title("Car Development & Qualifying Performance Index (2021-2026)\n[Lower/Higher on chart = Faster Car on Qualifying Pace]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Championship Season", fontsize=11)
    ax.set_ylabel("Average Starting Grid Position (P1 is top)", fontsize=11)
    ax.set_xticks(pace_index.index)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "car_development_pace_gap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SUCCESS] Exported Car Development Pace chart to {out_path}")


def plot_driver_efficiency_profile(df: pd.DataFrame):
    """
    Analyzes driver performance profiles: Average Grid vs. Average Finish Position.
    Drivers below the diagonal line gain positions on race day (race craft efficiency).
    """
    driver_stats = df.groupby("driver_clean").agg(
        Starts=("race_id", "count"),
        Avg_Grid=("grid_position", "mean"),
        Avg_Finish=("finish_position", "mean"),
        Points=("points", "sum")
    ).reset_index()

    # Filter drivers with at least 30 starts
    driver_stats = driver_stats[driver_stats["Starts"] >= 30]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    # Scatter plot
    scatter = ax.scatter(
        driver_stats["Avg_Grid"],
        driver_stats["Avg_Finish"],
        s=driver_stats["Points"] / 4.0,
        c=driver_stats["Points"],
        cmap="turbo",
        alpha=0.85,
        edgecolors="black",
        linewidths=1.2
    )

    # Diagonal reference line: finish == grid
    max_val = max(driver_stats["Avg_Grid"].max(), driver_stats["Avg_Finish"].max()) + 1.0
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.6, label="Grid = Finish Line")

    ax.set_title("Driver Performance Profiling: Qualifying vs. Race Finish (2021-2026)\n[Below dashed line = Net Position Gainers; Bubble size = Total Points]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Average Starting Grid Position (Lower = Better)", fontsize=11)
    ax.set_ylabel("Average Race Finish Position (Lower = Better)", fontsize=11)

    # Annotate top drivers
    for _, row in driver_stats.iterrows():
        if row["Points"] > 350:
            ax.annotate(
                row["driver_clean"],
                (row["Avg_Grid"], row["Avg_Finish"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Championship Points Scored", fontsize=10)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "driver_performance_profile.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SUCCESS] Exported Driver Performance Profile to {out_path}")


def generate_all_visualizations():
    df = load_f1_data()
    plot_driver_wins_and_podiums(df)
    plot_constructor_dominance_shift(df)
    plot_car_development_pace_gap(df)
    plot_driver_efficiency_profile(df)


if __name__ == "__main__":
    generate_all_visualizations()
