"""
Formula 1 Visual Analytics & Comparative Performance Engine
Generates publication-quality charts for:
1. Driver Wins & Podiums Evolution (2021-2026)
2. Constructor Dominance & Championship Share Shift (2021-2026)
3. Car Development Trajectory & Pace Gap Analysis ("Who Has the Better Car?")
4. Driver Performance & Race Craft Profiling (Grid vs Finish Conversion)
Author: Dhyey Teraiya (Data Scientist)
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
    "Alfa Romeo": "#900000",
    "AlphaTauri": "#5E8FAA",
    "RB": "#6692FF",
    "Racing Bulls": "#6692FF"
}

DRIVER_PRIMARY_TEAM = {
    "Max Verstappen": "Red Bull",
    "Lewis Hamilton": "Mercedes",
    "Lando Norris": "McLaren",
    "Charles Leclerc": "Ferrari",
    "Oscar Piastri": "McLaren",
    "Carlos Sainz": "Ferrari",
    "Sergio Perez": "Red Bull",
    "George Russell": "Mercedes",
    "Kimi Antonelli": "Mercedes",
    "Fernando Alonso": "Aston Martin",
    "Valtteri Bottas": "Mercedes",
    "Pierre Gasly": "Alpine",
    "Esteban Ocon": "Alpine"
}


def load_f1_data():
    races_csv = os.path.join(RAW_DIR, "f1_races.csv")
    df = pd.read_csv(races_csv, encoding="utf-8")
    df["driver_clean"] = df["driver_name"].astype(str).str.strip()
    return df


def plot_driver_wins_and_podiums(df: pd.DataFrame):
    """
    Plots real driver wins and podium finishes across the 2021-2026 era.
    Features completely clean driver names and team color styling.
    """
    driver_stats = df.groupby("driver_clean").agg(
        Starts=("race_id", "count"),
        Wins=("race_winner", "sum"),
        Podiums=("podium_finish", "sum"),
        Points=("points", "sum")
    ).sort_values(by=["Wins", "Podiums"], ascending=False).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)

    # Plot 1: Total Grand Prix Wins
    y_pos = np.arange(len(driver_stats))
    colors_wins = [TEAM_COLORS.get(DRIVER_PRIMARY_TEAM.get(name, ""), "#E10600") for name in driver_stats.index[::-1]]
    bars1 = ax1.barh(y_pos, driver_stats["Wins"][::-1], color=colors_wins, alpha=0.9, edgecolor="#222222", height=0.68)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(driver_stats.index[::-1], fontsize=11, fontweight="bold")
    ax1.set_title("Total Grand Prix Victories by Driver (2021–2026)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Race Wins", fontsize=11)
    ax1.set_xlim(0, max(driver_stats["Wins"]) * 1.15)
    for bar in bars1:
        w = bar.get_width()
        if w > 0:
            ax1.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{int(w)} wins", va="center", fontsize=10, fontweight="bold")

    # Plot 2: Total Podium Finishes (Top 3)
    podium_stats = df.groupby("driver_clean").agg(
        Podiums=("podium_finish", "sum"),
        Points=("points", "sum")
    ).sort_values(by="Podiums", ascending=False).head(10)

    colors_pods = [TEAM_COLORS.get(DRIVER_PRIMARY_TEAM.get(name, ""), "#1E41FF") for name in podium_stats.index[::-1]]
    bars2 = ax2.barh(np.arange(len(podium_stats)), podium_stats["Podiums"][::-1], color=colors_pods, alpha=0.9, edgecolor="#222222", height=0.68)
    ax2.set_yticks(np.arange(len(podium_stats)))
    ax2.set_yticklabels(podium_stats.index[::-1], fontsize=11, fontweight="bold")
    ax2.set_title("Total Podium Finishes (P1–P3) by Driver (2021–2026)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Podiums", fontsize=11)
    ax2.set_xlim(0, max(podium_stats["Podiums"]) * 1.15)
    for bar in bars2:
        w = bar.get_width()
        if w > 0:
            ax2.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{int(w)} pods", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "driver_wins_podiums_evolution.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Exported Driver Wins & Podiums chart to {out_path}")


def plot_constructor_dominance_shift(df: pd.DataFrame):
    """
    Plots real constructor points and win evolution across seasons 2021 through 2026.
    Highlights the era shift from Red Bull dominance (2022-2023) to McLaren (2024-2025)
    and Mercedes/Ferrari resurgence (2026).
    """
    top_teams = ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Aston Martin"]
    df_top = df[df["constructor"].isin(top_teams)]

    yearly_pts = df_top.groupby(["season", "constructor"])["points"].sum().unstack(fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)

    # Plot 1: Points Trend Lines
    for team in top_teams:
        if team in yearly_pts.columns:
            ax1.plot(yearly_pts.index, yearly_pts[team], marker="o", lw=3.2, label=team,
                     color=TEAM_COLORS.get(team, "#333333"), markersize=8)

    ax1.set_title("Constructor Championship Points Evolution (2021–2026 Seasons)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Championship Season", fontsize=11)
    ax1.set_ylabel("Championship Points Scored", fontsize=11)
    ax1.set_xticks(yearly_pts.index)
    ax1.legend(frameon=True, fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Team Win Share by Season
    yearly_wins = df.groupby(["season", "constructor"])["race_winner"].sum().unstack(fill_value=0)
    yearly_win_pct = yearly_wins.div(yearly_wins.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(yearly_win_pct))
    for team in ["Red Bull", "Mercedes", "McLaren", "Ferrari", "Alpine"]:
        if team in yearly_win_pct.columns:
            vals = yearly_win_pct[team].values
            ax2.bar(yearly_win_pct.index, vals, bottom=bottom, label=team,
                    color=TEAM_COLORS.get(team, "#777777"), alpha=0.88, edgecolor="white", width=0.55)
            bottom += vals

    ax2.set_title("Grand Prix Win Share (%) by Constructor (2021–2026)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Championship Season", fontsize=11)
    ax2.set_ylabel("Share of Total Wins (%)", fontsize=11)
    ax2.set_xticks(yearly_win_pct.index)
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=True, fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "constructor_dominance_shift.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Exported Constructor Dominance Shift chart to {out_path}")


def plot_car_development_pace_gap(df: pd.DataFrame):
    """
    Evaluates 'Who has the better car / development?' across 2021-2026:
    Panel 1: Average Starting Grid Rank by Constructor (lower = faster qualifying car)
    Panel 2: Race Day Position Delta (Grid vs Finish Delta = race pace & tire degradation advantage)
    """
    top_teams = ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Aston Martin"]
    df_top = df[df["constructor"].isin(top_teams)].copy()

    # Panel 1: Qualifying Grid Rank by Season
    pace_index = df_top.groupby(["season", "constructor"])["grid_position"].mean().unstack(fill_value=15.0)

    # Panel 2: Race Day Position Delta (grid_position - finish_position: >0 means gained positions on race day)
    df_top["position_gain"] = df_top["grid_position"] - df_top["finish_position"]
    delta_index = df_top.groupby(["season", "constructor"])["position_gain"].mean().unstack(fill_value=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)

    # Left: Average Grid Rank (inverted axis: P1 is at top)
    for team in top_teams:
        if team in pace_index.columns:
            ax1.plot(pace_index.index, pace_index[team], marker="s", lw=3.0, label=team,
                     color=TEAM_COLORS.get(team, "#333333"), markersize=8)

    ax1.invert_yaxis()
    ax1.set_title("Single-Lap Qualifying Pace Index by Team (2021–2026)\n[P1 Top = Faster Aerodynamic & Engine Package]", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Championship Season", fontsize=11)
    ax1.set_ylabel("Average Starting Grid Position (P1 is top)", fontsize=11)
    ax1.set_xticks(pace_index.index)
    ax1.legend(frameon=True, fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Right: Race Day Position Delta (Race Craft & Degradation Pace)
    for team in top_teams:
        if team in delta_index.columns:
            ax2.plot(delta_index.index, delta_index[team], marker="^", lw=2.8, label=team,
                     color=TEAM_COLORS.get(team, "#333333"), markersize=8)

    ax2.axhline(0, color="black", linestyle="--", alpha=0.6, lw=1.2)
    ax2.set_title("Race Day Position Delta by Team (2021–2026)\n[Above 0 = Gained Positions on Sunday via Race Pace & Strategy]", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Championship Season", fontsize=11)
    ax2.set_ylabel("Average Net Positions Gained per Race", fontsize=11)
    ax2.set_xticks(delta_index.index)
    ax2.legend(frameon=True, fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "car_development_pace_gap.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Exported Car Development Pace chart to {out_path}")


def plot_driver_efficiency_profile(df: pd.DataFrame):
    """
    Analyzes driver performance profiles: Average Grid vs. Average Finish Position.
    Drivers below the diagonal line gain positions on race day (race craft efficiency).
    Driver names are completely clean with no formatting errors.
    """
    driver_stats = df.groupby("driver_clean").agg(
        Starts=("race_id", "count"),
        Avg_Grid=("grid_position", "mean"),
        Avg_Finish=("finish_position", "mean"),
        Points=("points", "sum")
    ).reset_index()

    # Filter drivers with at least 25 starts
    driver_stats = driver_stats[driver_stats["Starts"] >= 25]

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=150)

    # Scatter plot with bubble size proportional to points
    scatter = ax.scatter(
        driver_stats["Avg_Grid"],
        driver_stats["Avg_Finish"],
        s=(driver_stats["Points"] / 4.5).clip(lower=40, upper=550),
        c=driver_stats["Points"],
        cmap="viridis",
        alpha=0.88,
        edgecolors="black",
        linewidths=1.2
    )

    # Diagonal reference line: finish == grid
    max_val = max(driver_stats["Avg_Grid"].max(), driver_stats["Avg_Finish"].max()) + 1.0
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.7, lw=1.5, label="Parity Line (Grid = Finish)")

    ax.set_title("Driver Race Craft Profiling: Qualifying vs. Race Finish (2021–2026)\n[Below red dashed line = Net Position Gainers; Bubble size = Total Championship Points]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Average Starting Grid Position (Lower = Better)", fontsize=11)
    ax.set_ylabel("Average Race Finish Position (Lower = Better)", fontsize=11)

    # Annotate top drivers with clean names and custom non-overlapping offsets
    custom_offsets = {
        "Max Verstappen": (12, -5),
        "Lewis Hamilton": (12, -8),
        "Lando Norris": (-85, 6),
        "Charles Leclerc": (-95, -12),
        "Oscar Piastri": (-85, 12),
        "George Russell": (12, -6),
        "Kimi Antonelli": (12, 12),
        "Carlos Sainz": (12, 8),
        "Sergio Perez": (12, -10),
        "Fernando Alonso": (12, -4),
    }

    for _, row in driver_stats.iterrows():
        d_name = row["driver_clean"]
        if row["Points"] > 300 or d_name in ["Carlos Sainz", "Kimi Antonelli", "Fernando Alonso"]:
            offset = custom_offsets.get(d_name, (8, 4))
            ax.annotate(
                d_name,
                (row["Avg_Grid"], row["Avg_Finish"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Championship Points Scored", fontsize=10)
    ax.legend(frameon=True, loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "driver_performance_profile.png")
    plt.savefig(out_path, bbox_inches="tight")
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
