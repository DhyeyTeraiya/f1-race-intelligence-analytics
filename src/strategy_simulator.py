"""
Formula 1 Race Strategy & Undercut Simulator
Models pit stop deltas, in-lap/out-lap timing, tire wear cross-over points,
and calculates probability of a successful undercut or overcut.
"""

import numpy as np


class StrategySimulator:
    def __init__(self, pit_lane_loss_sec: float = 21.5, avg_pit_stop_sec: float = 2.6):
        self.pit_lane_loss = pit_lane_loss_sec
        self.avg_pit_stop = avg_pit_stop_sec
        self.total_pit_cost = self.pit_lane_loss + self.avg_pit_stop

    def simulate_undercut(
        self,
        gap_before_pit_sec: float,
        chaser_pit_lap: int,
        leader_pit_lap: int,
        chaser_compound: str = "HARD",
        leader_old_compound: str = "MEDIUM",
        laps_on_leader_tire: int = 24
    ) -> dict:
        """
        Simulate whether the trailing car (Chaser) can jump the leading car (Leader)
        by pitting earlier (the Undercut).
        """
        lap_delta = leader_pit_lap - chaser_pit_lap
        if lap_delta <= 0:
            return {"success": False, "reason": "Chaser must pit before Leader for undercut"}

        # Fresh tire pace advantage on out-lap (seconds per lap)
        out_lap_advantage = {
            "SOFT": 1.45,
            "MEDIUM": 1.15,
            "HARD": 0.85
        }.get(chaser_compound, 0.95)

        # Old tire degradation penalty on leader staying out (seconds per lap)
        deg_penalty_per_lap = 0.065 * (laps_on_leader_tire / 15.0)

        total_pace_delta = 0.0
        lap_breakdown = []

        for lap in range(1, lap_delta + 1):
            effective_delta = out_lap_advantage + (deg_penalty_per_lap * lap)
            total_pace_delta += effective_delta
            lap_breakdown.append({
                "lap_offset": lap,
                "fresh_tire_advantage": round(out_lap_advantage, 3),
                "leader_deg_penalty": round(deg_penalty_per_lap * lap, 3),
                "cumulative_delta": round(total_pace_delta, 3)
            })

        # Net gap when leader emerges after their pit stop
        # Chaser was gap_before_pit behind.
        # Chaser gained total_pace_delta during the undercut window.
        # Chaser spent pit stop time, then leader spent pit stop time (net 0 pit time delta assuming equal stops).
        net_margin = total_pace_delta - gap_before_pit_sec
        success = net_margin > 0.0

        return {
            "success": success,
            "gap_before_pit": gap_before_pit_sec,
            "undercut_window_laps": lap_delta,
            "total_pace_gained_sec": round(total_pace_delta, 3),
            "net_margin_sec": round(net_margin, 3),
            "recommendation": "EXECUTE UNDERCUT" if success else "DEFEND / EXTEND STINT (OVERCUT)",
            "lap_breakdown": lap_breakdown
        }


if __name__ == "__main__":
    sim = StrategySimulator()
    result = sim.simulate_undercut(gap_before_pit_sec=1.8, chaser_pit_lap=22, leader_pit_lap=24)
    print("Strategy Simulator Test Result:")
    print(f"  • Undercut Success: {result['success']}")
    print(f"  • Net Margin: {result['net_margin_sec']}s")
    print(f"  • Strategy Call: {result['recommendation']}")
