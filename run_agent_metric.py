# run_agent_metric.py
# Applies our performance metric to the HumanRuleAgent (agents.py).
# Put this file in the SAME folder as fishing.py, agents.py and the three log files.
# Run it with:  python run_agent_metric.py

import numpy as np
from fishing import FishingBay, load_log
from agents import Agent, RandomAgent, HumanRuleAgent, play

N_BAYS = 1000   # number of random bays per season


# ---------------------------------------------------------------------
# The metric (same as task3.py)
# ---------------------------------------------------------------------
def true_means_by_day(bay, n_days):
    info = bay.reveal()
    before = info["means"]
    after = info.get("means_after_change")     # only in Season 3
    shift = info.get("change_day")
    return [after if (after is not None and t >= shift) else before
            for t in range(1, n_days + 1)]


def metrics(bay, spots, catches):
    mu = true_means_by_day(bay, len(spots))
    expected = sum(mu[t][a] for t, a in enumerate(spots))     # Expected catch (E[Catch])
    v_star = sum(max(mu[t]) for t in range(len(spots)))        # Best possible catch (V*)
    actual = sum(catches)                                      # Actual catch
    return {"actual": actual, "E": expected, "luck": actual - expected,
            "V*": v_star, "eta": expected / v_star, "R_T": v_star - expected}


def play_and_measure(agent, bay):
    """Play one season with play() from agents.py, then compute the metric."""
    play(agent, bay)
    spots = [s for s, c in bay.history]
    catches = [c for s, c in bay.history]
    return metrics(bay, spots, catches)


# ---------------------------------------------------------------------
# A simple baseline to compare with: Explore then Stay
# ---------------------------------------------------------------------
class ExploreThenStayAgent(Agent):
    """Try each spot once, then always fish the spot with the best average so far."""
    def reset(self):
        self.n = [0] * self.n_spots
        self.s = [0.0] * self.n_spots

    def choose(self):
        untried = [k for k in range(self.n_spots) if self.n[k] == 0]
        if untried:
            return untried[0]
        return max(range(self.n_spots), key=lambda k: self.s[k] / self.n[k])

    def update(self, spot, catch):
        self.n[spot] += 1
        self.s[spot] += catch


def make_agents(bay, seed):
    """New agents for one bay. HumanRuleAgent needs the season length (horizon)."""
    return {
        "Random": RandomAgent(seed=seed),
        "Explore then Stay": ExploreThenStayAgent(seed=seed),
        "Human Rule": HumanRuleAgent(horizon=bay.days, seed=seed),
    }


COLS = ["Actual (kg)", "Expected (kg)", "Luck (kg)", "Best Possible (kg)",
        "Efficiency (%)", "Pseudo-Regret (kg)"]
KEYS = ["actual", "E", "luck", "V*", "eta", "R_T"]


def fmt(key, value):
    return f"{value * 100:.2f}" if key == "eta" else f"{value:.1f}"


# ---------------------------------------------------------------------
# PART A: every agent on the same 1000 random bays per season
# ---------------------------------------------------------------------
print("=" * 100)
print(f"PART A: average over {N_BAYS} random bays (seeds 0-{N_BAYS - 1}); in brackets: standard deviation")
print("=" * 100)
summary = {}
for season in (1, 2, 3):
    rows = {name: [] for name in make_agents(FishingBay(season=season, seed=0), 0)}
    for seed in range(N_BAYS):
        bay = FishingBay(season=season, seed=seed)
        for name, agent in make_agents(bay, seed).items():
            rows[name].append(play_and_measure(agent, bay))
    days = FishingBay(season=season, seed=0).days
    print(f"\nSeason {season} ({days} days)")
    print(f"{'Agent':18s}" + "".join(f"{c:>22s}" for c in COLS))
    for name, ms in rows.items():
        arr = {k: np.array([m[k] for m in ms]) for k in KEYS}
        summary[(season, name)] = arr
        cells = [f"{fmt(k, arr[k].mean())} ({fmt(k, arr[k].std())})" for k in KEYS]
        print(f"{name:18s}" + "".join(f"{c:>22s}" for c in cells))
    h, e = summary[(season, "Human Rule")]["eta"], summary[(season, "Explore then Stay")]["eta"]
    r = summary[(season, "Random")]["eta"]
    print(f"  Human Rule has a higher efficiency than Explore then Stay in {np.mean(h > e) * 100:.1f}% of bays, "
          f"and than Random in {np.mean(h > r) * 100:.1f}% of bays.")


# ---------------------------------------------------------------------
# PART B: the Human Rule agent on our own team's bays
# ---------------------------------------------------------------------
LOG_FILES = {
    "Player 1": "fishing_log_ioni.json",
    "Player 2": "fishing_log_szabo.json",
    "Player 3": "fishing_log_kimj.json",
}
print("\n" + "=" * 100)
print("PART B: Human Rule agent on each player's own bay (rebuilt from the log), compared with the player")
print("=" * 100)
print(f"{'Player':9s} {'Round':>5s} {'Days':>4s} {'Player Eff. (%)':>16s} {'Agent Actual (kg)':>18s} "
      f"{'Agent Expected (kg)':>20s} {'Agent Luck (kg)':>16s} {'Best Possible (kg)':>19s} "
      f"{'Agent Eff. (%)':>15s} {'Agent Pseudo-Regret (kg)':>25s}")
for name, path in LOG_FILES.items():
    log = load_log(path)
    for season in (1, 2, 3):
        bay = FishingBay.from_log(log, season)
        # the player's own efficiency, from the log
        from fishing import choices
        spots, catches = choices(log, season)
        player_eta = metrics(bay, spots, catches)["eta"]
        # the agent on the same bay (the agent's random order uses seed 0)
        m = play_and_measure(HumanRuleAgent(horizon=bay.days, seed=0), bay)
        print(f"{name:9s} {season:5d} {bay.days:4d} {player_eta * 100:16.2f} {m['actual']:18.1f} "
              f"{m['E']:20.1f} {m['luck']:16.1f} {m['V*']:19.1f} {m['eta'] * 100:15.2f} {m['R_T']:25.1f}")
print("\nNote: on the same bay the catch table is fixed, so the agent and the player face the same sea.")
