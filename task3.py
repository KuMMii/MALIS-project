# task3.py  (version 3: uses our team's performance metric)
# Put this file in the SAME folder as fishing.py and your three log files:
#   fishing_log_ioni.json, fishing_log_szabo.json, fishing_log_kimj.json
# Run it with:  python task3.py     (or python3 task3.py)

import random
import numpy as np
from fishing import FishingBay, load_log, choices

# =====================================================================
# PART 1. THE METRIC
# =====================================================================
# For one played season:
#   mu(k, t)      = true mean of spot k on day t (from reveal(), only after the season)
#   E[Catch]      = sum over days of mu(a_t, t)            -> decision quality (no noise)
#   Actual catch  = sum over days of r_t                    -> what the player really got
#   Luck L        = Actual catch - E[Catch]                 -> the noise eps_t added up
#   V*            = sum over days of max_k mu(k, t)         -> oracle: best spot every day
#   R_T           = V* - E[Catch]                           -> pseudo-regret (kg lost by choices)
#   eta           = E[Catch] / V* = 1 - R_T / V*            -> efficiency ratio in [0, 1]

def true_means_by_day(bay, n_days):
    """List with, for every day t = 1..T, the true means of the 5 spots on that day."""
    info = bay.reveal()
    before = info["means"]
    after = info.get("means_after_change")     # only exists in Season 3
    shift = info.get("change_day")             # Day 16 in Season 3
    return [after if (after is not None and t >= shift) else before
            for t in range(1, n_days + 1)]


def metrics(bay, spots, catches):
    """Compute E[Catch], actual catch, luck, V*, R_T and eta for one season."""
    mu = true_means_by_day(bay, len(spots))
    expected = sum(mu[t][a] for t, a in enumerate(spots))      # E[Catch]
    v_star = sum(max(mu[t]) for t in range(len(spots)))         # V*
    actual = sum(catches)
    return {
        "E": expected,
        "actual": actual,
        "luck": actual - expected,
        "V*": v_star,
        "R_T": v_star - expected,
        "eta": expected / v_star,
    }


# Quick check: our V* must equal the simulator's own best_possible()
_bay = FishingBay(season=3, seed=0)
while not _bay.done:
    _bay.fish(0)
assert abs(metrics(_bay, [s for s, c in _bay.history], [c for s, c in _bay.history])["V*"]
           - _bay.best_possible()) < 1e-6, "V* does not match best_possible()"


# =====================================================================
# PART 2. PROVE IT WITH THE SIMULATOR (same strategy, many random bays)
# =====================================================================

def random_strategy(bay, seed):
    """Every day, pick a spot completely at random."""
    rng = random.Random(seed)
    while not bay.done:
        bay.fish(rng.randrange(5))


def explore_then_stay(bay, seed):
    """Explore then Stay: try each spot once, then always fish the spot with the best average so far."""
    n = [0] * 5
    s = [0.0] * 5
    while not bay.done:
        untried = [k for k in range(5) if n[k] == 0]
        spot = untried[0] if untried else max(range(5), key=lambda k: s[k] / n[k])
        catch = bay.fish(spot)
        n[spot] += 1
        s[spot] += catch


def run_many(strategy, season, n_bays=1000):
    """Play the same strategy on n_bays random bays (seeds 0..n_bays-1)."""
    rows = []
    for seed in range(n_bays):
        bay = FishingBay(season=season, seed=seed)
        strategy(bay, seed)
        spots = [s for s, c in bay.history]
        catches = [c for s, c in bay.history]
        rows.append(metrics(bay, spots, catches))
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def separation(a, b):
    """How easy is it to tell two strategies apart? (difference of means / typical spread)"""
    return (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)


print("=" * 78)
print("PART 2: same strategy on 1000 random bays")
print("=" * 78)
results = {}
for season in (1, 2, 3):
    print(f"\nSeason {season}")
    print(f"{'Strategy':18s} {'Actual catch (kg)':>18s} {'Luck (L, kg)':>14s} {'Pseudo-regret (R_T, kg)':>24s} {'Efficiency (eta)':>17s}")
    print(f"{'':18s} {'mean     sd':>18s} {'mean    sd':>14s} {'mean     sd':>24s} {'mean      sd':>17s}")
    for name, strat in [("Random", random_strategy), ("Explore then Stay", explore_then_stay)]:
        r = run_many(strat, season)
        results[(season, name)] = r
        print(f"{name:18s} {r['actual'].mean():11.1f} {r['actual'].std():6.1f}  "
              f"{r['luck'].mean():8.1f} {r['luck'].std():5.1f}  "
              f"{r['R_T'].mean():17.1f} {r['R_T'].std():6.1f}  "
              f"{r['eta'].mean():10.3f} {r['eta'].std():6.3f}")
    rnd, ets = results[(season, "Random")], results[(season, "Explore then Stay")]
    n_days = FishingBay(season=season, seed=0).days
    print(f"  Explore then Stay catches FEWER kg than Random in {np.mean(ets['actual'] < rnd['actual']) * 100:5.1f}% of bays,")
    print(f"  but has a LOWER efficiency ratio (eta) than Random in only {np.mean(ets['eta'] < rnd['eta']) * 100:5.1f}% of bays.")
    print(f"  How clearly the two strategies are told apart (separation, d): "
          f"by actual catch {separation(ets['actual'], rnd['actual']):.2f}, "
          f"by efficiency ratio (eta) {separation(ets['eta'], rnd['eta']):.2f}")
    print(f"  Luck (L) of Explore then Stay: average {ets['luck'].mean():.2f} kg (close to 0), "
          f"typical size (standard deviation) {ets['luck'].std():.1f} kg")
    print(f"  Expected typical size of luck from daily noise over the season "
          f"(2.5 kg x square root of season length, 2.5*sqrt(T), T = {n_days} days) = {2.5 * np.sqrt(n_days):.1f} kg")

# Optional picture for the report (needs matplotlib; skipped if not installed)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = results[(1, "Explore then Stay")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].hist(r["actual"], bins=30, color="#4472C4")
    axes[0].set_title("Season 1: actual catch (kg)\nExplore then Stay, 1000 bays")
    axes[0].set_xlabel("kg")
    axes[1].hist(r["luck"], bins=30, color="#4472C4")
    axes[1].set_title("Season 1: luck (L)\n= actual catch - expected catch")
    axes[1].set_xlabel("kg")
    axes[2].hist(r["eta"], bins=30, color="#4472C4")
    axes[2].set_title("Season 1: efficiency ratio (eta)")
    axes[2].set_xlabel("efficiency ratio (eta)")
    plt.tight_layout()
    plt.savefig("task3_histograms.png", dpi=150)
    print("\nSaved the picture task3_histograms.png")
except ImportError:
    print("\n(matplotlib not installed, no picture made)")


# =====================================================================
# PART 3. YOUR TEAM
# =====================================================================
LOG_FILES = {
    "Player 1": "fishing_log_ioni.json",
    "Player 2": "fishing_log_szabo.json",
    "Player 3": "fishing_log_kimj.json",
}
logs = {name: load_log(path) for name, path in LOG_FILES.items()}

# One format for every row of the Part 3 tables
COLUMNS = ["Player", "Round", "Days", "Actual (kg)", "Expected (kg)", "Luck (kg)",
           "Best Possible (kg)", "Efficiency (%)", "Pseudo-Regret (kg)"]


def print_header(first_cols):
    print("  ".join(f"{c:>{w}s}" for c, w in first_cols) + "  " +
          "  ".join(f"{c:>{len(c)}s}" for c in COLUMNS[3:]))


def print_row(first_values, m, n_days_label):
    cells = [f"{v:>{w}}" for v, w in first_values]
    cells.append(f"{n_days_label:>4}")
    values = [m["actual"], m["E"], m["luck"], m["V*"], m["eta"] * 100, m["R_T"]]
    formats = [".1f", ".1f", ".1f", ".1f", ".2f", ".1f"]
    cells += [f"{v:{len(c)}{f}}" for v, c, f in zip(values, COLUMNS[3:], formats)]
    print("  ".join(cells))


def metrics_for_days(mu, spots, catches, days):
    """Same metric as metrics(), but only for some days (used to split Season 3)."""
    e = sum(mu[t - 1][spots[t - 1]] for t in days)
    v = sum(max(mu[t - 1]) for t in days)
    a = sum(catches[t - 1] for t in days)
    return {"E": e, "actual": a, "luck": a - e, "V*": v, "R_T": v - e, "eta": e / v}


print("\n" + "=" * 78)
print("PART 3: team table (each player's own bay rebuilt from the log)")
print("=" * 78)
print_header([("Player", 8), ("Round", 5), ("Days", 4)])
for name, log in logs.items():
    for season in (1, 2, 3):
        bay = FishingBay.from_log(log, season)       # the exact bay this person played
        spots, catches = choices(log, season)
        print_row([(name, 8), (season, 5)], metrics(bay, spots, catches), len(spots))


# ---- Season 3: what changed and who adapted ----
print("\n" + "=" * 78)
print("PART 3b: Season 3 (Round 3), before and after the change on Day 16")
print("=" * 78)
print("Table 3b-1: the same metric, split into the days before and after the change")
adapt = []
print_header([("Player", 8), ("Period", 11), ("Days", 4)])
for name, log in logs.items():
    bay = FishingBay.from_log(log, 3)
    spots, catches = choices(log, 3)
    info = bay.reveal()
    shift = info["change_day"]
    before, after = info["means"], info["means_after_change"]
    n_days = len(spots)
    mu = true_means_by_day(bay, n_days)
    for label, days in [(f"Days 1-{shift - 1}", range(1, shift)),
                        (f"Days {shift}-{n_days}", range(shift, n_days + 1))]:
        print_row([(name, 8), (label, 11)], metrics_for_days(mu, spots, catches, days), len(days))

    old_spot = spots[shift - 2]                        # spot used on Day 15
    new_best = after.index(max(after))                 # best spot after the change
    leave_day = next((d for d in range(shift, n_days + 1) if spots[d - 1] != old_spot), None)
    first_new = next((d for d in range(shift, n_days + 1) if spots[d - 1] == new_best), None)
    adapt.append((name, old_spot, before[old_spot], after[old_spot], leave_day, new_best,
                  first_new, spots[shift - 3:shift + 8], before, after))

print("\nTable 3b-2: how fast each player adapted to the change")
print(f"{'Player':>8s}  {'Spot on Day 15':>14s}  {'Its true mean (kg)':>18s}  {'Left on':>7s}  "
      f"{'Leaving delay (days)':>20s}  {'New best spot':>13s}  {'First visit':>11s}  {'Finding delay (days)':>20s}")
for name, old, mb, ma, leave, best, first, _, _, _ in adapt:
    print(f"{name:>8s}  {'Spot ' + str(old):>14s}  {f'{mb} -> {ma}':>18s}  "
          f"{('Day ' + str(leave)) if leave else 'never':>7s}  {(leave - shift) if leave else '-':>20}  "
          f"{'Spot ' + str(best):>13s}  {('Day ' + str(first)) if first else 'never':>11s}  "
          f"{(first - shift) if first else '-':>20}")

print("\nTable 3b-3: true mean catch of each spot (kg), and spots chosen on Days 14-24")
for name, _, _, _, _, _, _, chosen, before, after in adapt:
    print(f"{name}: before the change {before}, after the change {after}")
    print(f"{'':10s}spots chosen on Days 14-24: {chosen}")
print("\nNote: a player who switches spots every day has no real leaving day.")