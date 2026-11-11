"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np

from fishing import ContextualBay, FishingBay

# ---------------------------------------------------------------------------
# Provided: do not modify
# ---------------------------------------------------------------------------


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------


class HumanRuleAgent(Agent):
    """Task 4: the best human strategy identified in Part II, written as a rule.

    Describe the rule here, precisely enough that someone could apply it by hand.
    Add constructor parameters if your rule has any (call super().__init__ first).
    
    Rules:
    1. Explore: visit every spot k times (random order).
    2. Estimate: the mean of a spot = average of its last w catches.
    3. Prune: a spot whose estmate is below (best estimate - delta) is not a candidate when switching.
    4. Stay: keep fishing the current spot while today's catch >= ref - theta, where ref = average of the catches made at this spot since arrival. 
        (before arrival, its estimate, for the first day)
    5. Patience: a day with catch - ref - theta is a drop. Leave only after more than p consecutive drops (p=0 -> leave at first drop).
    6. Switch: go to the non-pruned spot with the highest estimate if that estimate is above the ref of the spot being left; otherwise re-sample the least recently visited spot (any spot, even pruned one).
    7. Commit: if the best spot has at least m visits and its estimate exceeds every other estimate by more than commit_margin, stay there until the end (m=None disables it).
    8. End game: in the last e days do not explore, fish the spot with the best estimate. (We need to know the horizon = the number of days of the season)
    
    Parameters: 
    k, w, delta, theta, p, commit_margin, m, e
    horizon: int or None -> number of days of the season (provided by the player)

    Ex:
    bay = FishingBay(season=1, seed=0)
    total = play(HumanRuleAgent(horizon=bay.days, seed=0), bay)

    """

    def __init__(self, k=1, w=3, delta=4.0, theta=2.5, p=1, commit_margin=2.5, m=None, e=3, horizon=None, n_spots=5, seed=None):
        super().__init__(n_spots=n_spots, seed=seed)
        self.k = k
        self.w = w
        self.delta = delta
        self.theta = theta
        self.p = p
        self.commit_margin = commit_margin
        self.m = m
        self.e = e
        self.horizon = horizon

    def reset(self):
        # raise NotImplementedError("Task 4")
        n = self.n_spots
        self.hist = [[] for _ in range(n)] # catches per spot
        self.last_visit = [-1] * n  # day index of the last visit
        self.t = 0  # days already played
        self.order = [int(s) for _ in range(self.k) for s in self.rng.permutation(n)]
        self.current = None # spot of the current stay
        self.stay = []  # catches since arrival
        self.ref0 = 0.0 # reference of the first day of stay
        self.drops = 0  # consecutive bad days
        self.commited = False
        self.next = self.order[0]   # spot for tomorrow

    def _est(self, i):
        h = self.hist[i]
        return float(np.mean(h[-self.w:])) if h else -np.inf

    def _arrive(self, spot):
        self.current, self.stay, self.drops = spot, [], 0
        self.ref0 = self._est(spot)

    def _best(self):
        return int(np.argmax([self._est(i) for i in range(self.n_spots)]))

    def choose(self):
        # raise NotImplementedError("Task 4")
        return self.next

    def update(self, spot, catch):
        # raise NotImplementedError("Task 4")
        self.hist[spot].append(catch)
        self.last_visit[spot] = self.t
        self.t += 1
        n_explore = self.k * self.n_spots

        if self.t < n_explore:  # rule 1 -> explore
            self.next = self.order[self.t]
            return

        if self.t == n_explore: # end of exploration
            self._arrive(self._best())
            self.next = self.current
            return

        # exploitation -> spot == self.current
        ref = float(np.mean(self.stay)) if self.stay else self.ref0
        self.drops = self.drops + 1 if catch < ref - self.theta else 0
        self.stay.append(catch)
        est = [self._est(i) for i in range(self.n_spots)]

        # rule 8 -> end game, no exploring
        if self.horizon is not None and self.horizon - self.t <= self.e:
            b = self._best()
            if b != self.current:
                self._arrive(b)
            self.next = self.current
            return

        # rule 7 -> commit
        if self.m is not None and not self.commited:
            b = self._best()
            others = max(est[i] for i in range(self.n_spots) if i != b)
            if len(self.hist[b]) >= self.m and est[b] - others > self.commit_margin:
                self.commited = True
                if b != self.current:
                    self._arrive(b)

        if self.commited:
            self.next = self.current
            return

        # rule 4-5 -> stay unless more than p consecutive drops
        if self.drops <= self.p:
            self.next = self.current
            return

        # rule 6 -> switch
        ref_leave = float(np.mean(self.stay))
        best_est = max(est)
        cands = [i for i in range(self.n_spots) if i != self.current and est[i] >= best_est - self.delta] # rule 3
        target = max(cands, key = lambda i: est[i]) if cands else None

        if target is None or est[target] <= ref_leave:
            others = [i for i in range(self.n_spots) if i != self.current]
            oldest = min(self.last_visit[i] for i in others)
            target = int(self.rng.choice([i for i in others if self.last_visit[i] == oldest]))

        self._arrive(target)
        self.next = target   

# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.


# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.

