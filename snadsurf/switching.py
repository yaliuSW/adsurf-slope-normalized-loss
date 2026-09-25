"""Per-start, one-way loss switch using recent raw-data-loss progress.

This controller sees no models, roots, reference profiles, or candidate ranks.
It measures optimization progress, not validity of the local slope expansion.
"""
import numpy as np


class PlateauSwitch:
    def __init__(self, starts, window=20, tolerance=.01, patience=2, min_raw=80):
        if starts < 1 or window < 1 or patience < 1 or min_raw < 0:
            raise ValueError('Invalid counts in plateau switch')
        if not 0 < tolerance < 1:
            raise ValueError('Relative improvement tolerance must be in (0,1)')
        self.n = starts
        self.window = int(window)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.min_raw = int(min_raw)
        self.switched = np.zeros(starts, dtype=bool)
        self.switch_updates = np.full(starts, -1, dtype=int)
        self.streak = np.zeros(starts, dtype=int)
        self.regime = None
        self.last_update = -1
        self.history = []
        self.events = []
        self.checks = []

    def observe(self, update, raw, regime, allow_switch=True):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (self.n,) or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError('Expected one finite nonnegative data loss per start')
        if update != self.last_update + 1:
            raise ValueError('Observe every update exactly once, starting at zero')
        self.last_update = update
        regime = tuple(regime)
        if regime != self.regime:
            self.regime = regime
            self.regime_start = update
            self.history = []
            self.streak[:] = 0
        previous = self.history[-1] if self.history else raw
        best = np.where(self.switched, previous, np.minimum(previous, raw))
        self.history.append(best.copy())
        newly = np.zeros(self.n, dtype=bool)
        age = update - self.regime_start
        if age == 0 or age % self.window:
            return newly
        earlier = self.history[-self.window-1]
        improvement = (earlier-best) / np.maximum(earlier, 1e-30)
        stagnant = improvement < self.tolerance
        self.streak = np.where(~self.switched & stagnant, self.streak+1, 0)
        if allow_switch and update >= self.min_raw:
            newly = ~self.switched & (self.streak >= self.patience)
            self.switched |= newly
            self.switch_updates[newly] = update
        self.checks.append(dict(update=update, regime=list(regime),
                                improvement=improvement.tolist(),
                                streak=self.streak.tolist(),
                                allow_switch=bool(allow_switch)))
        for sid in np.flatnonzero(newly):
            self.events.append(dict(start=int(sid), update=int(update),
                                    recent_relative_improvement=float(improvement[sid]),
                                    best_raw=float(best[sid]), raw_at_switch=float(raw[sid]),
                                    regime=list(regime)))
        return newly

    def report(self):
        return dict(window=self.window, tolerance=self.tolerance,
                    patience=self.patience, minimum_raw_updates=self.min_raw,
                    switch_updates=self.switch_updates.tolist(),
                    events=self.events, checks=self.checks)
