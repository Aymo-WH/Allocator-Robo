"""
goal.py — "am I on track?" Forward Monte Carlo projection.

We do NOT predict returns. We resample the allocator's own historical daily
returns (block bootstrap, to preserve volatility clustering) forward over the
remaining horizon, adding the monthly contribution schedule, and report the
distribution of terminal wealth and the probability of hitting the goal.

This is honest planning, not forecasting: it answers "given how this strategy has
behaved and what I keep contributing, what's the range of outcomes?" — not "what
will the market do."
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
CONTRIB_EVERY = 21


def block_bootstrap_paths(daily_returns, horizon_days, n_paths, block=21, seed=0):
    """
    Generate (n_paths x horizon_days) of simulated daily returns by stitching
    random contiguous blocks from the historical series (preserves short-term
    autocorrelation / vol clustering that iid sampling would destroy).
    """
    r = np.asarray(daily_returns.dropna())
    if len(r) < block:
        raise ValueError("Not enough return history for the chosen block size.")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon_days / block))
    max_start = len(r) - block
    paths = np.empty((n_paths, n_blocks * block))
    for p in range(n_paths):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        chunk = np.concatenate([r[s:s + block] for s in starts])
        paths[p] = chunk
    return paths[:, :horizon_days]


def project_goal(daily_returns, cfg, current_value=None, n_paths=5000, seed=0):
    """
    Returns a dict summarizing P(hit goal) and the wealth distribution at horizon.
    current_value defaults to starting_capital (project from today / day 0).
    """
    goal = cfg["goal"]
    horizon_days = int(goal["horizon_years"] * TRADING_DAYS)
    start = float(current_value if current_value is not None else goal["starting_capital"])
    monthly = float(goal["monthly_contribution"])
    target = float(goal["target_amount"])

    paths = block_bootstrap_paths(daily_returns, horizon_days, n_paths, seed=seed)

    terminal = np.empty(n_paths)
    for p in range(n_paths):
        v = start
        for d in range(horizon_days):
            v *= (1.0 + paths[p, d])
            if d > 0 and d % CONTRIB_EVERY == 0:
                v += monthly
        terminal[p] = v

    total_contributed = start + monthly * (horizon_days // CONTRIB_EVERY)
    return {
        "target": target,
        "horizon_years": goal["horizon_years"],
        "total_contributed": total_contributed,
        "p_hit_goal": float((terminal >= target).mean()),
        "median": float(np.median(terminal)),
        "p10": float(np.percentile(terminal, 10)),
        "p25": float(np.percentile(terminal, 25)),
        "p75": float(np.percentile(terminal, 75)),
        "p90": float(np.percentile(terminal, 90)),
    }
