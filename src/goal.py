"""
goal.py — "am I on track?" Forward Monte Carlo projection. CURRENCY-AWARE.

Two honesty fixes over the naive version:

1. DRIFT IS AN ASSUMPTION, NOT A SAMPLE STAT.
   The allocator's own return history is short and mostly bull (2021-2026). Its
   *mean* return is therefore an unreliable, over-optimistic estimate of the
   future — bootstrapping it raw gave a fake "100% chance of success". Its
   *volatility/clustering*, by contrast, is far more stable and worth keeping.
   So we KEEP the empirical volatility (block bootstrap) but RE-CENTER the drift
   to conservative planning assumptions you set in config, and report a
   conservative / base / optimistic spread instead of one false-precise number.

2. FX RISK IS REAL.
   You fund in MYR and your goal is in MYR, but the assets are USD. A USD pool
   converted back to MYR carries USD/MYR risk. We model it explicitly: bootstrap
   real USD/MYR returns for realistic FX volatility, but neutralize the FX drift
   by default (no house view that the ringgit weakens or strengthens).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
CONTRIB_EVERY = 21


def _recenter(returns, target_annual_drift):
    """Shift a daily return series so its annualized mean equals the target,
    preserving shape, volatility, and (via later block sampling) autocorrelation."""
    r = np.asarray(returns.dropna(), dtype=float)
    daily_target = (1.0 + target_annual_drift) ** (1.0 / TRADING_DAYS) - 1.0
    return r - r.mean() + daily_target


def _block_paths(r, horizon_days, n_paths, block, rng):
    """(n_paths x horizon_days) of stitched contiguous blocks (preserves vol clustering)."""
    if len(r) < block:
        raise ValueError("Not enough return history for the chosen block size.")
    n_blocks = int(np.ceil(horizon_days / block))
    max_start = len(r) - block
    out = np.empty((n_paths, n_blocks * block))
    for p in range(n_paths):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        out[p] = np.concatenate([r[s:s + block] for s in starts])
    return out[:, :horizon_days]


def _simulate_scenario(asset_r, mu_annual, cfg, current_value,
                       fx_r=None, fx_drift_annual=0.0, n_paths=5000, block=21, seed=0):
    goal = cfg["goal"]
    horizon_days = int(goal["horizon_years"] * TRADING_DAYS)
    monthly = float(goal["monthly_contribution"])
    rng = np.random.default_rng(seed)

    asset_c = _recenter(asset_r, mu_annual)
    asset_paths = _block_paths(asset_c, horizon_days, n_paths, block, rng)

    if fx_r is not None:
        fx_c = _recenter(fx_r, fx_drift_annual)        # FX vol kept, drift neutralized
        fx_paths = _block_paths(fx_c, horizon_days, n_paths, block, rng)
        combined = (1.0 + asset_paths) * (1.0 + fx_paths) - 1.0
    else:
        combined = asset_paths

    terminal = np.empty(n_paths)
    for p in range(n_paths):
        v = float(current_value)
        cp = combined[p]
        for d in range(horizon_days):
            v *= (1.0 + cp[d])
            if d > 0 and d % CONTRIB_EVERY == 0:
                v += monthly
        terminal[p] = v

    total_contributed = current_value + monthly * (horizon_days // CONTRIB_EVERY)
    target = float(goal["target_amount"])
    return {
        "p_hit_goal": float((terminal >= target).mean()),
        "median": float(np.median(terminal)),
        "p10": float(np.percentile(terminal, 10)),
        "p90": float(np.percentile(terminal, 90)),
        "total_contributed": total_contributed,
    }


def project_goal(asset_returns, cfg, current_value, fx_returns=None, n_paths=5000):
    """
    Returns dict keyed by scenario name (conservative/base/optimistic), each with
    P(hit goal) and the wealth distribution, all in the goal's base currency.
    Drift comes from config assumptions; volatility from the data. FX modeled if
    fx_returns is provided and projection.model_fx is true.
    """
    proj = cfg.get("projection", {})
    scenarios = proj.get("annual_return_scenarios",
                         {"conservative": 0.04, "base": 0.07, "optimistic": 0.10})
    model_fx = proj.get("model_fx", False)
    fx_drift = float(proj.get("fx_drift_annual", 0.0))
    fx = fx_returns if (model_fx and fx_returns is not None) else None

    out = {}
    for i, (name, mu) in enumerate(scenarios.items()):
        out[name] = _simulate_scenario(
            asset_returns, float(mu), cfg, current_value,
            fx_r=fx, fx_drift_annual=fx_drift, n_paths=n_paths, seed=i,
        )
    out["_fx_modeled"] = fx is not None
    return out
