"""
rebalance.py — the heart of the allocator (you flagged this as the critical part,
and you're right).

Simulates the managed pool in *dollar* terms across the tiers, applying:
  - monthly contributions, deployed to whichever tiers are UNDERWEIGHT
    (this doubles as free rebalancing — you buy what's lagging),
  - calendar rebalancing every `frequency_days`,
  - drift-band rebalancing whenever any tier strays more than `drift_band_pct`
    (relative) from its policy weight, whichever fires first.

Returns both a dollar value path (with contributions, for goal tracking) and a
clean time-weighted daily return series (contribution-neutral, for Sharpe/MaxDD).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
CONTRIB_EVERY = 21        # ~monthly, in trading days
REBAL_FEE = 0.0005        # 5 bps on turnover when we rebalance/deploy


def _deploy_contribution(values, policy, cash):
    """
    Add `cash` to the tiers, filling the gap toward policy weights first
    (buy the laggards). Returns the new values vector and the dollar turnover.
    """
    total_after = values.sum() + cash
    desired = policy * total_after
    gap = (desired - values).clip(min=0.0)   # how far each tier is below target
    if gap.sum() > 0:
        alloc = cash * gap / gap.sum()
    else:                                     # everyone at/above target -> by policy
        alloc = cash * policy
    new_values = values + alloc
    turnover = alloc.sum()                    # all of it is a buy
    return new_values, turnover


def _rebalance_to_policy(values, policy):
    """Reset to policy weights. Turnover = dollars that had to move."""
    total = values.sum()
    target = policy * total
    turnover = np.abs(target - values).sum() / 2.0   # buys == sells, count once
    return target, turnover


def simulate_pool(tier_rets, cfg, return_trades=False):
    """
    tier_rets : DataFrame (dates x tier names) of daily tier returns.
    Returns dict with:
      value      : Series of total pool value (USD, includes contributions)
      port_ret   : Series of contribution-neutral daily portfolio returns
      weights    : DataFrame of end-of-day tier weights
      contributed: total cash put in (starting capital + all contributions)
      fees_paid  : total rebalancing/turnover fees
    """
    goal = cfg["goal"]
    reb = cfg["rebalance"]
    tier_names = [t["name"] for t in cfg["tiers"]]
    policy = np.array([t["policy_weight"] for t in cfg["tiers"]], dtype=float)
    policy = policy / policy.sum()

    R = tier_rets[tier_names].copy()
    dates = R.index

    values = policy * float(goal["starting_capital"])
    contributed = float(goal["starting_capital"])
    fees_paid = 0.0
    monthly = float(goal["monthly_contribution"])
    band = float(reb["drift_band_pct"])
    freq = int(reb["frequency_days"])

    value_path = np.empty(len(dates))
    port_ret = np.empty(len(dates))
    weight_path = np.empty((len(dates), len(tier_names)))
    last_rebal = 0

    for i, dt in enumerate(dates):
        total = values.sum()
        w_start = values / total if total > 0 else policy
        r_today = R.iloc[i].values
        port_ret[i] = float(np.dot(w_start, r_today))

        # grow
        values = values * (1.0 + r_today)

        # monthly contribution (deploy to underweight tiers)
        if i > 0 and i % CONTRIB_EVERY == 0:
            values, turn = _deploy_contribution(values, policy, monthly)
            contributed += monthly
            fee = turn * REBAL_FEE
            values *= (1.0 - fee / values.sum()) if values.sum() > 0 else 1.0
            fees_paid += fee

        # rebalance: calendar OR drift band
        total = values.sum()
        w_now = values / total if total > 0 else policy
        rel_drift = np.abs(w_now - policy) / policy
        time_due = (i - last_rebal) >= freq
        drift_due = bool((rel_drift > band).any())
        if (time_due or drift_due) and total > 0:
            values, turn = _rebalance_to_policy(values, policy)
            fee = turn * REBAL_FEE
            values *= (1.0 - fee / values.sum())
            fees_paid += fee
            last_rebal = i

        value_path[i] = values.sum()
        weight_path[i] = values / values.sum() if values.sum() > 0 else policy

    out = {
        "value": pd.Series(value_path, index=dates),
        "port_ret": pd.Series(port_ret, index=dates),
        "weights": pd.DataFrame(weight_path, index=dates, columns=tier_names),
        "contributed": contributed,
        "fees_paid": fees_paid,
    }
    return out


def benchmark_dca(prices, cfg):
    """
    Apples-to-apples benchmark: dollar-cost-average the SAME cash schedule into a
    single buy-and-hold ticker (default SPY). Same money in, no tiering, no
    rebalancing, no vol overlay.
    """
    goal = cfg["goal"]
    tkr = cfg["benchmark"]["ticker"]
    px = prices[tkr].dropna()
    ret = px.pct_change().fillna(0.0)
    dates = px.index

    units = float(goal["starting_capital"]) / px.iloc[0]
    value_path = np.empty(len(dates))
    monthly = float(goal["monthly_contribution"])

    for i in range(len(dates)):
        if i > 0 and i % CONTRIB_EVERY == 0:
            units += monthly / px.iloc[i]
        value_path[i] = units * px.iloc[i]

    return pd.Series(value_path, index=dates), ret.reindex(dates).fillna(0.0)
