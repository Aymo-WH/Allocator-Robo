"""
tiers.py — turn each risk tier into a single daily return stream.

Within a tier we do two rules-based things (NO prediction):
  1. Inverse-vol weighting across the tier's candidates ("risk-parity-lite"):
     calmer names get a bigger share so no single holding dominates tier risk.
  2. The vol overlay sizes each position toward the tier's target volatility.

The tier return is the weighted blend. This is the "smart enough to know what to
buy" piece, done honestly: it weights and rebalances a basket you approve, it
does not forecast which candidate will win.
"""
import numpy as np
import pandas as pd

from vol_overlay import ewma_vol, overlay_weight


def inverse_vol_weights(returns_df, halflife):
    """
    Cross-sectional inverse-vol weights (one row per date, columns = candidates).
    Each candidate's weight ~ 1 / its recent vol, renormalized to sum to 1.
    Uses only past info (EWMA vol shifted one bar).
    """
    vol = returns_df.apply(lambda c: ewma_vol(c, halflife)).shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.fillna(0.0)


def tier_returns(prices, candidates, target_vol_annual, halflife=20, vol_cap=1.0):
    """
    Build the daily return series for one tier.

    prices : DataFrame of adjusted close (all tickers); we use `candidates`.
    Returns (tier_ret, avg_exposure) where tier_ret is a daily return Series and
    avg_exposure is the mean total long exposure (a diagnostic; <1 means the
    overlay is holding cash on average).
    """
    have = [c for c in candidates if c in prices.columns]
    if not have:
        raise ValueError(f"None of the tier candidates {candidates} are in the price data.")

    rets = prices[have].pct_change()

    # (1) how to split the tier across its names
    cross_w = inverse_vol_weights(rets, halflife)

    # (2) how much of each name to actually hold (vol targeting per position)
    pos_w = pd.DataFrame(
        {c: overlay_weight(rets[c], target_vol_annual, halflife, vol_cap) for c in have},
        index=rets.index,
    )

    # effective weight on each candidate = split * sizing
    eff = cross_w[have] * pos_w[have]
    tier_ret = (eff * rets[have]).sum(axis=1)
    avg_exposure = float(eff.sum(axis=1).mean())
    return tier_ret.fillna(0.0), avg_exposure


def latest_target_weights(prices, cfg):
    """
    Today's target weight for each PRODUCT as a fraction of the whole pool, using
    the same math as the backtest evaluated at the last available date:
        weight = policy_weight(tier) x inverse_vol_split x vol_overlay_sizing
    A product appearing in several tiers gets its weights summed (you hold one
    line of it). Weights sum to <= 1.0; the remainder is the intended cash buffer
    (the overlay deliberately holds cash when forecast vol runs hot).

    Returns (weights: dict product->weight, cash_weight: float).
    """
    ov = cfg["vol_overlay"]
    hl, cap = ov["halflife_days"], ov["vol_cap"]
    weights = {}
    for tier in cfg["tiers"]:
        have = [c for c in tier["candidates"] if c in prices.columns]
        if not have:
            continue
        rets = prices[have].pct_change()
        cross = inverse_vol_weights(rets, hl).iloc[-1]
        for c in have:
            pos = overlay_weight(rets[c], tier["target_vol_annual"], hl, cap).iloc[-1]
            weights[c] = weights.get(c, 0.0) + float(tier["policy_weight"]) * float(cross[c]) * float(pos)
    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    return weights, cash_weight


def build_all_tiers(prices, cfg):
    """
    Returns a DataFrame (index = dates, columns = tier names) of per-tier daily
    returns, plus a dict of average exposures keyed by tier name.
    """
    ov = cfg["vol_overlay"]
    out = {}
    expo = {}
    for tier in cfg["tiers"]:
        r, e = tier_returns(
            prices,
            tier["candidates"],
            tier["target_vol_annual"],
            halflife=ov["halflife_days"],
            vol_cap=ov["vol_cap"],
        )
        out[tier["name"]] = r
        expo[tier["name"]] = e
    return pd.DataFrame(out).fillna(0.0), expo
