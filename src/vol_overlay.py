"""
vol_overlay.py — the proven risk engine, carried over from the Vol-Gordian
result. Across 8 assets and every regime, sizing a long position inversely to
forecast volatility did NOT beat buy-and-hold on Sharpe, but it cut max drawdown
on 8/8 assets (~30-36% shallower). That drawdown control is exactly what a
goal-based allocator wants — it is the one empirically robust lever we found.

    w_t = clip( target_vol / forecast_vol_t , 0, vol_cap )
    forecast_vol_t = EWMA realized vol using returns up to t-1   (no look-ahead)

vol_cap = 1.0 => long-only, never levered. target_vol is a per-tier constant
(annualized) set in config; we convert it to the per-bar scale here.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_vol(returns, halflife):
    """Backward-looking EWMA volatility of a return series (per-bar scale)."""
    return returns.ewm(halflife=halflife, min_periods=halflife).std()


def overlay_weight(returns, target_vol_annual, halflife=20, vol_cap=1.0):
    """
    Per-bar position weight in [0, vol_cap] that targets a constant volatility.

    returns           : daily simple returns of one asset (pd.Series)
    target_vol_annual : desired annualized vol (e.g. 0.20 = 20%/yr)
    Returns a weight series aligned to `returns`, strictly using only past info
    (forecast vol is shifted by one bar).
    """
    target_per_bar = target_vol_annual / np.sqrt(TRADING_DAYS)
    fvol = ewma_vol(returns, halflife).shift(1)  # no look-ahead
    w = (target_per_bar / fvol).clip(lower=0.0, upper=vol_cap)
    return w.fillna(0.0)


def annualized_sharpe(returns):
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(returns):
    """Max drawdown of a return series, as a negative fraction (e.g. -0.23)."""
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())
