"""
backtest.py — run the full goal-allocator and judge it.

Pipeline:
  prices -> per-tier return streams (tiers.py)
         -> dollar-level pool simulation w/ contributions + rebalancing (rebalance.py)
         -> metrics vs a same-cash SPY DCA benchmark
         -> forward Monte Carlo goal projection (goal.py)

Success is NOT "beat SPY on raw return" — your own data showed that's not on the
table. Success = comparable risk-adjusted return with SHALLOWER drawdown, plus a
credible probability of hitting the goal given your contributions.
"""
import argparse

import pandas as pd

from data import load_config, fx_symbol
from tiers import build_all_tiers
from rebalance import simulate_pool, benchmark_dca
from vol_overlay import annualized_sharpe, max_drawdown
from goal import project_goal


def _fmt_money(x):
    return f"${x:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--prices", default="data/prices.csv")
    ap.add_argument("--mc_paths", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index()

    # 1) tier return streams
    tier_rets, expo = build_all_tiers(prices, cfg)

    # 2) simulate the managed pool
    sim = simulate_pool(tier_rets, cfg)
    pool_val = sim["value"]
    pool_ret = sim["port_ret"]

    # 3) benchmark: same cash schedule DCA'd into SPY.
    # Use the CLEAN, contribution-neutral return series for Sharpe/MaxDD — the
    # dollar path (bench_val) includes cash inflows, so its pct_change would read
    # every contribution day as a fake return and inflate Sharpe.
    bench_val, bench_ret = benchmark_dca(prices, cfg)

    # ---- report --------------------------------------------------------------
    print("=" * 78)
    print("GOAL-ALLOCATOR BACKTEST  |  5 tiers, vol-overlay risk engine, "
          "drift-band + calendar rebalancing")
    print(f"Period: {prices.index[0].date()} -> {prices.index[-1].date()}  "
          f"({len(prices)} trading days)")
    print("=" * 78)

    print("\nPer-tier average exposure (mean total long weight; <1 => overlay "
          "held cash):")
    for tier in cfg["tiers"]:
        n = tier["name"]
        print(f"  {n:16s} {tier['label']:22s} policy {tier['policy_weight']:.0%}"
              f"   avg_expo {expo[n]:.2f}   tier Sharpe {annualized_sharpe(tier_rets[n]):+.2f}")

    print("\n" + "-" * 78)
    print(f"{'':22s}{'ALLOCATOR':>16s}{'SPY DCA (bench)':>18s}")
    print("-" * 78)
    print(f"{'Final value':22s}{_fmt_money(pool_val.iloc[-1]):>16s}"
          f"{_fmt_money(bench_val.iloc[-1]):>18s}")
    print(f"{'Total contributed':22s}{_fmt_money(sim['contributed']):>16s}"
          f"{_fmt_money(sim['contributed']):>18s}")
    print(f"{'Sharpe (ann.)':22s}{annualized_sharpe(pool_ret):>16.2f}"
          f"{annualized_sharpe(bench_ret):>18.2f}")
    print(f"{'Max drawdown':22s}{max_drawdown(pool_ret):>15.1%}"
          f"{max_drawdown(bench_ret):>17.1%}")
    print(f"{'Rebalancing fees':22s}{_fmt_money(sim['fees_paid']):>16s}{'-':>18s}")
    print("-" * 78)
    dd_better = max_drawdown(pool_ret) > max_drawdown(bench_ret)
    print("Read: the allocator should show a SHALLOWER drawdown than SPY DCA "
          "(its job).")
    print(f"      Drawdown shallower than benchmark? {'YES' if dd_better else 'NO'}")

    # 4) forward goal projection — currency-aware, conservative-drift scenarios
    g = cfg["goal"]
    cur = g.get("base_currency", "USD")
    fx_returns = None
    fx_col = fx_symbol(cfg)
    if fx_col and fx_col in prices.columns:
        fx_returns = prices[fx_col].pct_change().dropna()

    # Project FORWARD from your real current capital (starting_capital), using the
    # strategy's empirical volatility. Do NOT start from the backtest's terminal
    # value — that would assume you'd already been invested since the data began.
    start_today = float(g["starting_capital"])
    proj = project_goal(pool_ret, cfg, current_value=start_today,
                        fx_returns=fx_returns, n_paths=cfg["projection"]["n_paths"])

    print("\n" + "=" * 78)
    print("GOAL PROJECTION (forward from today's capital) | drift = ASSUMPTION, vol = DATA")
    print("=" * 78)
    print(f"Goal: {cur} {g['target_amount']:,.0f} in {g['horizon_years']} yrs"
          f"  |  contributing {cur} {g['monthly_contribution']:,.0f}/mo"
          f"  |  starting from {cur} {start_today:,.0f}")
    fx_note = ("ON (USD/MYR vol modeled, drift neutral)" if proj["_fx_modeled"]
               else "OFF")
    print(f"FX risk modeling: {fx_note}\n")
    print(f"  {'Scenario':16s}{'assumed ann. return':>22s}{'P(hit goal)':>14s}"
          f"{'median':>16s}{'P10 - P90':>26s}")
    print("  " + "-" * 92)
    scen = cfg["projection"]["annual_return_scenarios"]
    for name in scen:
        r = proj[name]
        rng = f"{cur} {r['p10']:,.0f} - {r['p90']:,.0f}"
        print(f"  {name:16s}{scen[name]:>21.0%}{r['p_hit_goal']:>14.0%}"
              f"{cur+' '+format(r['median'], ',.0f'):>16s}{rng:>26s}")
    print("  " + "-" * 92)
    print(f"  Total you'd contribute over the horizon: "
          f"{cur} {proj['base']['total_contributed']:,.0f}")
    print("Read: trust the CONSERVATIVE/BASE rows for planning; the optimistic row is "
          "an upside case, not an expectation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
