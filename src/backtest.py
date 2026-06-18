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

from data import load_config
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

    # 4) forward goal projection from today's value
    print("\n" + "=" * 78)
    print("GOAL PROJECTION (forward Monte Carlo, block bootstrap of allocator returns)")
    print("=" * 78)
    proj = project_goal(pool_ret, cfg, current_value=pool_val.iloc[-1],
                        n_paths=args.mc_paths)
    g = cfg["goal"]
    print(f"Goal: {_fmt_money(proj['target'])} in {proj['horizon_years']} yrs"
          f"  |  contributing {_fmt_money(g['monthly_contribution'])}/mo"
          f"  |  starting from {_fmt_money(pool_val.iloc[-1])}")
    print(f"  Probability of hitting goal : {proj['p_hit_goal']:.0%}")
    print(f"  Median outcome             : {_fmt_money(proj['median'])}")
    print(f"  Likely range (P10-P90)     : {_fmt_money(proj['p10'])} - {_fmt_money(proj['p90'])}")
    print(f"  Total you'd contribute     : {_fmt_money(proj['total_contributed'])}")
    print("=" * 78)


if __name__ == "__main__":
    main()
