"""
data.py — fetch daily adjusted-close prices for every candidate ticker in the
config (plus the benchmark) and cache them to data/prices.csv.

Daily bars are the right granularity for an allocator: we are deciding how to
split a pool across tiers and rebalance quarterly, not trading microstructure.
No dollar bars, no FFD, no triple-barrier labels — those belonged to the
direction-prediction projects, which is a problem we are deliberately not solving
here. This is allocation + risk management, not alpha.
"""
import argparse
import json
import os
import sys

import pandas as pd
import yfinance as yf


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def all_tickers(cfg):
    """Every unique ticker referenced anywhere in the config."""
    tickers = set()
    for tier in cfg["tiers"]:
        tickers.update(tier["candidates"])
    tickers.add(cfg["benchmark"]["ticker"])
    return sorted(tickers)


def fetch_prices(tickers, lookback_days, master_ticker):
    """
    Download daily adjusted close for all tickers, then align everything to the
    NYSE trading calendar defined by `master_ticker` (the benchmark, e.g. SPY).

    Crypto (BTC-USD, ETH-USD) trades 7 days/week; equities don't. If we kept the
    union index we'd get ~365 rows/yr with equities forward-filled across
    weekends, which corrupts annualized Sharpe (it assumes 252). Reindexing onto
    the benchmark's real trading days fixes this and loses no crypto info — crypto
    has a print on every NYSE day too.
    """
    print(f"Fetching daily data for {len(tickers)} tickers "
          f"({lookback_days} calendar days)...")
    raw = yf.download(
        tickers,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=True,
    )

    closes = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes[t] = raw[t]["Close"]
            else:  # single-ticker frame has flat columns
                closes[t] = raw["Close"]
        except KeyError:
            print(f"  WARNING: no data for {t}; dropping it.")

    df = pd.DataFrame(closes).sort_index()
    if master_ticker not in df.columns:
        raise ValueError(f"Benchmark {master_ticker} has no data; cannot set calendar.")

    # master calendar = days the benchmark actually traded (its native prints)
    cal = df[master_ticker].dropna().index
    df = df.reindex(cal).ffill()   # ffill only fills equity holidays/late listings
    df = df.dropna()               # drop warmup head where some series lack history
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--out", default="data/prices.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tickers = all_tickers(cfg)
    df = fetch_prices(tickers, cfg["data"]["lookback_days"], cfg["benchmark"]["ticker"])

    if df.empty:
        print("ERROR: no overlapping price history across tickers.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} tickers to {args.out}")
    print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        print(f"NOTE: these tickers had no usable data and were dropped: {missing}")


if __name__ == "__main__":
    main()
