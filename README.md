# Goal-Allocator — a personal, rules-based robo-advisor

A goal-based money manager for one client (you). It does **not** try to beat the
market — two prior projects (Gordian: direction; Vol-Gordian: vol-for-Sharpe)
proved that edge isn't reachable with the data on hand. Instead it does the part
that *is* solvable and that real-world AUM actually runs on:

> **Split a pool across risk tiers, manage each tier's risk, rebalance with
> discipline, track contributions, and report progress toward a goal.**

## What it is — and isn't

| It does (solvable) | It does NOT (the disproven part) |
|---|---|
| Allocate across 5 risk tiers | Predict which asset will go up |
| Weight a basket by inverse-vol | Pick "winners" |
| Size positions by forecast vol (drawdown control) | Time the market |
| Rebalance on drift bands + calendar | Generate alpha |
| Deploy contributions to laggards | Promise high-risk returns |
| Project goal probability (Monte Carlo) | Forecast returns |

The high-risk tier gives **higher expected return with bigger swings**; the
allocator's job is to keep those swings inside what your goal can survive — not
to make a bad year good.

## Design choices (locked)

- **All 5 tiers are growth-and-above** — no internal bond/cash ballast. Your
  treasury bills + EFP account (held externally) are the ballast. This pool is
  volatile by design; the vol overlay is its only internal cushion.
- **Vol overlay is the risk engine** — carried from Vol-Gordian, where it cut max
  drawdown on 8/8 assets. That drawdown control is the empirically robust lever.
- **Options overlays** (covered calls / protective puts) — modeled
  parametrically in a later phase (free historical option chains are unreliable).

## Layout

```
config/config.json   5 tiers, goal, contributions, rebalancing rules, universe
src/data.py          fetch daily adjusted-close for the candidate universe
src/vol_overlay.py    EWMA vol forecast + inverse-vol position sizing (no look-ahead)
src/tiers.py         per-tier return streams + today's target weights
src/rebalance.py     dollar-level pool sim: contributions + drift-band rebalancing
src/goal.py          forward Monte Carlo goal projection (currency-aware)
src/backtest.py      orchestrates + judges vs same-cash benchmark DCA
src/account.py       ADVISORY ledger: deposits, trades, holdings, order generation
app.py               Streamlit account manager (config editor + deposit/allocate UI)
```

## Quick start — research backtest

```bash
pip install -r requirements.txt
python src/data.py     --config config/config.json   # fetch prices -> data/prices.csv
python src/backtest.py --config config/config.json   # run + report
```

## Run the account-manager app

```bash
streamlit run app.py
```

Advisory only — it never touches your money. You deposit into your IBKR account;
the app tells you exactly what to buy, you execute there, and you record the
fills. Your ledger (`data/ledger.json`) is private and gitignored. To host it,
see [DEPLOY.md](DEPLOY.md) (private repo + password + durable storage).

## Roadmap

1. **Core engine** — tiers + rebalancing + goal tracking + backtest. ← current
2. **Options overlays** — model covered calls (income) / protective puts (hedge).
3. **Dashboard** — Streamlit: view the pool, edit tiers/goals, re-run live.
4. **Regime awareness** — optional HMM bull/bear tilt (reused from the MEI project).

## Disclaimer

**Not financial advice.** Research/educational. Nothing here is cleared for real
capital until validated out-of-sample.
