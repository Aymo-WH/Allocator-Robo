"""
account.py — the operational layer that turns the strategy into a real,
trackable account. ADVISORY ONLY: it never touches your money. Your cash and
shares live in your IBKR account; this records what you've done and tells you
what to do next.

Single source of truth = a ledger of (a) RM deposits and (b) executed trades.
Everything else (holdings, cash, current allocation) is derived from those, so
the numbers can never silently drift from reality.

The ledger lives at data/ledger.json, which is gitignored — your financial data
is never committed to the repo.
"""
import json
import os

import pandas as pd
import yfinance as yf

from tiers import latest_target_weights

LEDGER_PATH = "data/ledger.json"


# ---------- ledger persistence ------------------------------------------------
def empty_ledger():
    return {"deposits": [], "trades": []}


def load_ledger(path=LEDGER_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return empty_ledger()


def save_ledger(ledger, path=LEDGER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2)


def add_deposit(ledger, date, amount_myr, fx_usdmyr):
    """Record an RM deposit; convert to USD at the rate used."""
    amount_usd = float(amount_myr) / float(fx_usdmyr)
    ledger["deposits"].append({
        "date": date, "amount_myr": float(amount_myr),
        "fx_usdmyr": float(fx_usdmyr), "amount_usd": amount_usd,
    })
    return amount_usd


def add_trade(ledger, date, product, side, units, price_usd, fee_usd=0.0):
    """Record an executed fill. side = 'BUY' or 'SELL'."""
    ledger["trades"].append({
        "date": date, "product": product, "side": side.upper(),
        "units": float(units), "price_usd": float(price_usd), "fee_usd": float(fee_usd),
    })


# ---------- derived state -----------------------------------------------------
def derive_holdings(ledger):
    """product -> units held, from the trade history."""
    h = {}
    for t in ledger["trades"]:
        sign = 1.0 if t["side"] == "BUY" else -1.0
        h[t["product"]] = h.get(t["product"], 0.0) + sign * t["units"]
    return {p: u for p, u in h.items() if abs(u) > 1e-9}


def derive_cash_usd(ledger):
    """Uninvested USD cash = deposits - buys - fees + sells."""
    cash = sum(d["amount_usd"] for d in ledger["deposits"])
    for t in ledger["trades"]:
        gross = t["units"] * t["price_usd"]
        cash += (-gross if t["side"] == "BUY" else gross) - t["fee_usd"]
    return cash


def total_deposited_myr(ledger):
    return sum(d["amount_myr"] for d in ledger["deposits"])


# ---------- live market data --------------------------------------------------
def fetch_market(cfg, hist_days=200):
    """
    Returns (hist_df, latest_prices: dict, latest_fx_usdmyr: float).
    hist_df is recent daily history (for the inverse-vol / overlay calc);
    latest_prices marks holdings to market.
    """
    from data import all_tickers, fx_symbol
    tickers = all_tickers(cfg)
    raw = yf.download(tickers, period=f"{hist_days}d", interval="1d",
                      auto_adjust=True, group_by="ticker", progress=False)
    closes = {}
    for t in tickers:
        try:
            closes[t] = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        except KeyError:
            pass
    hist = pd.DataFrame(closes).sort_index().ffill().dropna()

    fx_col = fx_symbol(cfg)
    latest_fx = float(hist[fx_col].iloc[-1]) if fx_col and fx_col in hist.columns else None
    product_cols = [c for c in hist.columns if c != fx_col]
    latest = {c: float(hist[c].iloc[-1]) for c in product_cols}
    return hist[product_cols], latest, latest_fx


# ---------- the advisory engine ----------------------------------------------
def mark_to_market(holdings, latest_prices):
    """product -> current USD value."""
    return {p: holdings.get(p, 0.0) * latest_prices.get(p, 0.0) for p in
            set(holdings) | set(latest_prices)}


def build_orders(cfg, hist, latest_prices, holdings, cash_usd, whole_units=True):
    """
    Compare current holdings to today's target allocation and produce an order
    list to close the gap. Returns (orders_df, summary_dict).

    orders_df columns: product, tier_target_%, current_value, target_value,
                       delta_value, side, units, est_cost
    """
    weights, cash_weight = latest_target_weights(hist, cfg)
    mtm = mark_to_market(holdings, latest_prices)
    invested_now = sum(mtm.values())
    total = invested_now + cash_usd

    universe = sorted(set(weights) | set(holdings))
    rows = []
    for p in universe:
        px = latest_prices.get(p)
        if not px:
            continue
        tgt_val = total * weights.get(p, 0.0)
        cur_val = mtm.get(p, 0.0)
        delta = tgt_val - cur_val
        raw_units = delta / px
        units = round(raw_units) if whole_units else round(raw_units, 4)
        if units == 0:
            continue
        rows.append({
            "product": p,
            "target_%": round(100 * weights.get(p, 0.0), 2),
            "current_value": round(cur_val, 2),
            "target_value": round(tgt_val, 2),
            "delta_value": round(delta, 2),
            "side": "BUY" if units > 0 else "SELL",
            "units": abs(units),
            "est_cost": round(abs(units) * px, 2),
        })
    orders = pd.DataFrame(rows).sort_values("delta_value", ascending=False) if rows else pd.DataFrame(
        columns=["product", "target_%", "current_value", "target_value",
                 "delta_value", "side", "units", "est_cost"])

    buy_cost = orders.loc[orders.side == "BUY", "est_cost"].sum() if not orders.empty else 0.0
    sell_proceeds = orders.loc[orders.side == "SELL", "est_cost"].sum() if not orders.empty else 0.0
    summary = {
        "total_value_usd": total,
        "invested_usd": invested_now,
        "cash_usd": cash_usd,
        "target_cash_%": round(100 * cash_weight, 2),
        "net_buy_usd": buy_cost - sell_proceeds,
        "cash_after_orders": cash_usd - (buy_cost - sell_proceeds),
    }
    return orders, summary
