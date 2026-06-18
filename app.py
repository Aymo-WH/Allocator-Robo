"""
app.py — Goal-Allocator account manager (Streamlit).

ADVISORY ONLY. It never touches your money. You deposit into your IBKR account;
this app tells you exactly what to buy, you execute there, and you record the
fills here. Your ledger (deposits + trades) is stored locally/privately and is
gitignored — never committed.

Run locally:   streamlit run app.py
Deploy:        see DEPLOY.md (use a PRIVATE repo + set APP_PASSWORD in secrets).
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import load_config  # noqa: E402
from tiers import build_all_tiers  # noqa: E402
from goal import project_goal  # noqa: E402
import account as acct  # noqa: E402

CONFIG_PATH = "config/config.json"
PRICES_PATH = "data/prices.csv"

st.set_page_config(page_title="Goal-Allocator", page_icon="📈", layout="wide")


# ---------- optional password gate (enabled if APP_PASSWORD secret is set) ----
def gate():
    pw = st.secrets.get("APP_PASSWORD") if hasattr(st, "secrets") else None
    if not pw:
        return True  # no password configured (local use) -> open
    if st.session_state.get("authed"):
        return True
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Enter") and entered == pw:
            st.session_state["authed"] = True
            st.rerun()
    st.stop()


gate()


# ---------- helpers -----------------------------------------------------------
def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


@st.cache_data(ttl=900)
def market(_cfg_json):
    cfg = json.loads(_cfg_json)
    return acct.fetch_market(cfg, hist_days=200)


def fmt(x, ccy=""):
    return f"{ccy}{x:,.0f}" if ccy else f"{x:,.2f}"


cfg = load_config(CONFIG_PATH)
ledger = acct.load_ledger()
cur = cfg["goal"].get("base_currency", "USD")

st.title("📈 Goal-Allocator — personal account manager")
st.caption("Advisory only. Your money stays in your IBKR account; this tells you "
           "what to buy and tracks your progress.")

# live market (cached 15 min)
try:
    hist, latest_px, latest_fx = market(json.dumps(cfg))
    mkt_ok = True
except Exception as e:  # noqa: BLE001
    st.warning(f"Could not fetch live prices: {e}")
    hist, latest_px, latest_fx, mkt_ok = None, {}, None, False

holdings = acct.derive_holdings(ledger)
cash_usd = acct.derive_cash_usd(ledger)

tab_overview, tab_alloc, tab_ledger, tab_goal, tab_config = st.tabs(
    ["Overview", "Deposit & Allocate", "Holdings & Ledger", "Goal", "Config"])


# ============================== OVERVIEW =====================================
with tab_overview:
    if not mkt_ok:
        st.stop()
    mtm = acct.mark_to_market(holdings, latest_px)
    invested = sum(mtm.values())
    total_usd = invested + cash_usd
    total_myr = total_usd * latest_fx if latest_fx else total_usd

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Total value ({cur})", fmt(total_myr, cur + " "))
    c2.metric("Total value (USD)", fmt(total_usd, "$"))
    c3.metric("Invested (USD)", fmt(invested, "$"))
    c4.metric("Uninvested cash (USD)", fmt(cash_usd, "$"))

    st.divider()
    st.subheader("Current vs target allocation")
    orders, summ = acct.build_orders(cfg, hist, latest_px, holdings, cash_usd)
    from tiers import latest_target_weights
    weights, cash_w = latest_target_weights(hist, cfg)
    rows = []
    for p in sorted(set(weights) | set(holdings)):
        cur_v = mtm.get(p, 0.0)
        rows.append({"product": p,
                     "current %": round(100 * cur_v / total_usd, 1) if total_usd else 0,
                     "target %": round(100 * weights.get(p, 0.0), 1)})
    rows.append({"product": "CASH",
                 "current %": round(100 * cash_usd / total_usd, 1) if total_usd else 0,
                 "target %": round(100 * cash_w, 1)})
    alloc = pd.DataFrame(rows).set_index("product")
    st.bar_chart(alloc)
    st.dataframe(alloc, use_container_width=True)


# ========================== DEPOSIT & ALLOCATE ===============================
with tab_alloc:
    st.subheader("1) Record a deposit (in " + cur + ")")
    col1, col2, col3 = st.columns(3)
    dep_amt = col1.number_input(f"Deposit amount ({cur})", min_value=0.0, value=0.0, step=100.0)
    dep_date = col2.text_input("Date", value=str(pd.Timestamp.today().date()))
    fx_default = float(latest_fx) if latest_fx else 4.5
    fx_used = col3.number_input("USD/MYR rate", min_value=0.0, value=round(fx_default, 4), step=0.01)
    if st.button("Record deposit") and dep_amt > 0:
        usd = acct.add_deposit(ledger, dep_date, dep_amt, fx_used)
        acct.save_ledger(ledger)
        st.success(f"Recorded {cur} {dep_amt:,.0f}  →  ${usd:,.2f} added to cash. Rerun to refresh.")

    st.divider()
    st.subheader("2) Proposed orders to reach target allocation")
    if not mkt_ok:
        st.info("Live prices unavailable; cannot compute orders right now.")
    else:
        orders, summ = acct.build_orders(cfg, hist, latest_px, holdings, cash_usd)
        s1, s2, s3 = st.columns(3)
        s1.metric("Pool value (USD)", fmt(summ["total_value_usd"], "$"))
        s2.metric("Cash available (USD)", fmt(summ["cash_usd"], "$"))
        s3.metric("Net buying needed (USD)", fmt(summ["net_buy_usd"], "$"))
        if summ["net_buy_usd"] > summ["cash_usd"] + 1:
            st.warning("Net buys exceed available cash — deposit more or the orders "
                       "assume selling the listed SELL lines first.")
        if orders.empty:
            st.info("Already at target — no orders needed.")
        else:
            st.caption("Place these at IBKR, then record the actual fills below.")
            st.dataframe(orders, use_container_width=True)

            st.subheader("3) Record fills (edit to match your actual IBKR execution)")
            editable = orders[["product", "side", "units"]].copy()
            editable["fill_price_usd"] = [latest_px.get(p, 0.0) for p in editable["product"]]
            editable["fee_usd"] = 0.0
            edited = st.data_editor(editable, use_container_width=True, num_rows="dynamic",
                                    key="fills_editor")
            if st.button("Commit these fills to the ledger"):
                n = 0
                for _, r in edited.iterrows():
                    if r["units"] and r["fill_price_usd"]:
                        acct.add_trade(ledger, dep_date, r["product"], r["side"],
                                       r["units"], r["fill_price_usd"], r.get("fee_usd", 0.0))
                        n += 1
                acct.save_ledger(ledger)
                st.success(f"Recorded {n} fills. Rerun to refresh holdings.")


# ========================== HOLDINGS & LEDGER ================================
with tab_ledger:
    st.subheader("Current holdings")
    if holdings:
        rows = [{"product": p, "units": round(u, 4),
                 "price_usd": round(latest_px.get(p, 0.0), 2),
                 "value_usd": round(u * latest_px.get(p, 0.0), 2)}
                for p, u in sorted(holdings.items())]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No holdings yet.")
    st.metric("Uninvested cash (USD)", fmt(cash_usd, "$"))

    st.divider()
    st.subheader("Deposits")
    st.dataframe(pd.DataFrame(ledger["deposits"]) if ledger["deposits"]
                 else pd.DataFrame(columns=["date", "amount_myr", "fx_usdmyr", "amount_usd"]),
                 use_container_width=True)
    st.subheader("Trades")
    st.dataframe(pd.DataFrame(ledger["trades"]) if ledger["trades"]
                 else pd.DataFrame(columns=["date", "product", "side", "units", "price_usd", "fee_usd"]),
                 use_container_width=True)

    st.divider()
    cc1, cc2 = st.columns(2)
    if cc1.button("Undo last trade") and ledger["trades"]:
        ledger["trades"].pop()
        acct.save_ledger(ledger)
        st.success("Removed last trade. Rerun to refresh.")
    if cc2.button("Undo last deposit") and ledger["deposits"]:
        ledger["deposits"].pop()
        acct.save_ledger(ledger)
        st.success("Removed last deposit. Rerun to refresh.")


# =============================== GOAL ========================================
with tab_goal:
    st.subheader("Goal projection")
    g = cfg["goal"]
    st.write(f"**Goal:** {cur} {g['target_amount']:,.0f} in {g['horizon_years']} years "
             f"· contributing {cur} {g['monthly_contribution']:,.0f}/mo")
    if not os.path.exists(PRICES_PATH):
        st.info("Run `python src/data.py` once to build history for the projection.")
    else:
        prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True).sort_index()
        tier_rets, _ = build_all_tiers(prices, cfg)
        import numpy as np
        policy = np.array([t["policy_weight"] for t in cfg["tiers"]])
        policy = policy / policy.sum()
        pool_ret = (tier_rets[[t["name"] for t in cfg["tiers"]]] * policy).sum(axis=1)

        from data import fx_symbol
        fxc = fx_symbol(cfg)
        fx_ret = prices[fxc].pct_change().dropna() if fxc and fxc in prices.columns else None

        # start from live pool value if we have one, else starting_capital
        if mkt_ok and (holdings or cash_usd):
            mtm = acct.mark_to_market(holdings, latest_px)
            start_myr = (sum(mtm.values()) + cash_usd) * (latest_fx or 1.0)
            st.caption(f"Projecting forward from your CURRENT pool value: {cur} {start_myr:,.0f}")
        else:
            start_myr = float(g["starting_capital"])
            st.caption(f"No holdings yet — projecting from starting_capital: {cur} {start_myr:,.0f}")

        proj = project_goal(pool_ret, cfg, current_value=start_myr,
                            fx_returns=fx_ret, n_paths=cfg["projection"]["n_paths"])
        scen = cfg["projection"]["annual_return_scenarios"]
        rows = [{"scenario": n, "assumed return": f"{scen[n]:.0%}",
                 "P(hit goal)": f"{proj[n]['p_hit_goal']:.0%}",
                 "median": f"{cur} {proj[n]['median']:,.0f}",
                 "P10–P90": f"{cur} {proj[n]['p10']:,.0f} – {proj[n]['p90']:,.0f}"}
                for n in scen]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Drift is an assumption (conservative/base/optimistic); volatility "
                   "is from the data; USD/MYR FX risk is modeled. Trust the "
                   "conservative/base rows for planning.")


# ============================== CONFIG =======================================
with tab_config:
    st.subheader("Goal")
    g = cfg["goal"]
    gc1, gc2, gc3 = st.columns(3)
    g["target_amount"] = gc1.number_input("Target amount", value=float(g["target_amount"]), step=10000.0)
    g["horizon_years"] = gc2.number_input("Horizon (years)", value=int(g["horizon_years"]), step=1)
    g["base_currency"] = gc3.text_input("Base currency", value=g.get("base_currency", "MYR"))
    g["starting_capital"] = gc1.number_input("Starting capital", value=float(g["starting_capital"]), step=1000.0)
    g["monthly_contribution"] = gc2.number_input("Monthly contribution", value=float(g["monthly_contribution"]), step=100.0)

    st.divider()
    st.subheader("Tiers")
    st.caption("Policy weights should sum to 1.0. Candidates are comma-separated tickers.")
    for t in cfg["tiers"]:
        st.markdown(f"**{t['name']}** — {t['label']}")
        tc1, tc2, tc3 = st.columns([1, 1, 3])
        t["policy_weight"] = tc1.number_input(f"policy_weight [{t['name']}]", value=float(t["policy_weight"]),
                                              step=0.05, min_value=0.0, max_value=1.0, key=f"pw_{t['name']}")
        t["target_vol_annual"] = tc2.number_input(f"target_vol [{t['name']}]", value=float(t["target_vol_annual"]),
                                                  step=0.01, min_value=0.01, key=f"tv_{t['name']}")
        cand = tc3.text_input(f"candidates [{t['name']}]", value=", ".join(t["candidates"]), key=f"cd_{t['name']}")
        t["candidates"] = [c.strip() for c in cand.split(",") if c.strip()]

    wsum = sum(t["policy_weight"] for t in cfg["tiers"])
    if abs(wsum - 1.0) > 0.001:
        st.warning(f"Policy weights sum to {wsum:.2f}, not 1.0 — fix before saving.")

    st.divider()
    if st.button("💾 Save config"):
        save_config(cfg)
        st.cache_data.clear()
        st.success("Saved to config/config.json. Rerun to apply everywhere.")
