import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
import os
import sqlite3
from datetime import date, timedelta

import database
from core import (
    format_inr, format_date_in, get_stage, months_between, add_months,
    calc_emi, calc_dpd_asof, payment_status, is_loan_closed,
    build_monthly_snapshot, compute_9_features, FEATURE_ORDER,
    MIN_MONTHS_FOR_PREDICTION, TOLERANCE, validate_payment,
)

st.set_page_config(page_title="PraSMA Dashboard", layout="wide")

# NOTE (DB migration): account data now lives in SQLite (database.py), not
# st.session_state. "active_account" (which account is currently selected)
# stays in session_state on purpose — that's UI state, not data, and needs
# to persist across reruns exactly the way it did before.

# NOTE: priority_score (risk % x stage_severity_weight) was retired by design —
# it let a stable-but-severe account (e.g. SMA-2, 30% risk) outrank an
# unstable-but-early one (e.g. Standard, 95% risk), since severity and risk
# got multiplied into one blended number and compared ACROSS stages.
# Fix: rank accounts only WITHIN their current stage, purely by risk_probability.
# Four separate watchlists, one per transition, replace the single blended list.
WATCHLIST_LABEL = {
    "Standard": "Standard \u2192 SMA-0 Watchlist",
    "SMA-0": "SMA-0 \u2192 SMA-1 Watchlist",
    "SMA-1": "SMA-1 \u2192 SMA-2 Watchlist",
    "SMA-2": "SMA-2 \u2192 NPA Watchlist",
}
WATCHLIST_STAGES = list(WATCHLIST_LABEL.keys())  # NPA excluded — handled as a separate reference table, not a predictive watchlist

MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model():
    """Loads model.pkl once per app session and reuses it — without this,
    Streamlit would re-read the pickle file from disk on every single
    interaction (every button click triggers a full script rerun)."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["feature_order"]


def real_risk_score(features):
    """Runs the trained Logistic Regression (train_model.py) on this
    account's 9 features. Returns None if model.pkl isn't present yet,
    so the dashboard can fall back gracefully instead of crashing."""
    model, scaler, feature_order = load_model()
    if model is None:
        return None
    x = np.array([[features[f] for f in feature_order]])
    x_scaled = scaler.transform(x)
    return round(float(model.predict_proba(x_scaled)[0][1]), 3)


def placeholder_risk_score(features):
    """
    Fallback ONLY — used if model.pkl hasn't been generated yet (i.e.
    train_model.py hasn't been run). Rough weighted rule, NOT the real
    model. Once model.pkl exists, real_risk_score() is used instead.
    """
    score = 0.0
    score += min(features["dpd_trend"] / 10, 1.0) * 0.35 if features["dpd_trend"] > 0 else 0
    score += (features["consecutive_missed_months"] / 6) * 0.30  # replaces missed_payment_count, removed for collinearity
    score += (features["partial_payment_count"] / 6) * 0.15
    score += min(features["payment_volatility"] * 2, 1.0) * 0.10
    score += min(features["prior_sma_transitions"] / 4, 1.0) * 0.10
    return round(min(score, 1.0), 3)


def get_risk_score(features):
    """Single entry point the rest of the app calls — tries the real
    trained model first, falls back to the placeholder rule if model.pkl
    isn't present, and tells you (once) which one is active."""
    score = real_risk_score(features)
    if score is not None:
        return score, True  # True = using the real trained model
    return placeholder_risk_score(features), False  # False = fallback rule


@st.cache_data(show_spinner=False)
def _compute_account_metrics_cached(loan_amount, start_date, emi_due, tenure_months, history_tuple):
    """
    The expensive per-account work (DPD/stage engine + 9 features + model
    inference), cached and keyed on the account's actual data.

    WHY THIS EXISTS: without it, every account in the portfolio gets fully
    recomputed on EVERY rerun — and Streamlit reruns the entire script on
    every single interaction anywhere in the app, not just changes to that
    account. With enough accounts, this made the app feel laggy, and it got
    worse with every account added since the cost is per-account x every
    rerun. Caching on (loan terms, history) means an account whose data
    hasn't changed since the last rerun is served instantly from cache —
    only the account that actually just changed gets recomputed.

    history_tuple must be a hashable, order-stable representation of the
    account's payment history (a tuple of (month_date, amount_paid) tuples)
    — plain dicts/lists aren't hashable, so the caller converts first.
    """
    acc = {
        "loan_amount": loan_amount,
        "start_date": start_date,
        "emi_due": emi_due,
        "tenure_months": tenure_months,
        "history": [{"month_date": d, "amount_paid": a} for d, a in history_tuple],
    }
    snapshots = build_monthly_snapshot(acc)
    features = compute_9_features(acc, snapshots)
    if len(snapshots) >= MIN_MONTHS_FOR_PREDICTION:
        risk_pct, using_real_model = get_risk_score(features)
    else:
        risk_pct, using_real_model = None, False
    return snapshots, features, risk_pct, using_real_model


def get_account_metrics(acc):
    """Public wrapper — converts the account's history into the hashable
    form the cache needs, then delegates to the cached computation."""
    history_tuple = tuple((r["month_date"], r["amount_paid"]) for r in acc["history"])
    return _compute_account_metrics_cached(
        acc["loan_amount"], acc["start_date"], acc["emi_due"], acc["tenure_months"], history_tuple
    )


def load_account(account_id):
    """Reads one account + its full payment history from SQLite (database.py)
    and reshapes it into the exact dict shape core.py already expects:
    {loan_amount, interest_rate, start_date, end_date, tenure_months,
    emi_due, history: [{month_date, amount_paid}, ...]}.

    database.py stores dates as ISO strings ("2024-01-31") — this is the one
    place that converts them back into real date objects, so every other
    function in this file (and everything in core.py) keeps working with
    actual `date` objects exactly as it did when data lived in memory.
    Returns None if the account_id doesn't exist.
    """
    row = database.get_account(account_id)
    if not row:
        return None
    payments = database.get_payments(account_id)
    return {
        "loan_amount": row["loan_amount"],
        "interest_rate": row["interest_rate"],
        "start_date": date.fromisoformat(row["start_date"]),
        "end_date": date.fromisoformat(row["end_date"]),
        "tenure_months": row["tenure_months"],
        "emi_due": row["emi_due"],
        "history": [
            {"month_date": date.fromisoformat(p["month_date"]), "amount_paid": p["amount_paid"]}
            for p in payments
        ],
    }


# Streamlit reruns this whole script on every interaction, so the account
# list is fetched fresh once per rerun and reused everywhere below —
# avoids repeated database round-trips within the same rerun, while still
# always reflecting the latest data (a fresh fetch happens on every rerun).
all_account_ids = [a["account_id"] for a in database.get_all_accounts()]


# ---------- Sidebar: Setup inputs (5 one-time fields) ----------

with st.sidebar.expander("➕ Add New Account", expanded=(len(all_account_ids) == 0)):
    new_id = st.text_input("Account ID / Borrower Name")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=500000, step=1000)
    st.caption(f"= {format_inr(loan_amount)}")
    interest_rate = st.slider("Interest Rate (% per annum)", min_value=8.0, max_value=14.0, value=10.5, step=0.1)
    start_date = st.date_input("Loan Start Date", value=date.today() - timedelta(days=180), format="DD/MM/YYYY")
    end_date = st.date_input("Loan End Date", value=date.today() + timedelta(days=365 * 4), format="DD/MM/YYYY")

    if st.button("Create Account"):
        if not new_id:
            st.error("Enter an Account ID")
        elif new_id in all_account_ids:
            st.error("This Account ID already exists")
        elif end_date <= start_date:
            st.error("End Date must be after Start Date")
        else:
            emi_due, tenure_months = calc_emi(loan_amount, interest_rate, start_date, end_date)
            try:
                database.create_account(
                    new_id, loan_amount, interest_rate, start_date, end_date, tenure_months, emi_due,
                )
            except sqlite3.IntegrityError:
                # Backup for the all_account_ids check above (e.g. a stale list
                # within the same rerun) — accounts.account_id is a PRIMARY KEY,
                # so the database itself refuses a second row with the same id.
                st.error("This Account ID already exists")
            else:
                st.session_state["active_account"] = new_id  # newly created account becomes the active one
                st.success(f"Account {new_id} created — EMI {format_inr(emi_due, decimals=2)}/month, tenure {tenure_months} months")
                st.rerun()  # refresh all_account_ids so the new account shows up immediately everywhere

# ---------- Sidebar: single shared "Active Account" selector ----------
# FIX: previously the payment form and the Account Detail tab each had their
# OWN separate st.selectbox (different `key`s), so picking an account in one
# never updated the other. Paying against Account 2 in the sidebar left the
# main dashboard still showing whichever account the detail-tab dropdown
# happened to be on (often Account 1, untouched since creation) -- looking
# like the dashboard "switched back". One shared selector, used everywhere,
# fixes that at the source.
st.sidebar.markdown("---")
if all_account_ids:
    if "active_account" not in st.session_state or st.session_state["active_account"] not in all_account_ids:
        st.session_state["active_account"] = all_account_ids[0]
    st.sidebar.selectbox("🏦 Active Account", all_account_ids, key="active_account")

with st.sidebar.expander("💰 Add Monthly Payment", expanded=True):
    if all_account_ids:
        pay_id = st.session_state["active_account"]
        acc_for_default = load_account(pay_id)

        # First EMI is due one month after loan start — standard loan practice,
        # never the same day as disbursement. Each subsequent default follows the
        # same one-month cadence from whichever payment was logged most recently,
        # so the field is always pre-filled with the NEXT expected due date
        # instead of defaulting to today (which drifts from the real schedule).
        if acc_for_default["history"]:
            last_month_date = max(r["month_date"] for r in acc_for_default["history"])
            default_payment_date = add_months(last_month_date, 1)
        else:
            default_payment_date = add_months(acc_for_default["start_date"], 1)

        # --- FIX (Problem 3): loan already fully repaid -> no more payments ---
        # Checked BEFORE rendering the input fields, using the account's state
        # as it stood before this submission, so the payment that actually
        # completes the loan is still accepted (it's the one that flips
        # is_loan_closed to True) -- only payments AFTER that are blocked.
        if is_loan_closed(acc_for_default):
            st.success("🎉 Loan Completed — this account is fully repaid. No further payments can be added.")
        else:
            # IMPORTANT: key= is tied to (account, number of payments logged so far).
            # Without this, Streamlit treats the date field as the SAME widget across
            # reruns and keeps whatever value it last held — so after submitting a
            # payment, the field would stay stuck instead of actually advancing to
            # the next month. Changing the key each time a payment is added (or the
            # selected account changes) forces Streamlit to re-initialize the widget
            # fresh, so the auto-advance actually takes effect.
            #
            # If YOU manually pick a different date before submitting, that manual
            # choice is exactly what gets saved — key= only controls what the field
            # starts on, never overrides an active user selection.
            widget_key = f"payment_date_{pay_id}_{len(acc_for_default['history'])}"
            month_date = st.date_input("Payment Date", value=default_payment_date, format="DD/MM/YYYY", key=widget_key)

            # Shown BEFORE submission, using amount_paid=0 just to compute the
            # ceiling — the caption is purely informational at this point.
            _, _, remaining_payable = validate_payment(acc_for_default, month_date, 0)
            st.caption(f"Remaining balance on this loan: {format_inr(remaining_payable, decimals=2)}")

            amount_paid = st.number_input("Amount Paid (₹)", min_value=0, value=0)

            if st.button("Submit Payment"):
                # --- FIX: block a payment that exceeds the loan's remaining
                # balance, AND block re-paying an already-fully-paid month —
                # both checked here via the single shared validate_payment()
                # (core.py), before ever touching the database. Confirmed via
                # testing: without this check, an arbitrarily large payment
                # (e.g. 999999999 on a loan owing only ~3.76 lakh) was
                # previously accepted with no error at all. ---
                is_valid, error_message, _ = validate_payment(acc_for_default, month_date, amount_paid)
                if not is_valid:
                    st.error(f"⚠️ {error_message}")
                else:
                    try:
                        database.add_payment(pay_id, month_date, amount_paid)
                    except sqlite3.IntegrityError:
                        # Backup for the validate_payment() check above —
                        # payments has UNIQUE(account_id, month_date), so the
                        # database itself refuses a second row for the same month.
                        st.error(
                            f"⚠️ {month_date.strftime('%b %Y')} already has a payment logged for this account."
                        )
                    else:
                        st.success("Payment recorded")
                        st.rerun()  # forces the date field to re-initialize immediately with the new next-month default
    else:
        st.info("Create an account first")

# ---------- Main dashboard ----------
st.title("PraSMA — Risk Dashboard")

tab1, tab2 = st.tabs(["Account Detail", "Portfolio Risk List"])

with tab1:
    if not all_account_ids:
        st.info("No accounts yet. Add one from the sidebar.")
    else:
        selected = st.session_state["active_account"]
        st.caption(f"Showing: **{selected}** (change via 🏦 Active Account in the sidebar)")
        acc = load_account(selected)
        snapshots, features, risk_pct, using_real_model = get_account_metrics(acc)
        dpd = snapshots[-1]["dpd"] if snapshots else 0
        stage = snapshots[-1]["stage"] if snapshots else "Standard"
        has_enough_history = len(snapshots) >= MIN_MONTHS_FOR_PREDICTION

        st.subheader("Auto-Calculated Loan Terms")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EMI Due", format_inr(acc['emi_due'], decimals=2))
        c2.metric("Tenure", f"{acc['tenure_months']} months")
        c3.metric("Interest Rate", f"{acc['interest_rate']}%")
        c4.metric("Loan Amount", format_inr(acc['loan_amount']))

        if is_loan_closed(acc):
            total_payable = acc["emi_due"] * acc["tenure_months"]
            total_paid = sum(r["amount_paid"] for r in acc["history"])
            st.success(f"✅ Loan Fully Repaid — Account Closed  ({format_inr(total_paid)} paid of {format_inr(total_payable)} owed)")

        st.subheader("Current Risk Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stage", stage)
        c2.metric("DPD", f"{dpd} days")
        if has_enough_history:
            risk_label = "Risk % (trained model)" if using_real_model else "Risk % (placeholder — run train_model.py)"
            c3.metric(risk_label, f"{risk_pct * 100:.1f}%")
        else:
            c3.metric("Risk %", "—")
            c3.caption(f"Need {MIN_MONTHS_FOR_PREDICTION}+ months of history ({len(snapshots)} so far)")
        c4.metric("Loan Status", "Closed" if is_loan_closed(acc) else "Active")

        oldest_unpaid = snapshots[-1]["oldest_unpaid_date"] if snapshots else None
        if oldest_unpaid:
            st.info(
                f"🔗 **Oldest Unpaid Installment: {oldest_unpaid.strftime('%B %Y')}** "
                f"(due {format_date_in(oldest_unpaid)}) — DPD is counted from this exact date. "
                f"It will only advance once cumulative payments fully clear it, per the FIFO ledger above."
            )
        elif snapshots:
            st.success("✅ Fully caught up — no unpaid installment anchoring DPD right now.")

        if stage in ["SMA-1", "SMA-2", "NPA"]:
            st.warning(f"⚠️ Account has slipped to {stage}")

        if snapshots:
            latest = snapshots[-1]
            st.subheader("Total Due vs. Total Paid (Till This Month)")
            st.caption(
                "This running comparison is what actually drives DPD and Stage above — "
                "not any single month's payment in isolation. The oldest unpaid installment "
                "only clears once cumulative payments catch up to cumulative dues."
            )
            d1, d2, d3 = st.columns(3)
            d1.metric("Total Due Till Now", format_inr(latest["cumulative_due"], decimals=2))
            d2.metric("Total Paid Till Now", format_inr(latest["cumulative_paid"], decimals=2))
            if latest["shortfall_so_far"] > 0:
                d3.metric("Shortfall", format_inr(latest["shortfall_so_far"], decimals=2), delta="behind", delta_color="inverse")
            else:
                d3.metric("Shortfall", "₹0.00 — Caught up", delta="on track", delta_color="normal")

        st.subheader("The 9 Behavioral Features (last 6 months)")
        feat_df = pd.DataFrame([features]).T.rename(columns={0: "Value"})
        st.dataframe(feat_df, width='stretch')

        if len(snapshots) > 1:
            st.subheader("DPD Trend (Month-wise)")
            # "Mon YYYY" labels, spaced evenly per month. We must pass an
            # explicit chronological sort order to Altair — otherwise it
            # sorts the labels alphabetically (e.g. "Aug" before "Jul"),
            # which scrambles the timeline even though the data itself
            # is in the correct order.
            month_labels = [s["month_date"].strftime("%b %Y") for s in snapshots]
            trend_df = pd.DataFrame({"Month": month_labels, "DPD": [s["dpd"] for s in snapshots]})
            chart = (
                alt.Chart(trend_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Month:N", sort=month_labels, title=None),
                    y=alt.Y("DPD:Q", title="DPD (days)"),
                    tooltip=["Month", "DPD"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width='stretch')

        if snapshots:
            st.subheader("Payment History")
            hist_df = pd.DataFrame(snapshots)[
                ["month_date", "amount_paid", "status", "month_status", "dpd", "stage",
                 "cumulative_due", "cumulative_paid", "shortfall_so_far", "oldest_unpaid_date"]
            ].rename(columns={
                "month_status": "Month Status",
                "cumulative_due": "Total Due Till Now",
                "cumulative_paid": "Total Paid Till Now",
                "shortfall_so_far": "Shortfall",
                "oldest_unpaid_date": "Oldest Unpaid Installment",
            })
            hist_df["month_date"] = hist_df["month_date"].apply(format_date_in)
            hist_df["amount_paid"] = hist_df["amount_paid"].apply(lambda v: format_inr(v, decimals=2))
            hist_df["Total Due Till Now"] = hist_df["Total Due Till Now"].apply(lambda v: format_inr(v, decimals=2))
            hist_df["Total Paid Till Now"] = hist_df["Total Paid Till Now"].apply(lambda v: format_inr(v, decimals=2))
            hist_df["Shortfall"] = hist_df["Shortfall"].apply(lambda v: format_inr(v, decimals=2))
            hist_df["Oldest Unpaid Installment"] = hist_df["Oldest Unpaid Installment"].apply(
                lambda v: v.strftime("%b %Y") if v else "— Caught up"
            )
            st.dataframe(hist_df, width='stretch')

with tab2:
    if not all_account_ids:
        st.info("No accounts yet.")
    else:
        _, model_active_check = get_risk_score({f: 0 for f in FEATURE_ORDER})
        risk_col_label = "Risk %" if model_active_check else "Risk % (placeholder)"
        if not model_active_check:
            st.caption("⚠️ model.pkl not found — showing placeholder risk scores. Run train_model.py to enable the trained model.")

        rows = []
        for acc_id in all_account_ids:
            acc = load_account(acc_id)
            snapshots, features, risk_pct, _ = get_account_metrics(acc)
            dpd = snapshots[-1]["dpd"] if snapshots else 0
            stage = snapshots[-1]["stage"] if snapshots else "Standard"

            if risk_pct is not None:
                risk_display = round(risk_pct * 100, 1)
            else:
                risk_display = None  # renders as blank/NaN in the table — "not enough history yet"

            # Repayment Trajectory — NOT used to change the NPA classification
            # itself (RBI's rule is strictly "90+ days overdue," regardless of
            # partial payments — that's the real regulation, not a modeling
            # choice, and it's what makes Layer 1 auditable/defensible). This
            # is a SEPARATE signal shown only in the NPA reference table, to
            # distinguish an account still sending money each month from one
            # that's gone completely dark — genuinely useful for how recovery
            # teams prioritize NPA accounts, without touching the stage rule.
            recent = snapshots[-3:] if len(snapshots) >= 3 else snapshots
            recent_avg_ratio = (sum(s["payment_ratio"] for s in recent) / len(recent)) if recent else 0
            if recent_avg_ratio >= 0.5:
                trajectory = "🟢 Actively Repaying"
            elif recent_avg_ratio > 0:
                trajectory = "🟡 Token Payments Only"
            else:
                trajectory = "🔴 Dormant (no recent payment)"

            rows.append({
                "Account": acc_id,
                "Stage": stage,
                "DPD": dpd,
                "risk_pct_raw": risk_pct,  # kept as a float for sorting; not shown directly
                risk_col_label: risk_display,
                "Loan Amount": acc["loan_amount"],  # kept numeric here for sorting the NPA table
                "Loan Amount ": format_inr(acc["loan_amount"]),  # display copy
                "Loan Status": "Closed" if is_loan_closed(acc) else "Active",
                "Repayment Trajectory": trajectory,
            })
        risk_df = pd.DataFrame(rows)

        # ---- 4 stage-wise watchlists: each account appears in exactly ONE list —
        # the one matching its CURRENT stage — ranked purely by risk % within
        # that stage. No cross-stage comparison, so severity can never distort
        # the ranking (see the note above STAGE_WEIGHT's removal). ----
        watch_tabs = st.tabs([WATCHLIST_LABEL[s] for s in WATCHLIST_STAGES] + ["NPA (Reference)"])

        for tab, stage_key in zip(watch_tabs[:-1], WATCHLIST_STAGES):
            with tab:
                stage_df = (
                    risk_df[risk_df["Stage"] == stage_key]
                    .sort_values("risk_pct_raw", ascending=False, na_position="last")
                    .drop(columns=["risk_pct_raw", "Loan Amount"])
                    .rename(columns={"Loan Amount ": "Loan Amount"})
                )
                if stage_df.empty:
                    st.caption(f"No accounts currently in {stage_key}.")
                else:
                    st.caption(f"{len(stage_df)} account(s) in {stage_key}, sorted by risk of moving to the next stage")
                    st.dataframe(stage_df, width='stretch', hide_index=True)

        # ---- NPA accounts: NOT a predictive watchlist — there's no "next worse
        # stage" for the model to score, so ranking by risk_probability doesn't
        # apply here. Shown as a plain reference table instead, sorted by
        # outstanding loan amount (highest exposure first) since that's what
        # matters for recovery prioritization, a different workflow from the
        # early-warning watchlists above. ----
        with watch_tabs[-1]:
            TRAJECTORY_URGENCY = {"🔴 Dormant (no recent payment)": 0, "🟡 Token Payments Only": 1, "🟢 Actively Repaying": 2}
            npa_slice = risk_df[risk_df["Stage"] == "NPA"].copy()
            npa_slice["_urgency"] = npa_slice["Repayment Trajectory"].map(TRAJECTORY_URGENCY)
            npa_df = (
                npa_slice
                .sort_values(["_urgency", "Loan Amount"], ascending=[True, False])
                .drop(columns=["risk_pct_raw", risk_col_label, "Loan Amount", "_urgency"])
                .rename(columns={"Loan Amount ": "Loan Amount"})
            )
            if npa_df.empty:
                st.caption("No accounts currently in NPA.")
            else:
                st.caption(
                    f"{len(npa_df)} account(s) already in NPA \u2014 handed off to recovery/legal. "
                    "Sorted by urgency first (dormant accounts on top, still-paying ones lower), "
                    "then by outstanding amount. Not part of the predictive watchlists above, since "
                    "NPA is RBI's terminal stage \u2014 there's no further stage for the model to predict "
                    "toward. Repayment Trajectory does NOT affect the NPA classification itself (that's "
                    "strictly DPD-based per RBI norms) \u2014 it only helps recovery teams prioritize WITHIN "
                    "the NPA list."
                )
                st.dataframe(npa_df, width='stretch', hide_index=True)