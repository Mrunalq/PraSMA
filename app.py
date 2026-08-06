import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

st.set_page_config(page_title="PraSMA Dashboard", layout="wide")

if "accounts" not in st.session_state:
    st.session_state.accounts = {}

STAGE_WEIGHT = {"Standard": 0.5, "SMA-0": 1.0, "SMA-1": 1.5, "SMA-2": 2.0, "NPA": 2.5}

# Rupee tolerance for "underpaid" checks — real EMIs have paise (e.g. ₹10,264.90)
# but people naturally type rounded amounts (e.g. ₹10,264). Without this, a
# genuinely full payment gets flagged as short by a few paise and DPD starts
# counting from that month forever, even though nothing was actually missed.
TOLERANCE = 5.0  # ₹5 buffer — adjust if your team wants a different threshold


# ---------- Core calculation functions ----------

def get_stage(dpd):
    if dpd == 0:
        return "Standard"
    elif dpd <= 30:
        return "SMA-0"
    elif dpd <= 60:
        return "SMA-1"
    elif dpd <= 90:
        return "SMA-2"
    else:
        return "NPA"


def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def calc_emi(loan_amount, interest_rate, start_date, end_date):
    n = months_between(start_date, end_date)
    n = max(n, 1)
    r = (interest_rate / 12) / 100
    if r == 0:
        emi = loan_amount / n
    else:
        emi = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return round(emi, 2), n


def calc_dpd_asof(eval_date, start_date, emi_due, history):
    """DPD as of a given date, using only history entries up to that date."""
    due_day = start_date.day
    oldest_unpaid_date = None
    sorted_history = sorted(history, key=lambda r: r["month_date"])
    for record in sorted_history:
        if record["month_date"] > eval_date:
            continue
        if record["amount_paid"] < emi_due - TOLERANCE:
            if oldest_unpaid_date is None:
                oldest_unpaid_date = date(record["month_date"].year, record["month_date"].month, due_day)
    if oldest_unpaid_date is None:
        return 0
    return max((eval_date - oldest_unpaid_date).days, 0)


def payment_status(paid, due):
    if paid <= 0:
        return "Missed"
    elif paid < due - TOLERANCE:
        return "Partial"
    else:
        return "Full"


def build_monthly_snapshot(acc):
    """For each month in history, compute DPD-as-of and stage-as-of that point in time."""
    sorted_history = sorted(acc["history"], key=lambda r: r["month_date"])
    snapshots = []
    for i, record in enumerate(sorted_history):
        history_so_far = sorted_history[: i + 1]
        dpd = calc_dpd_asof(record["month_date"], acc["start_date"], acc["emi_due"], history_so_far)
        stage = get_stage(dpd)
        ratio = record["amount_paid"] / acc["emi_due"] if acc["emi_due"] else 0
        snapshots.append({
            "month_date": record["month_date"],
            "amount_paid": record["amount_paid"],
            "payment_ratio": ratio,
            "status": payment_status(record["amount_paid"], acc["emi_due"]),
            "dpd": dpd,
            "stage": stage,
        })
    return snapshots


def compute_8_features(acc, snapshots):
    """Compute the 8 rolling behavioral features from the last 6 months of snapshots."""
    window = snapshots[-6:]  # last up to 6 months

    if len(window) == 0:
        return {
            "dpd_trend": 0.0, "payment_ratio_trend": 0.0, "partial_payment_count": 0,
            "missed_payment_count": 0, "payment_volatility": 0.0, "prior_sma_transitions": 0,
            "account_age_months": months_between(acc["start_date"], date.today()),
            "outstanding_principal_ratio": 1.0,
        }

    dpd_values = [s["dpd"] for s in window]
    ratio_values = [s["payment_ratio"] for s in window]
    x = np.arange(len(window))

    dpd_trend = float(np.polyfit(x, dpd_values, 1)[0]) if len(window) > 1 else 0.0
    payment_ratio_trend = float(np.polyfit(x, ratio_values, 1)[0]) if len(window) > 1 else 0.0
    partial_count = sum(1 for s in window if s["status"] == "Partial")
    missed_count = sum(1 for s in window if s["status"] == "Missed")
    volatility = float(np.var(ratio_values))

    all_stages = [s["stage"] for s in snapshots]
    transitions = sum(1 for i in range(1, len(all_stages)) if all_stages[i] != all_stages[i - 1])

    account_age = months_between(acc["start_date"], date.today())

    total_paid = sum(s["amount_paid"] for s in snapshots)
    outstanding_ratio = max(0.0, (acc["loan_amount"] - total_paid) / acc["loan_amount"])

    return {
        "dpd_trend": round(dpd_trend, 3),
        "payment_ratio_trend": round(payment_ratio_trend, 3),
        "partial_payment_count": partial_count,
        "missed_payment_count": missed_count,
        "payment_volatility": round(volatility, 4),
        "prior_sma_transitions": transitions,
        "account_age_months": account_age,
        "outstanding_principal_ratio": round(outstanding_ratio, 3),
    }


def placeholder_risk_score(features):
    """
    TEMPORARY stand-in until Phase 6's trained Logistic Regression model
    (model.pkl) is wired in. Rough weighted rule, NOT the real model.
    Swap this function's body for model.predict_proba(...) later.
    """
    score = 0.0
    score += min(features["dpd_trend"] / 10, 1.0) * 0.35 if features["dpd_trend"] > 0 else 0
    score += (features["missed_payment_count"] / 6) * 0.30
    score += (features["partial_payment_count"] / 6) * 0.15
    score += min(features["payment_volatility"] * 2, 1.0) * 0.10
    score += min(features["prior_sma_transitions"] / 4, 1.0) * 0.10
    return round(min(score, 1.0), 3)


# ---------- Sidebar: Setup inputs (5 one-time fields) ----------

st.sidebar.header("Add New Account")
new_id = st.sidebar.text_input("Account ID / Borrower Name")
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=1000, value=500000, step=1000)
interest_rate = st.sidebar.slider("Interest Rate (% per annum)", min_value=8.0, max_value=14.0, value=10.5, step=0.1)
start_date = st.sidebar.date_input("Loan Start Date", value=date.today() - timedelta(days=180))
end_date = st.sidebar.date_input("Loan End Date", value=date.today() + timedelta(days=365 * 4))

if st.sidebar.button("Create Account"):
    if not new_id:
        st.sidebar.error("Enter an Account ID")
    elif new_id in st.session_state.accounts:
        st.sidebar.error("This Account ID already exists")
    elif end_date <= start_date:
        st.sidebar.error("End Date must be after Start Date")
    else:
        emi_due, tenure_months = calc_emi(loan_amount, interest_rate, start_date, end_date)
        st.session_state.accounts[new_id] = {
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "start_date": start_date,
            "end_date": end_date,
            "tenure_months": tenure_months,
            "emi_due": emi_due,
            "history": [],
        }
        st.sidebar.success(f"Account {new_id} created — EMI ₹{emi_due:,.0f}/month, tenure {tenure_months} months")

# ---------- Sidebar: Recurring inputs (2 fields, every month) ----------

st.sidebar.markdown("---")
st.sidebar.header("Add Monthly Payment")

if st.session_state.accounts:
    pay_id = st.sidebar.selectbox("Select Account", list(st.session_state.accounts.keys()))
    month_date = st.sidebar.date_input("Payment Date", value=date.today())
    amount_paid = st.sidebar.number_input("Amount Paid (₹)", min_value=0, value=0)

    if st.sidebar.button("Submit Payment"):
        st.session_state.accounts[pay_id]["history"].append({
            "month_date": month_date,
            "amount_paid": amount_paid,
        })
        st.sidebar.success("Payment recorded")
else:
    st.sidebar.info("Create an account first")

# ---------- Main dashboard ----------

st.title("PraSMA — Risk Dashboard")
tab1, tab2 = st.tabs(["Account Detail", "Portfolio Risk List"])

with tab1:
    if not st.session_state.accounts:
        st.info("No accounts yet. Add one from the sidebar.")
    else:
        selected = st.selectbox("Choose Account", list(st.session_state.accounts.keys()))
        acc = st.session_state.accounts[selected]
        snapshots = build_monthly_snapshot(acc)
        dpd = snapshots[-1]["dpd"] if snapshots else 0
        stage = snapshots[-1]["stage"] if snapshots else "Standard"
        features = compute_8_features(acc, snapshots)
        risk_pct = placeholder_risk_score(features)
        priority_score = round(risk_pct * STAGE_WEIGHT[stage], 3)

        st.subheader("Auto-Calculated Loan Terms")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EMI Due", f"₹{acc['emi_due']:,.0f}")
        c2.metric("Tenure", f"{acc['tenure_months']} months")
        c3.metric("Interest Rate", f"{acc['interest_rate']}%")
        c4.metric("Loan Amount", f"₹{acc['loan_amount']:,}")

        st.subheader("Current Risk Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stage", stage)
        c2.metric("DPD", f"{dpd} days")
        c3.metric("Risk % (placeholder)", f"{risk_pct * 100:.1f}%")
        c4.metric("Priority Score", priority_score)

        if stage in ["SMA-1", "SMA-2", "NPA"]:
            st.warning(f"⚠️ Account has slipped to {stage}")

        st.subheader("The 8 Behavioral Features (last 6 months)")
        feat_df = pd.DataFrame([features]).T.rename(columns={0: "Value"})
        st.dataframe(feat_df, use_container_width=True)

        if len(snapshots) > 1:
            st.subheader("DPD Trend")
            trend_df = pd.DataFrame(
                {"DPD": [s["dpd"] for s in snapshots]},
                index=pd.to_datetime([s["month_date"] for s in snapshots]),
            )
            st.line_chart(trend_df)

        if snapshots:
            st.subheader("Payment History")
            hist_df = pd.DataFrame(snapshots)[["month_date", "amount_paid", "status", "dpd", "stage"]]
            st.dataframe(hist_df, use_container_width=True)

with tab2:
    if not st.session_state.accounts:
        st.info("No accounts yet.")
    else:
        rows = []
        for acc_id, acc in st.session_state.accounts.items():
            snapshots = build_monthly_snapshot(acc)
            dpd = snapshots[-1]["dpd"] if snapshots else 0
            stage = snapshots[-1]["stage"] if snapshots else "Standard"
            features = compute_8_features(acc, snapshots)
            risk_pct = placeholder_risk_score(features)
            priority_score = round(risk_pct * STAGE_WEIGHT[stage], 3)
            rows.append({
                "Account": acc_id,
                "Stage": stage,
                "DPD": dpd,
                "Risk % (placeholder)": round(risk_pct * 100, 1),
                "Priority Score": priority_score,
                "Loan Amount": acc["loan_amount"],
            })

        risk_df = pd.DataFrame(rows).sort_values("Priority Score", ascending=False)
        st.dataframe(risk_df, use_container_width=True)