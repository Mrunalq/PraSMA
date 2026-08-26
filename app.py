import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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


def format_inr(amount, decimals=0):
    """Indian digit grouping, e.g. 500000 -> '₹5,00,000', 1000000 -> '₹10,00,000'."""
    if amount is None:
        return "-"
    is_negative = amount < 0
    amount = abs(amount)
    if decimals > 0:
        whole = int(amount)
        frac = round((amount - whole) * (10 ** decimals))
        frac_str = str(frac).zfill(decimals)
    else:
        whole = int(round(amount))
        frac_str = None

    s = str(whole)
    if len(s) <= 3:
        grouped = s
    else:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3

    result = f"₹{grouped}" + (f".{frac_str}" if frac_str else "")
    return f"-{result}" if is_negative else result


def format_date_in(d):
    """dd/mm/yyyy display format."""
    return d.strftime("%d/%m/%Y") if d else "-"


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
    """
    DPD as of a given date, using a cumulative FIFO ledger.

    Instead of checking each month's payment in isolation, we track the
    running total paid vs. running total due. This lets a lump-sum payment
    (e.g. paying 4 months' EMI at once after missing 4 months) correctly
    clear the whole backlog, rather than only being credited to the month
    it was logged against.
    
    "Fully paid installments" = how many EMIs the cumulative payment total
    can fully cover, in FIFO order (oldest first). The next installment
    after that is the oldest unpaid one, and DPD counts from its due date.
    """
    due_day = start_date.day
    sorted_history = sorted(
        (r for r in history if r["month_date"] <= eval_date),
        key=lambda r: r["month_date"],
    )
    if not sorted_history or emi_due <= 0:
        return 0

    cumulative_paid = sum(r["amount_paid"] for r in sorted_history)
    # How many installments does the cumulative payment fully cover (FIFO)?
    fully_paid_count = int((cumulative_paid + TOLERANCE) // emi_due)

    if fully_paid_count >= len(sorted_history):
        # Entire logged backlog is covered by payments so far
        return 0

    oldest_unpaid_record = sorted_history[fully_paid_count]
    oldest_unpaid_date = date(
        oldest_unpaid_record["month_date"].year,
        oldest_unpaid_record["month_date"].month,
        due_day,
    )
    return max((eval_date - oldest_unpaid_date).days, 0)


def payment_status(paid, due):
    if paid <= 0:
        return "Missed"
    elif paid < due - TOLERANCE:
        return "Partial"
    else:
        return "Full"


def is_loan_closed(acc):
    """Loan is fully repaid once cumulative payments cover total EMI x tenure owed."""
    total_payable = acc["emi_due"] * acc["tenure_months"]
    if total_payable <= 0:
        return False
    cumulative_paid = sum(r["amount_paid"] for r in acc["history"])
    return cumulative_paid + TOLERANCE >= total_payable


def build_monthly_snapshot(acc):
    """For each month in history, compute DPD-as-of and stage-as-of that point in time."""
    sorted_history = sorted(acc["history"], key=lambda r: r["month_date"])
    emi_due = acc["emi_due"]
    snapshots = []
    cumulative_paid = 0.0
    for i, record in enumerate(sorted_history):
        history_so_far = sorted_history[: i + 1]
        dpd = calc_dpd_asof(record["month_date"], acc["start_date"], acc["emi_due"], history_so_far)
        stage = get_stage(dpd)
        ratio = record["amount_paid"] / acc["emi_due"] if acc["emi_due"] else 0

        # Which due-month(s) does this payment actually clear, in FIFO order?
        cumulative_before = cumulative_paid
        cumulative_paid += record["amount_paid"]
        fully_before = int((cumulative_before + TOLERANCE) // emi_due) if emi_due else 0
        fully_after = int((cumulative_paid + TOLERANCE) // emi_due) if emi_due else 0
        newly_cleared = list(range(fully_before, min(fully_after, i + 1)))
        if newly_cleared:
            covered = ", ".join(sorted_history[j]["month_date"].strftime("%b %Y") for j in newly_cleared)
            month_status = f"Done: {covered}"
        elif fully_before <= i:
            month_status = f"Pending: {sorted_history[fully_before]['month_date'].strftime('%b %Y')}"
        else:
            month_status = "Advance (ahead of schedule)"

        snapshots.append({
            "month_date": record["month_date"],
            "amount_paid": record["amount_paid"],
            "payment_ratio": ratio,
            "status": payment_status(record["amount_paid"], acc["emi_due"]),
            "dpd": dpd,
            "stage": stage,
            "month_status": month_status,
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
start_date = st.sidebar.date_input("Loan Start Date", value=date.today() - timedelta(days=180), format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Loan End Date", value=date.today() + timedelta(days=365 * 4), format="DD/MM/YYYY")

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
        st.sidebar.success(f"Account {new_id} created — EMI {format_inr(emi_due, decimals=2)}/month, tenure {tenure_months} months")

# ---------- Sidebar: Recurring inputs (2 fields, every month) ----------

st.sidebar.markdown("---")
st.sidebar.header("Add Monthly Payment")

if st.session_state.accounts:
    pay_id = st.sidebar.selectbox("Select Account", list(st.session_state.accounts.keys()))
    month_date = st.sidebar.date_input("Payment Date", value=date.today(), format="DD/MM/YYYY")
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
        c3.metric("Risk % (placeholder)", f"{risk_pct * 100:.1f}%")
        c4.metric("Priority Score", priority_score)

        if stage in ["SMA-1", "SMA-2", "NPA"]:
            st.warning(f"⚠️ Account has slipped to {stage}")

        st.subheader("The 8 Behavioral Features (last 6 months)")
        feat_df = pd.DataFrame([features]).T.rename(columns={0: "Value"})
        st.dataframe(feat_df, use_container_width=True)

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
            st.altair_chart(chart, use_container_width=True)

        if snapshots:
            st.subheader("Payment History")
            hist_df = pd.DataFrame(snapshots)[
                ["month_date", "amount_paid", "status", "month_status", "dpd", "stage"]
            ].rename(columns={"month_status": "Month Status"})
            hist_df["month_date"] = hist_df["month_date"].apply(format_date_in)
            hist_df["amount_paid"] = hist_df["amount_paid"].apply(lambda v: format_inr(v, decimals=2))
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
                "Loan Amount": format_inr(acc["loan_amount"]),
                "Loan Status": "Closed" if is_loan_closed(acc) else "Active",
            })

        risk_df = pd.DataFrame(rows).sort_values("Priority Score", ascending=False)

        stage_options = ["All Stages"] + list(STAGE_WEIGHT.keys())
        selected_stage = st.selectbox("Filter by Stage", stage_options)

        if selected_stage != "All Stages":
            display_df = risk_df[risk_df["Stage"] == selected_stage]
            st.caption(f"Showing {len(display_df)} account(s) in {selected_stage}")
        else:
            display_df = risk_df

        st.dataframe(display_df, use_container_width=True)