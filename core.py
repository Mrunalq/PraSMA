"""
PraSMA Core Logic
==================
Pure business logic, zero Streamlit dependency. This module is imported by
BOTH app.py (the live dashboard) and generate_data.py / train_model.py
(offline synthetic data + training). That's deliberate: if training-time
feature calculation and live-dashboard feature calculation ever drift apart,
the model's predictions stop matching what it learned from. Importing the
same functions in both places makes that impossible by construction.
"""

from datetime import date
import numpy as np

TOLERANCE = 5.0  # Rupee tolerance for "underpaid" checks (real EMIs have paise, people type rounded amounts)

STAGE_ORDER = ["Standard", "SMA-0", "SMA-1", "SMA-2", "NPA"]

FEATURE_ORDER = [
    "dpd_trend",
    "payment_ratio_trend",
    "partial_payment_count",
    # "missed_payment_count" removed — verified 100% collinear with
    # consecutive_missed_months across all 40,911 training rows (correlation
    # = 1.0, zero differing rows). Root cause: in this data generator, a
    # "Missed" month only ever occurs at the tail of the decline archetypes'
    # trajectory, so the total-miss-count and the trailing-streak-count are
    # structurally forced to agree. Kept consecutive_missed_months instead —
    # it's the strictly more specific signal (a run of consecutive misses is
    # a stronger warning than the same count spread out), and subsumes
    # missed_payment_count as a special case whenever misses are consecutive,
    # which is always true here anyway. Perfectly duplicate features make
    # Logistic Regression's individual coefficients non-identifiable (only
    # their SUM is determined by the data), which would have undermined the
    # explainability bar chart's credibility.
    "payment_volatility",
    "prior_sma_transitions",
    "account_age_months",
    "outstanding_principal_ratio",
    "consecutive_missed_months",
    "max_dpd_last_6m",
]

MIN_MONTHS_FOR_PREDICTION = 3  # cold-start guard — matches the "insufficient history" rule from the design doc


# ---------- Formatting ----------

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


# ---------- Stage classification ----------

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


def stage_rank(stage):
    return STAGE_ORDER.index(stage)


# ---------- Date arithmetic ----------

def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def add_months(d, n):
    """Add n calendar months to date d, clamping the day if the target month
    is shorter (e.g. Jan 31 + 1 month -> Feb 28/29, not an invalid date)."""
    month_index = d.month - 1 + n
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(d.day, days_in_month[month - 1])
    return date(year, month, day)


# ---------- EMI / DPD engine ----------

def calc_emi(loan_amount, interest_rate, start_date, end_date):
    n = months_between(start_date, end_date)
    n = max(n, 1)
    r = (interest_rate / 12) / 100
    if r == 0:
        emi = loan_amount / n
    else:
        emi = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return round(emi, 2), n


def _oldest_unpaid_info(eval_date, start_date, emi_due, history):
    """
    Shared internal logic: given payment history up to eval_date, find the
    OLDEST unpaid installment (FIFO — the exact same ledger logic DPD is
    based on), and return both its due date and the resulting DPD.

    Returns (dpd, oldest_unpaid_date). oldest_unpaid_date is None when the
    account is fully caught up — nothing is "the oldest unpaid" installment
    if there isn't one.
    """
    due_day = start_date.day
    sorted_history = sorted(
        (r for r in history if r["month_date"] <= eval_date),
        key=lambda r: r["month_date"],
    )
    if not sorted_history or emi_due <= 0:
        return 0, None

    cumulative_paid = sum(r["amount_paid"] for r in sorted_history)
    fully_paid_count = int((cumulative_paid + TOLERANCE) // emi_due)

    if fully_paid_count >= len(sorted_history):
        return 0, None

    oldest_unpaid_record = sorted_history[fully_paid_count]
    # month_date was already generated via add_months(), which safely clamps
    # the day for short months (e.g. Jan 31 -> Feb 28/29). Re-deriving it here
    # with a raw date(year, month, due_day) call used to crash with
    # "ValueError: day is out of range for month" whenever due_day (from
    # start_date) didn't exist in that month. Just reuse the valid date.
    oldest_unpaid_date = oldest_unpaid_record["month_date"]
    dpd = max((eval_date - oldest_unpaid_date).days, 0)
    return dpd, oldest_unpaid_date


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
    dpd, _ = _oldest_unpaid_info(eval_date, start_date, emi_due, history)
    return dpd


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
        dpd, oldest_unpaid_date = _oldest_unpaid_info(record["month_date"], acc["start_date"], acc["emi_due"], history_so_far)
        stage = get_stage(dpd)
        ratio = record["amount_paid"] / acc["emi_due"] if acc["emi_due"] else 0

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

        # Total due vs total paid, AS OF this month — this is the running
        # comparison that actually drives DPD/stage above (via the FIFO
        # ledger), made explicit here so it can be shown directly instead of
        # only being implicit in the DPD number.
        cumulative_due = emi_due * (i + 1)
        shortfall_so_far = max(0.0, round(cumulative_due - cumulative_paid, 2))

        snapshots.append({
            "month_date": record["month_date"],
            "amount_paid": record["amount_paid"],
            "payment_ratio": ratio,
            "status": payment_status(record["amount_paid"], acc["emi_due"]),
            "dpd": dpd,
            "stage": stage,
            "month_status": month_status,
            "cumulative_due": round(cumulative_due, 2),
            "cumulative_paid": round(cumulative_paid, 2),
            "shortfall_so_far": shortfall_so_far,
            "oldest_unpaid_date": oldest_unpaid_date,  # None = fully caught up
        })
    return snapshots


# ---------- Feature engineering (the 9 parameters) ----------

def compute_9_features(acc, snapshots, as_of_index=None):
    """
    Compute the 9 rolling behavioral features using snapshots[:as_of_index+1]
    — i.e. history up to and including that month ONLY.

    Originally 10 features. missed_payment_count was removed after being
    found 100% collinear with consecutive_missed_months (correlation = 1.0,
    zero differing rows across 40,911 training rows) — verified empirically,
    not assumed. Root cause: in this data generator, a "Missed" month only
    ever occurs at the tail of the decline archetypes' trajectory, so the
    total-miss-count and the trailing-streak-count are structurally forced
    to agree. Perfectly duplicate features make Logistic Regression's
    individual coefficients non-identifiable — only their SUM is determined
    by the data, so the specific split the solver lands on is arbitrary and
    solver-dependent, undermining the explainability bar chart's credibility.
    consecutive_missed_months was kept over missed_payment_count because it's
    the strictly more specific signal (a consecutive run is a stronger
    warning than the same count spread out) and subsumes the dropped
    feature as a special case whenever misses are consecutive — which, per
    the root cause above, is always true in this generator anyway.

    as_of_index=None (the default, used by the live dashboard) means "use
    everything in `snapshots`" — correct there, since live account history
    never contains future months.

    Training code (generate_data.py) passes an EXPLICIT as_of_index for each
    historical month being evaluated. This matters more than it looks: every
    part of this function — the 6-month window, prior_sma_transitions,
    outstanding_principal_ratio, account_age_months — must only see data up
    to that index. Using the account's FULL trajectory (including months
    that haven't "happened yet" relative to that evaluation point) would leak
    future information into training features, silently inflating reported
    accuracy the same way the account-level train/test leak did before it
    was fixed with GroupShuffleSplit.
    """
    if not snapshots:
        return {f: (0.0 if f not in ("partial_payment_count", "prior_sma_transitions",
                                      "account_age_months", "consecutive_missed_months",
                                      "max_dpd_last_6m") else 0)
                for f in FEATURE_ORDER}

    if as_of_index is None:
        as_of_index = len(snapshots) - 1

    history_so_far = snapshots[: as_of_index + 1]
    window = history_so_far[-6:]
    eval_date = history_so_far[-1]["month_date"]

    dpd_values = [s["dpd"] for s in window]
    ratio_values = [s["payment_ratio"] for s in window]
    x = np.arange(len(window))

    dpd_trend = float(np.polyfit(x, dpd_values, 1)[0]) if len(window) > 1 else 0.0
    payment_ratio_trend = float(np.polyfit(x, ratio_values, 1)[0]) if len(window) > 1 else 0.0
    partial_count = sum(1 for s in window if s["status"] == "Partial")
    volatility = float(np.var(ratio_values))

    # Only stages up to eval point — NOT the account's full future trajectory
    stages_so_far = [s["stage"] for s in history_so_far]
    transitions = sum(1 for i in range(1, len(stages_so_far)) if stages_so_far[i] != stages_so_far[i - 1])

    account_age = months_between(acc["start_date"], eval_date)

    total_paid_so_far = sum(s["amount_paid"] for s in history_so_far)
    outstanding_ratio = max(0.0, (acc["loan_amount"] - total_paid_so_far) / acc["loan_amount"]) if acc["loan_amount"] else 1.0

    consecutive_missed = 0
    for s in reversed(window):
        if s["status"] == "Missed":
            consecutive_missed += 1
        else:
            break

    max_dpd_6m = max((s["dpd"] for s in window), default=0)

    return {
        "dpd_trend": round(dpd_trend, 3),
        "payment_ratio_trend": round(payment_ratio_trend, 3),
        "partial_payment_count": partial_count,
        "payment_volatility": round(volatility, 4),
        "prior_sma_transitions": transitions,
        "account_age_months": account_age,
        "outstanding_principal_ratio": round(outstanding_ratio, 3),
        "consecutive_missed_months": consecutive_missed,
        "max_dpd_last_6m": max_dpd_6m,
    }
