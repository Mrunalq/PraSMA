"""
PraSMA Synthetic Data Generator
=================================
Generates labeled training data by simulating account payment histories
across 4 main archetypes, each split into randomized-severity sub-patterns
(11 total), then running every synthetic account through the SAME
build_monthly_snapshot() / compute_9_features() functions the live
dashboard uses (imported from core.py) — never a reimplemented copy.

Output: prasma_training_data.csv — one row per (account, month) with the
9 features, the label (did the account worsen the FOLLOWING month?), and
an account_id column, which GroupShuffleSplit needs at training time to
avoid splitting one account's months across both train and test sets.

NOTE: originally 10 features. missed_payment_count was removed after being
found 100% collinear with consecutive_missed_months (verified empirically —
correlation 1.0, zero differing rows across all 40,911 rows). See core.py's
compute_9_features() docstring for the full root-cause explanation.
"""

import random
import numpy as np
import pandas as pd
from datetime import date

from core import (
    calc_emi, build_monthly_snapshot, compute_9_features,
    FEATURE_ORDER, MIN_MONTHS_FOR_PREDICTION, stage_rank,
)

random.seed(42)
np.random.seed(42)

N_MONTHS = 24            # simulated history length per account
ACCOUNTS_PER_SUBPATTERN = 260   # scaled up to compensate for excluded NPA-stage rows (see below), still targeting ~40k usable training rows


# ---------------------------------------------------------------------------
# 4 main archetypes, 11 sub-patterns (randomized severity within each)
# ---------------------------------------------------------------------------
SUBPATTERNS = [
    # --- 1. STEADY PAYER ---
    {"archetype": "steady_payer", "sub": "1a_perfectly_on_time"},
    {"archetype": "steady_payer", "sub": "1b_mildly_inconsistent"},

    # --- 2. GRADUAL DECLINE ---
    {"archetype": "gradual_decline", "sub": "2a_slow_decline"},
    {"archetype": "gradual_decline", "sub": "2b_moderate_decline"},
    {"archetype": "gradual_decline", "sub": "2c_sharp_decline"},

    # --- 3. SUDDEN DEFAULT ---
    {"archetype": "sudden_default", "sub": "3a_early_default"},
    {"archetype": "sudden_default", "sub": "3b_late_default"},

    # --- 4. OSCILLATING PAYER ---
    {"archetype": "oscillating_payer", "sub": "4a_mild_oscillation"},
    {"archetype": "oscillating_payer", "sub": "4b_wild_oscillation"},
]
# Note: 9 rows above x ~175 accounts ~= 1,575. Two sub-patterns get double
# weight (steady_payer's 2 subs are cheap/fast to generate and under-represent
# "safe" accounts relative to real portfolios) to reach ~1,900 accounts total.
EXTRA_WEIGHT = {"1a_perfectly_on_time": 2, "1b_mildly_inconsistent": 2}


def generate_payment_ratios(sub, n_months):
    """Returns a list of (amount_paid / emi_due) ratios, one per month, shaped by sub-pattern."""

    if sub == "1a_perfectly_on_time":
        return [max(0.0, 1.0 + np.random.normal(0, 0.015)) for _ in range(n_months)]

    if sub == "1b_mildly_inconsistent":
        return [max(0.0, 1.0 + np.random.normal(0, 0.06)) for _ in range(n_months)]

    if sub in ("2a_slow_decline", "2b_moderate_decline", "2c_sharp_decline"):
        if sub == "2a_slow_decline":
            target_min = np.random.uniform(0.5, 0.7)
            decline_span = n_months  # gentle slope across the whole window
        elif sub == "2b_moderate_decline":
            target_min = np.random.uniform(0.3, 0.5)
            decline_span = random.randint(10, 15)
        else:  # 2c_sharp_decline
            target_min = np.random.uniform(0.0, 0.3)
            decline_span = random.randint(5, 8)

        ratios = []
        for m in range(n_months):
            progress = min(m / max(decline_span - 1, 1), 1.0)
            base = 1.0 - (1.0 - target_min) * progress
            ratios.append(max(0.0, base + np.random.normal(0, 0.04)))
        return ratios

    if sub in ("3a_early_default", "3b_late_default"):
        cutover = random.randint(3, 6) if sub == "3a_early_default" else random.randint(14, 20)
        ratios = []
        for m in range(n_months):
            if m < cutover:
                ratios.append(max(0.0, 1.0 + np.random.normal(0, 0.02)))
            else:
                ratios.append(max(0.0, np.random.uniform(0.0, 0.15)))
        return ratios

    if sub in ("4a_mild_oscillation", "4b_wild_oscillation"):
        if sub == "4a_mild_oscillation":
            hi_range, lo_range = (0.85, 1.05), (0.5, 0.75)
        else:
            hi_range, lo_range = (0.85, 1.1), (0.1, 0.4)
        return [
            np.random.uniform(*hi_range) if random.random() < 0.5 else np.random.uniform(*lo_range)
            for _ in range(n_months)
        ]

    raise ValueError(f"Unknown sub-pattern: {sub}")


def generate_one_account(account_id, archetype, sub):
    principal = random.choice([200000, 300000, 500000, 800000, 1200000, 1500000])
    rate = round(random.uniform(8.0, 14.0), 1)
    tenure_years = random.choice([3, 4, 5, 6, 7])
    start_date = date(random.randint(2021, 2024), random.randint(1, 12), random.randint(1, 28))
    end_date = date(start_date.year + tenure_years, start_date.month, start_date.day)

    emi_due, tenure_months = calc_emi(principal, rate, start_date, end_date)

    ratios = generate_payment_ratios(sub, N_MONTHS)
    history = []
    due_date = start_date
    for r in ratios:
        due_date = date(due_date.year + (1 if due_date.month == 12 else 0),
                         1 if due_date.month == 12 else due_date.month + 1,
                         min(start_date.day, 28))
        amount_paid = max(0.0, round(emi_due * r, 2))
        history.append({"month_date": due_date, "amount_paid": amount_paid})

    acc = {
        "account_id": account_id,
        "archetype": archetype,
        "sub_pattern": sub,
        "loan_amount": principal,
        "interest_rate": rate,
        "start_date": start_date,
        "end_date": end_date,
        "tenure_months": tenure_months,
        "emi_due": emi_due,
        "history": history,
    }
    return acc


def build_training_table():
    X_rows, y_rows, account_ids, archetypes, subs, stage_at_rows = [], [], [], [], [], []
    account_counter = 0
    accounts_generated = 0

    for pattern in SUBPATTERNS:
        n_accounts = ACCOUNTS_PER_SUBPATTERN * EXTRA_WEIGHT.get(pattern["sub"], 1)
        for _ in range(n_accounts):
            account_counter += 1
            acc_id = f"SYN-{account_counter:05d}"
            acc = generate_one_account(acc_id, pattern["archetype"], pattern["sub"])
            snapshots = build_monthly_snapshot(acc)
            accounts_generated += 1

            # Walk every month that has enough history AND a following month to label against
            for i in range(MIN_MONTHS_FOR_PREDICTION - 1, len(snapshots) - 1):
                current_stage_i = snapshots[i]["stage"]

                # Skip NPA-stage rows entirely: NPA is already the worst stage,
                # so label would ALWAYS be 0 (nothing "worse" to predict toward).
                # These rows carry zero learning signal and only dilute the
                # dataset — consistent with the design decision that NPA
                # accounts sit in a separate, unranked reference table in the
                # dashboard and never get a risk_probability computed at all.
                if current_stage_i == "NPA":
                    continue

                feats = compute_9_features(acc, snapshots, as_of_index=i)
                current_rank = stage_rank(current_stage_i)
                next_rank = stage_rank(snapshots[i + 1]["stage"])
                label = 1 if next_rank > current_rank else 0

                X_rows.append([feats[f] for f in FEATURE_ORDER])
                y_rows.append(label)
                account_ids.append(acc_id)
                archetypes.append(pattern["archetype"])
                subs.append(pattern["sub"])
                stage_at_rows.append(current_stage_i)

    X = np.array(X_rows)
    y = np.array(y_rows)

    df = pd.DataFrame(X, columns=FEATURE_ORDER)
    df["label_worsens_next_month"] = y
    df["account_id"] = account_ids
    df["archetype"] = archetypes
    df["sub_pattern"] = subs
    df["current_stage"] = stage_at_rows

    return df, accounts_generated


if __name__ == "__main__":
    df, n_accounts = build_training_table()

    print(f"Synthetic accounts generated : {n_accounts}")
    print(f"Training rows (account-months): {len(df)}")
    print(f"Positive label rate (worsens) : {df['label_worsens_next_month'].mean():.3f}")
    print()
    print("Rows per archetype:")
    print(df.groupby("archetype").size().to_string())
    print()
    print("Rows per sub-pattern:")
    print(df.groupby("sub_pattern").size().to_string())
    print()
    print("Rows per current_stage (before the predicted transition):")
    print(df.groupby("current_stage").size().to_string())
    print()
    print("Positive label rate BY archetype (sanity check — decline/default/oscillating")
    print("should show a clearly higher worsening rate than steady_payer):")
    print(df.groupby("archetype")["label_worsens_next_month"].mean().round(3).to_string())

    df.to_csv("prasma_training_data.csv", index=False)
    print("\nSaved: prasma_training_data.csv")
