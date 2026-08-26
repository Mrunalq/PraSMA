"""
PraSMA Model Training
=======================
Loads prasma_training_data.csv (from generate_data.py), splits by
account_id (GroupShuffleSplit) so no account's months leak across both
train and test, trains a Logistic Regression, evaluates honestly on the
held-out accounts, and saves model.pkl for the dashboard to load.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)

from core import FEATURE_ORDER

RANDOM_STATE = 42


def main():
    df = pd.read_csv("prasma_training_data.csv")

    # ---- Target-leakage guard ----
    # archetype, sub_pattern, and current_stage are GENERATOR METADATA — they
    # tag which synthetic behavior pattern produced each row, assigned by
    # generate_data.py after simulating an account's FULL 24-month
    # trajectory. A real bank account, observed live, has no equivalent tag —
    # nothing tells you at Month 3 that "this account was built to simulate
    # gradual_decline." If these columns were ever fed to the model, they'd
    # act as a near-perfect, disguised copy of the label (since the label
    # itself was derived from the same simulated trajectory), producing
    # excellent-looking evaluation metrics that are meaningless in practice.
    # account_id is excluded from X for a related but separate reason: it's
    # an identifier, not a behavioral signal, and is used below ONLY for
    # GroupShuffleSplit (a different leakage fix — preventing one account's
    # months from appearing in both train and test).
    #
    # FEATURE_ORDER (from core.py) already excludes all of these by being an
    # explicit whitelist, not "everything except the label" — but that
    # correctness previously depended on FEATURE_ORDER never being edited
    # carelessly. This assertion makes it a hard failure instead of a silent
    # assumption, and the explicit drop below makes the intent unmistakable
    # to anyone reading this script later.
    LEAKY_COLUMNS = {"archetype", "sub_pattern", "current_stage", "account_id", "label_worsens_next_month"}
    assert set(FEATURE_ORDER).isdisjoint(LEAKY_COLUMNS), (
        "FEATURE_ORDER must never include generator metadata or the label — "
        f"found overlap: {set(FEATURE_ORDER) & LEAKY_COLUMNS}"
    )

    df_model = df.drop(columns=["archetype", "sub_pattern", "current_stage"])
    X = df_model[FEATURE_ORDER].to_numpy()
    y = df_model["label_worsens_next_month"].to_numpy()
    groups = df_model["account_id"].to_numpy()  # used only for GroupShuffleSplit below, never as a feature

    print(f"Total rows: {len(df)}  |  Unique accounts: {df['account_id'].nunique()}")
    print(f"Overall positive rate: {y.mean():.3f}")
    print(f"Confirmed excluded from training: {sorted(LEAKY_COLUMNS - {'label_worsens_next_month'})}\n")

    # Split by ACCOUNT, not by row — every row from one account goes entirely
    # into train or entirely into test, never both. This is the fix for the
    # train/test leakage flaw identified earlier.
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_accounts = set(groups[train_idx])
    test_accounts = set(groups[test_idx])
    overlap = train_accounts & test_accounts
    print(f"Train accounts: {len(train_accounts)}  |  Test accounts: {len(test_accounts)}")
    print(f"Account overlap between train/test: {len(overlap)} (must be 0)\n")
    assert len(overlap) == 0, "Leakage check failed — an account appears in both sets"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("=" * 60)
    print("EVALUATION ON HELD-OUT ACCOUNTS (never seen in training)")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted Safe   Predicted Risky")
    print(f"  Actual Safe    {cm[0][0]:>13}   {cm[0][1]:>15}")
    print(f"  Actual Risky   {cm[1][0]:>13}   {cm[1][1]:>15}")

    print(f"\nAccuracy    : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision   : {precision_score(y_test, y_pred):.3f}")
    print(f"Recall (Sensitivity): {recall_score(y_test, y_pred):.3f}  <- prioritized metric")
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"Specificity : {specificity:.3f}")
    print(f"F1-Score    : {f1_score(y_test, y_pred):.3f}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Worsens (1)"]))

    print("Learned coefficients (feature -> weight):")
    for name, coef in sorted(zip(FEATURE_ORDER, model.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:30s} {coef:+.4f}")

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_order": FEATURE_ORDER}, f)
    print("\nSaved model.pkl")


if __name__ == "__main__":
    main()
