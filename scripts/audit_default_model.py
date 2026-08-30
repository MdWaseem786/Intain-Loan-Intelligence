from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN = DATA / "loan_monthly_performance_train.csv"
TEST = DATA / "loan_monthly_performance_test.csv"
HOLDOUT = DATA / "test_targets_holdout.csv"

TARGET = "next_12m_default_flag"
PERIOD = "monthly_reporting_period"

MODEL_DIR = MODELS / TARGET


def parse_period(s):
    values = s.astype(str).str.strip()

    result = pd.to_datetime(
        values,
        format="%Y-%m",
        errors="coerce",
    )

    missing = result.isna()

    if missing.any():
        result.loc[missing] = pd.to_datetime(
            values.loc[missing],
            format="%m%Y",
            errors="coerce",
        )

    return result


def get_features(df):
    forbidden = {
        "loan_id",
        "monthly_reporting_period",
    }

    features = []

    for col in df.columns:
        if col in forbidden:
            continue

        lower = col.lower()

        if (
            "next_" in lower
            or "target" in lower
            or "censor" in lower
            or "unknown_status_present" in lower
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    features = [
        c for c in features
        if not any(
            token in c.lower()
            for token in ["loan_id", "zip", "msa"]
        )
    ]

    features = [
        c for c in features
        if df[c].notna().any()
    ]

    return features


def ranking_metrics(y, probability):
    order = np.argsort(-probability)
    y_sorted = np.asarray(y)[order]

    n = len(y_sorted)

    result = {}

    for pct in [0.01, 0.05, 0.10]:

        k = max(1, int(n * pct))

        top = y_sorted[:k]

        precision = top.mean()
        recall = top.sum() / max(1, y_sorted.sum())

        result[f"precision_at_{int(pct * 100)}pct"] = float(
            precision
        )

        result[f"recall_at_{int(pct * 100)}pct"] = float(
            recall
        )

    return result


def evaluate(name, y, probability):

    baseline_rate = np.mean(y)

    result = {
        "dataset": name,
        "rows": len(y),
        "positive_events": int(np.sum(y)),
        "positive_rate": float(baseline_rate),
        "roc_auc": float(
            roc_auc_score(y, probability)
        ),
        "pr_auc": float(
            average_precision_score(y, probability)
        ),
    }

    result.update(
        ranking_metrics(
            y,
            probability,
        )
    )

    return result


def main():

    print("=" * 70)
    print("12-MONTH DEFAULT MODEL AUDIT")
    print("=" * 70)

    print("\nLoading datasets...")

    train = pd.read_csv(
        TRAIN,
        low_memory=False,
    )

    test = pd.read_csv(
        TEST,
        low_memory=False,
    )

    holdout = pd.read_csv(
        HOLDOUT,
        low_memory=False,
    )

    train["_period"] = parse_period(
        train[PERIOD]
    )

    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Holdout rows: {len(holdout):,}")

    print("\nTarget distribution:")

    print(
        train[TARGET]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nTraining-period distribution:")

    training_mask = (
        (train["_period"] < "2016-01-01")
        & train[TARGET].notna()
    )

    validation_mask = (
        (train["_period"] >= "2016-01-01")
        & (train["_period"] <= "2017-12-01")
        & train[TARGET].notna()
    )

    print(
        f"Model training: "
        f"{training_mask.sum():,}"
    )

    print(
        f"Validation: "
        f"{validation_mask.sum():,}"
    )

    print(
        f"Validation positives: "
        f"{train.loc[validation_mask, TARGET].sum():,}"
    )

    print(
        f"Validation rate: "
        f"{train.loc[validation_mask, TARGET].mean():.6%}"
    )

    # ------------------------------------------------------------
    # Load existing model
    # ------------------------------------------------------------

    model_path = (
        MODEL_DIR / "model.joblib"
    )

    metadata_path = (
        MODEL_DIR / "metadata.json"
    )

    print("\nLoading existing model...")

    model = joblib.load(
        model_path
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]

    print(
        f"Model: {metadata['model']}"
    )

    print(
        f"Features: {len(features)}"
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    X_val = train.loc[
        validation_mask,
        features
    ]

    y_val = train.loc[
        validation_mask,
        TARGET
    ].astype(int)

    print("\nGenerating validation probabilities...")

    p_val = model.predict_proba(
        X_val
    )[:, 1]

    validation_result = evaluate(
        "validation_2016_2017",
        y_val,
        p_val,
    )

    # ------------------------------------------------------------
    # Holdout
    # ------------------------------------------------------------

    print("\nJoining holdout targets to test features...")

    holdout_eval = test[
        [
            "loan_id",
            PERIOD,
        ] + features
    ].merge(
        holdout[
            [
                "loan_id",
                PERIOD,
                TARGET,
            ]
        ],
        on=[
            "loan_id",
            PERIOD,
        ],
        how="inner",
        validate="one_to_one",
    )

    holdout_mask = (
        holdout_eval[TARGET].notna()
    )

    X_holdout = holdout_eval.loc[
        holdout_mask,
        features
    ]

    y_holdout = holdout_eval.loc[
        holdout_mask,
        TARGET
    ].astype(int)

    print(
        f"Holdout evaluable rows: "
        f"{len(y_holdout):,}"
    )

    print(
        f"Holdout positives: "
        f"{y_holdout.sum():,}"
    )

    print(
        f"Holdout positive rate: "
        f"{y_holdout.mean():.6%}"
    )

    print("\nGenerating holdout probabilities...")

    p_holdout = model.predict_proba(
        X_holdout
    )[:, 1]

    holdout_result = evaluate(
        "holdout_2018_2025",
        y_holdout,
        p_holdout,
    )

    # ------------------------------------------------------------
    # Probability distributions
    # ------------------------------------------------------------

    distribution = {}

    for name, probability in [
        ("validation", p_val),
        ("holdout", p_holdout),
    ]:

        distribution[name] = {
            "min": float(np.min(probability)),
            "p50": float(np.percentile(probability, 50)),
            "p90": float(np.percentile(probability, 90)),
            "p95": float(np.percentile(probability, 95)),
            "p99": float(np.percentile(probability, 99)),
            "max": float(np.max(probability)),
            "mean": float(np.mean(probability)),
        }

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    for k, v in validation_result.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 70)
    print("HOLDOUT")
    print("=" * 70)

    for k, v in holdout_result.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTIONS")
    print("=" * 70)

    for dataset, values in distribution.items():

        print(f"\n{dataset}")

        for k, v in values.items():
            print(
                f"{k}: {v:.8f}"
            )

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    report = {
        "target": TARGET,
        "model": metadata["model"],
        "features": features,
        "validation": validation_result,
        "holdout": holdout_result,
        "probability_distribution": distribution,
    }

    REPORTS.mkdir(
        exist_ok=True
    )

    output = (
        REPORTS /
        "default_model_audit.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        f"\nAudit saved: {output}"
    )

    print("\nDEFAULT MODEL AUDIT COMPLETE")


if __name__ == "__main__":
    main()