from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TRAIN = DATA / "loan_monthly_performance_train.csv"
TEST = DATA / "loan_monthly_performance_test.csv"
HOLDOUT = DATA / "test_targets_holdout.csv"

TARGET = "next_12m_default_flag"
PERIOD = "monthly_reporting_period"


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

    excluded = {
        "loan_id",
        PERIOD,
    }

    features = []

    for col in df.columns:

        if col in excluded:
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
            for token in ["zip", "msa"]
        )
    ]

    features = [
        c for c in features
        if df[c].notna().any()
    ]

    return features


def numeric_shift(a, b):

    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")

    return {
        "train_mean": float(a.mean()),
        "holdout_mean": float(b.mean()),
        "train_median": float(a.median()),
        "holdout_median": float(b.median()),
        "train_missing_pct": float(a.isna().mean() * 100),
        "holdout_missing_pct": float(b.isna().mean() * 100),
    }


def main():

    print("=" * 70)
    print("TEMPORAL DISTRIBUTION SHIFT AUDIT")
    print("=" * 70)

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

    validation_mask = (
        (train["_period"] >= "2016-01-01")
        & (train["_period"] <= "2017-12-01")
    )

    validation = train.loc[
        validation_mask
    ].copy()

    holdout_eval = test.merge(
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

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Holdout rows: "
        f"{len(holdout_eval):,}"
    )

    print(
        f"\nValidation default rate: "
        f"{validation[TARGET].mean():.6%}"
    )

    print(
        f"Holdout default rate: "
        f"{holdout_eval[TARGET].mean():.6%}"
    )

    features = get_features(
        train
    )

    rows = []

    for feature in features:

        a = validation[feature]
        b = holdout_eval[feature]

        if pd.api.types.is_numeric_dtype(a):

            stats = numeric_shift(
                a,
                b,
            )

            train_values = pd.to_numeric(
                a,
                errors="coerce",
            ).dropna()

            holdout_values = pd.to_numeric(
                b,
                errors="coerce",
            ).dropna()

            if len(train_values) > 10 and len(holdout_values) > 10:

                # Build a classifier that tries to distinguish
                # validation observations from holdout observations.
                x = np.concatenate([
                    train_values.to_numpy(),
                    holdout_values.to_numpy(),
                ])

                y = np.concatenate([
                    np.zeros(len(train_values)),
                    np.ones(len(holdout_values)),
                ])

                order = np.argsort(x)

                x_sorted = x[order]
                y_sorted = y[order]

                # Simple empirical distribution separation:
                # maximum difference between cumulative distributions.
                cdf_a = np.searchsorted(
                    np.sort(train_values),
                    x_sorted,
                    side="right",
                ) / len(train_values)

                cdf_b = np.searchsorted(
                    np.sort(holdout_values),
                    x_sorted,
                    side="right",
                ) / len(holdout_values)

                ks = float(
                    np.max(
                        np.abs(
                            cdf_a - cdf_b
                        )
                    )
                )

            else:
                ks = np.nan

            rows.append({
                "feature": feature,
                "type": "numeric",
                "shift_score_ks": ks,
                **stats,
            })

    result = pd.DataFrame(
        rows
    ).sort_values(
        "shift_score_ks",
        ascending=False,
    )

    REPORTS.mkdir(
        exist_ok=True
    )

    output = (
        REPORTS /
        "temporal_shift_audit.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    print("\nTop distribution shifts:")

    print(
        result.head(15).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output}"
    )

    report = []

    report.append(
        "# Temporal Distribution Shift Audit\n"
    )

    report.append(
        f"- Validation period: 2016-01 to 2017-12\n"
        f"- Validation rows: {len(validation):,}\n"
        f"- Holdout rows: {len(holdout_eval):,}\n"
        f"- Validation default rate: {validation[TARGET].mean():.6%}\n"
        f"- Holdout default rate: {holdout_eval[TARGET].mean():.6%}\n"
    )

    report.append(
        "\n## Largest numeric distribution shifts\n"
    )

    report.append(
    result.head(20).to_string(
        index=False
    )
)

    (
        REPORTS /
        "temporal_shift_audit.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        "\nTEMPORAL SHIFT AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()