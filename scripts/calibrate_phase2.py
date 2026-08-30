from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
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


def calibration_stats(y, p, name):

    frac, mean_pred = calibration_curve(
        y,
        p,
        n_bins=10,
        strategy="quantile",
    )

    return {
        "method": name,
        "brier": float(
            brier_score_loss(y, p)
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "roc_auc": float(
            roc_auc_score(y, p)
        ),
        "pr_auc": float(
            average_precision_score(y, p)
        ),
        "mean_probability": float(
            np.mean(p)
        ),
        "actual_rate": float(
            np.mean(y)
        ),
        "calibration_bins": [
            {
                "predicted": float(pred),
                "actual": float(actual),
            }
            for pred, actual
            in zip(mean_pred, frac)
        ],
    }


def main():

    print("=" * 70)
    print("12-MONTH DEFAULT PROBABILITY CALIBRATION")
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

    # ------------------------------------------------------------
    # Existing model metadata
    # ------------------------------------------------------------

    model_dir = (
        MODELS / TARGET
    )

    model = joblib.load(
        model_dir / "model.joblib"
    )

    with open(
        model_dir / "metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]

    # ------------------------------------------------------------
    # Chronological validation
    # ------------------------------------------------------------

    validation_mask = (
        (train["_period"] >= "2016-01-01")
        & (train["_period"] <= "2017-12-01")
        & train[TARGET].notna()
    )

    X_val = train.loc[
        validation_mask,
        features
    ]

    y_val = train.loc[
        validation_mask,
        TARGET
    ].astype(int)

    print(
        f"Validation rows: {len(y_val):,}"
    )

    print(
        f"Validation defaults: {y_val.sum():,}"
    )

    # Existing model probabilities
    p_val = model.predict_proba(
        X_val
    )[:, 1]

    print(
        "\nRaw model calibration:"
    )

    raw = calibration_stats(
        y_val,
        p_val,
        "raw",
    )

    print(
        f"Brier: {raw['brier']:.6f}"
    )

    print(
        f"Log loss: {raw['log_loss']:.6f}"
    )

    print(
        f"ROC-AUC: {raw['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC: {raw['pr_auc']:.6f}"
    )

    # ------------------------------------------------------------
    # Platt / sigmoid calibration
    # ------------------------------------------------------------

    print(
        "\nFitting sigmoid calibration..."
    )

    sigmoid = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    # Logistic regression on log-odds is equivalent to
    # Platt-style probability calibration.
    eps = 1e-7

    logit_val = np.log(
        np.clip(
            p_val,
            eps,
            1 - eps,
        )
        /
        np.clip(
            1 - p_val,
            eps,
            1 - eps,
        )
    ).reshape(-1, 1)

    sigmoid.fit(
        logit_val,
        y_val,
    )

    p_sigmoid_val = sigmoid.predict_proba(
        logit_val
    )[:, 1]

    sigmoid_stats = calibration_stats(
        y_val,
        p_sigmoid_val,
        "sigmoid",
    )

    print(
        f"Sigmoid Brier: "
        f"{sigmoid_stats['brier']:.6f}"
    )

    print(
        f"Sigmoid Log loss: "
        f"{sigmoid_stats['log_loss']:.6f}"
    )

    # ------------------------------------------------------------
    # Isotonic calibration
    # ------------------------------------------------------------

    print(
        "\nFitting isotonic calibration..."
    )

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        p_val,
        y_val,
    )

    p_iso_val = isotonic.predict(
        p_val
    )

    isotonic_stats = calibration_stats(
        y_val,
        p_iso_val,
        "isotonic",
    )

    print(
        f"Isotonic Brier: "
        f"{isotonic_stats['brier']:.6f}"
    )

    print(
        f"Isotonic Log loss: "
        f"{isotonic_stats['log_loss']:.6f}"
    )

    # ------------------------------------------------------------
    # Choose calibration based ONLY on validation
    # ------------------------------------------------------------

    candidates = [
        raw,
        sigmoid_stats,
        isotonic_stats,
    ]

    selected = min(
        candidates,
        key=lambda x: (
            x["brier"],
            x["log_loss"],
        ),
    )

    print(
        "\nSelected calibration:"
    )

    print(
        selected["method"]
    )

    # ------------------------------------------------------------
    # Fit selected calibration
    # ------------------------------------------------------------

    if selected["method"] == "raw":

        calibrator = None

    elif selected["method"] == "sigmoid":

        calibrator = sigmoid

    else:

        calibrator = isotonic

    # ------------------------------------------------------------
    # Evaluate holdout
    # ------------------------------------------------------------

    print(
        "\nJoining holdout targets..."
    )

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

    holdout_eval = holdout_eval[
        holdout_eval[TARGET].notna()
    ]

    X_holdout = holdout_eval[
        features
    ]

    y_holdout = holdout_eval[
        TARGET
    ].astype(int)

    p_holdout_raw = model.predict_proba(
        X_holdout
    )[:, 1]

    if calibrator is None:

        p_holdout_calibrated = (
            p_holdout_raw
        )

    elif selected["method"] == "sigmoid":

        logit_holdout = np.log(
            np.clip(
                p_holdout_raw,
                eps,
                1 - eps,
            )
            /
            np.clip(
                1 - p_holdout_raw,
                eps,
                1 - eps,
            )
        ).reshape(-1, 1)

        p_holdout_calibrated = (
            calibrator.predict_proba(
                logit_holdout
            )[:, 1]
        )

    else:

        p_holdout_calibrated = (
            calibrator.predict(
                p_holdout_raw
            )
        )

    holdout_raw = calibration_stats(
        y_holdout,
        p_holdout_raw,
        "holdout_raw",
    )

    holdout_calibrated = calibration_stats(
        y_holdout,
        p_holdout_calibrated,
        "holdout_calibrated",
    )

    print(
        "\nHoldout raw:"
    )

    print(
        f"Brier: "
        f"{holdout_raw['brier']:.6f}"
    )

    print(
        f"Log loss: "
        f"{holdout_raw['log_loss']:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{holdout_raw['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{holdout_raw['pr_auc']:.6f}"
    )

    print(
        "\nHoldout calibrated:"
    )

    print(
        f"Brier: "
        f"{holdout_calibrated['brier']:.6f}"
    )

    print(
        f"Log loss: "
        f"{holdout_calibrated['log_loss']:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{holdout_calibrated['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{holdout_calibrated['pr_auc']:.6f}"
    )

    # ------------------------------------------------------------
    # Save calibration artifacts
    # ------------------------------------------------------------

    calibration_dir = (
        MODELS /
        "phase2_5" /
        TARGET
    )

    calibration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if calibrator is not None:

        joblib.dump(
            calibrator,
            calibration_dir /
            "calibrator.joblib",
        )

    results = {
        "target": TARGET,
        "selected_calibration": selected["method"],
        "validation": {
            "raw": raw,
            "sigmoid": sigmoid_stats,
            "isotonic": isotonic_stats,
        },
        "holdout": {
            "raw": holdout_raw,
            "calibrated": holdout_calibrated,
        },
    }

    with open(
        calibration_dir /
        "calibration_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Save calibrated test predictions
    # ------------------------------------------------------------

    p_test_raw = model.predict_proba(
        test[features]
    )[:, 1]

    if calibrator is None:

        p_test_calibrated = p_test_raw

    elif selected["method"] == "sigmoid":

        logit_test = np.log(
            np.clip(
                p_test_raw,
                eps,
                1 - eps,
            )
            /
            np.clip(
                1 - p_test_raw,
                eps,
                1 - eps,
            )
        ).reshape(-1, 1)

        p_test_calibrated = (
            calibrator.predict_proba(
                logit_test
            )[:, 1]
        )

    else:

        p_test_calibrated = (
            calibrator.predict(
                p_test_raw
            )
        )

    prediction_path = (
        DATA /
        "default_probability_calibrated.csv"
    )

    pd.DataFrame({
        "loan_id": test["loan_id"],
        PERIOD: test[PERIOD],
        "default_probability_raw": p_test_raw,
        "default_probability_calibrated":
            p_test_calibrated,
    }).to_csv(
        prediction_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = [
        "# 12-Month Default Probability Calibration",
        "",
        "## Validation",
        "",
        f"- Raw Brier: {raw['brier']:.6f}",
        f"- Sigmoid Brier: {sigmoid_stats['brier']:.6f}",
        f"- Isotonic Brier: {isotonic_stats['brier']:.6f}",
        "",
        f"- Raw Log Loss: {raw['log_loss']:.6f}",
        f"- Sigmoid Log Loss: {sigmoid_stats['log_loss']:.6f}",
        f"- Isotonic Log Loss: {isotonic_stats['log_loss']:.6f}",
        "",
        f"Selected calibration: **{selected['method']}**",
        "",
        "## Holdout",
        "",
        f"- Raw Brier: {holdout_raw['brier']:.6f}",
        f"- Calibrated Brier: {holdout_calibrated['brier']:.6f}",
        f"- Raw Log Loss: {holdout_raw['log_loss']:.6f}",
        f"- Calibrated Log Loss: {holdout_calibrated['log_loss']:.6f}",
        f"- Raw ROC-AUC: {holdout_raw['roc_auc']:.6f}",
        f"- Calibrated ROC-AUC: {holdout_calibrated['roc_auc']:.6f}",
        f"- Raw PR-AUC: {holdout_raw['pr_auc']:.6f}",
        f"- Calibrated PR-AUC: {holdout_calibrated['pr_auc']:.6f}",
    ]

    (
        REPORTS /
        "phase2_5_calibration_report.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        "\nCalibration artifacts saved."
    )

    print(
        "\nPHASE 2.5 CALIBRATION COMPLETE"
    )


if __name__ == "__main__":
    main()