from pathlib import Path
import json
import time
import platform

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    balanced_accuracy_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN = DATA / "loan_monthly_performance_train.csv"
TEST = DATA / "loan_monthly_performance_test.csv"
HOLDOUT = DATA / "test_targets_holdout.csv"

SEED = 42

BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]

PROBABILITY_NAMES = {
    "next_3m_delinquency_flag": "delinquency_3m_probability",
    "next_6m_delinquency_flag": "delinquency_6m_probability",
    "next_12m_default_flag": "default_12m_probability",
    "next_12m_prepayment_flag": "prepayment_12m_probability",
}


def period_to_date(s):
    values = s.astype(str).str.strip()

    # Phase 1B may contain YYYY-MM
    result = pd.to_datetime(
        values,
        format="%Y-%m",
        errors="coerce"
    )

    # Fannie Mae source format: MMYYYY
    missing = result.isna()

    if missing.any():
        result.loc[missing] = pd.to_datetime(
            values.loc[missing],
            format="%m%Y",
            errors="coerce"
        )

    return result


def metric_binary(y, p, threshold):
    pred = (p >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def choose_threshold(y, p):
    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.arange(0.01, 1.00, 0.01):
        score = f1_score(
            y,
            (p >= threshold).astype(int),
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def make_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )),
    ])


def make_hgb():
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=SEED,
        )),
    ])


def get_features(df):
    forbidden = {
        "loan_id",
        "monthly_reporting_period",
        "_period",
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

    # Avoid numeric identifiers/high-cardinality location identifiers
    features = [
        c for c in features
        if not any(
            token in c.lower()
            for token in ["loan_id", "zip", "msa"]
        )
    ]

    features = [c for c in features if df[c].notna().any()]

    return features


def train_binary(train, test, holdout, features, target):

    print(f"\n{'=' * 70}")
    print(f"TARGET: {target}")
    print(f"{'=' * 70}")

    # Chronological validation:
    # training: before 2016
    # validation: 2016-2017
    train_mask = (
        (train["_period"] < "2016-01-01")
        & train[target].notna()
    )

    val_mask = (
        (train["_period"] >= "2016-01-01")
        & (train["_period"] <= "2017-12-01")
        & train[target].notna()
    )

    X_train = train.loc[train_mask, features]
    y_train = train.loc[train_mask, target].astype(int)

    X_val = train.loc[val_mask, features]
    y_val = train.loc[val_mask, target].astype(int)

    print(f"Training rows:   {len(y_train):,}")
    print(f"Training events: {y_train.sum():,}")
    print(f"Validation rows: {len(y_val):,}")
    print(f"Validation events:{y_val.sum():,}")

    # ------------------------------------------------------------
    # Logistic baseline
    # ------------------------------------------------------------

    print("Training Logistic Regression...")
    start = time.time()

    logistic = make_logistic()
    logistic.fit(X_train, y_train)

    logistic_probability = logistic.predict_proba(X_val)[:, 1]

    logistic_threshold = choose_threshold(
        y_val,
        logistic_probability,
    )

    logistic_metrics = metric_binary(
        y_val,
        logistic_probability,
        logistic_threshold,
    )

    logistic_time = time.time() - start

    # ------------------------------------------------------------
    # HistGradientBoosting
    # ------------------------------------------------------------

    print("Training HistGradientBoosting...")
    start = time.time()

    hgb = make_hgb()
    hgb.fit(X_train, y_train)

    hgb_probability = hgb.predict_proba(X_val)[:, 1]

    hgb_threshold = choose_threshold(
        y_val,
        hgb_probability,
    )

    hgb_metrics = metric_binary(
        y_val,
        hgb_probability,
        hgb_threshold,
    )

    hgb_time = time.time() - start

    print(
        f"Logistic: PR-AUC={logistic_metrics['pr_auc']:.4f}, "
        f"ROC-AUC={logistic_metrics['roc_auc']:.4f}, "
        f"F1={logistic_metrics['f1']:.4f}"
    )

    print(
        f"HistGB:   PR-AUC={hgb_metrics['pr_auc']:.4f}, "
        f"ROC-AUC={hgb_metrics['roc_auc']:.4f}, "
        f"F1={hgb_metrics['f1']:.4f}"
    )

    # Select using validation PR-AUC.
    if hgb_metrics["pr_auc"] >= logistic_metrics["pr_auc"]:
        model_name = "HistGradientBoosting"
        selected_model = hgb
        threshold = hgb_threshold
        validation_metrics = hgb_metrics
        training_time = hgb_time
    else:
        model_name = "LogisticRegression"
        selected_model = logistic
        threshold = logistic_threshold
        validation_metrics = logistic_metrics
        training_time = logistic_time

    print(f"Selected model: {model_name}")
    print(f"Selected threshold: {threshold:.4f}")

    # ------------------------------------------------------------
    # Final fit using ALL eligible training observations <= 2017
    # ------------------------------------------------------------

    full_mask = train[target].notna()

    X_full = train.loc[full_mask, features]
    y_full = train.loc[full_mask, target].astype(int)

    print("Refitting selected model on full training data...")
    selected_model.fit(X_full, y_full)

    # ------------------------------------------------------------
    # Holdout evaluation
    # ------------------------------------------------------------

    # Holdout targets are stored separately from test features.
    # Join them on the exact loan-month key before evaluation.
    holdout_eval = test[
        ["loan_id", "monthly_reporting_period"] + features
    ].merge(
        holdout[
            ["loan_id", "monthly_reporting_period", target]
        ],
        on=["loan_id", "monthly_reporting_period"],
        how="inner",
        validate="one_to_one",
    )

    hold_mask = holdout_eval[target].notna()

    X_hold = holdout_eval.loc[hold_mask, features]
    y_hold = holdout_eval.loc[hold_mask, target].astype(int)

    hold_probability = selected_model.predict_proba(X_hold)[:, 1]

    hold_metrics = metric_binary(
        y_hold,
        hold_probability,
        threshold,
    )

    print(
        f"HOLDOUT: PR-AUC={hold_metrics['pr_auc']:.4f}, "
        f"ROC-AUC={hold_metrics['roc_auc']:.4f}, "
        f"F1={hold_metrics['f1']:.4f}, "
        f"Brier={hold_metrics['brier']:.4f}"
    )

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    model_dir = MODELS / target
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        selected_model,
        model_dir / "model.joblib",
    )

    metadata = {
        "target": target,
        "model": model_name,
        "features": features,
        "threshold": threshold,
        "validation_period": "2016-01 to 2017-12",
        "final_training_cutoff": "2017-12",
        "seed": SEED,
        "validation_metrics": validation_metrics,
        "holdout_metrics": hold_metrics,
        "training_seconds": training_time,
    }

    with open(
        model_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    # Test predictions
    test_probability = selected_model.predict_proba(
        test[features]
    )[:, 1]

    return {
        "model": model_name,
        "threshold": threshold,
        "validation": validation_metrics,
        "holdout": hold_metrics,
        "probability": test_probability,
        "training_seconds": training_time,
    }


def train_next_state(train, test, holdout, features):

    target = "next_state"

    print(f"\n{'=' * 70}")
    print("TARGET: next_state")
    print(f"{'=' * 70}")

    train_mask = (
        (train["_period"] < "2016-01-01")
        & train[target].notna()
    )

    val_mask = (
        (train["_period"] >= "2016-01-01")
        & (train["_period"] <= "2017-12-01")
        & train[target].notna()
    )

    y_train_raw = train.loc[
        train_mask,
        target
    ].astype(str)

    classes = sorted(y_train_raw.unique())

    mapping = {
        state: i
        for i, state in enumerate(classes)
    }

    y_train = y_train_raw.map(mapping)

    X_train = train.loc[train_mask, features]

    y_val_raw = train.loc[
        val_mask,
        target
    ].astype(str)

    valid_val = y_val_raw.isin(mapping)

    X_val = train.loc[
        val_mask,
        features
    ].loc[valid_val]

    y_val = y_val_raw.loc[valid_val].map(mapping)

    print(f"Classes: {classes}")
    print(f"Training rows: {len(y_train):,}")
    print(f"Validation rows: {len(y_val):,}")

    model = make_hgb()

    print("Training multiclass HistGradientBoosting...")
    start = time.time()

    model.fit(X_train, y_train)

    elapsed = time.time() - start

    probability = model.predict_proba(X_val)
    prediction = model.predict(X_val)

    validation = {
        "macro_f1": f1_score(
            y_val,
            prediction,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_val,
            prediction,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_val,
            prediction,
        ),
        "log_loss": log_loss(
            y_val,
            probability,
            labels=list(range(len(classes))),
        ),
    }

    print(
        f"Validation macro-F1={validation['macro_f1']:.4f}, "
        f"weighted-F1={validation['weighted_f1']:.4f}, "
        f"balanced-accuracy={validation['balanced_accuracy']:.4f}"
    )

    # Full training
    full_mask = train[target].notna()

    X_full = train.loc[full_mask, features]

    y_full = train.loc[
        full_mask,
        target
    ].astype(str).map(mapping)

    model.fit(X_full, y_full)

    # Holdout targets are stored separately from test features.
    # Join them on the exact loan-month key before evaluation.
    holdout_eval = test[
        ["loan_id", "monthly_reporting_period"] + features
    ].merge(
        holdout[
            ["loan_id", "monthly_reporting_period", target]
        ],
        on=["loan_id", "monthly_reporting_period"],
        how="inner",
        validate="one_to_one",
    )

    hold_mask = holdout_eval[target].notna()

    X_hold = holdout_eval.loc[
        hold_mask,
        features
    ]

    y_hold_raw = holdout_eval.loc[
        hold_mask,
        target
    ].astype(str)

    valid_hold = y_hold_raw.isin(mapping)

    y_hold = y_hold_raw.loc[
        valid_hold
    ].map(mapping)

    X_hold = X_hold.loc[valid_hold]

    hold_probability = model.predict_proba(X_hold)
    hold_prediction = model.predict(X_hold)

    holdout_metrics = {
        "macro_f1": f1_score(
            y_hold,
            hold_prediction,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_hold,
            hold_prediction,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_hold,
            hold_prediction,
        ),
        "log_loss": log_loss(
            y_hold,
            hold_probability,
            labels=list(range(len(classes))),
        ),
    }

    print(
        f"HOLDOUT macro-F1={holdout_metrics['macro_f1']:.4f}, "
        f"weighted-F1={holdout_metrics['weighted_f1']:.4f}, "
        f"balanced-accuracy={holdout_metrics['balanced_accuracy']:.4f}"
    )

    # Save model
    model_dir = MODELS / "next_state"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        model_dir / "model.joblib",
    )

    with open(
        model_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "classes": classes,
                "mapping": mapping,
                "features": features,
                "validation_metrics": validation,
                "holdout_metrics": holdout_metrics,
                "training_seconds": elapsed,
            },
            f,
            indent=2,
            default=str,
        )

    # Test prediction
    test_probability = model.predict_proba(
        test[features]
    )

    test_prediction = [
        classes[i]
        for i in np.argmax(test_probability, axis=1)
    ]

    return {
        "model": "HistGradientBoosting",
        "classes": classes,
        "validation": validation,
        "holdout": holdout_metrics,
        "probability": test_probability,
        "prediction": test_prediction,
    }


def main():

    start = time.time()

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    print("=" * 70)
    print("INTain PHASE 2 — ML PREDICTION ENGINE")
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

    train["_period"] = period_to_date(
        train["monthly_reporting_period"]
    )

    test["_period"] = period_to_date(
        test["monthly_reporting_period"]
    )

    holdout["_period"] = period_to_date(
        holdout["monthly_reporting_period"]
    )

    print(f"Train:   {len(train):,}")
    print(f"Test:    {len(test):,}")
    print(f"Holdout: {len(holdout):,}")

    # ------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------

    features = get_features(train)

    print(f"\nUsing {len(features)} numeric prediction-time features.")

    # ------------------------------------------------------------
    # Binary models
    # ------------------------------------------------------------

    results = {}
    predictions = test[
        ["loan_id", "monthly_reporting_period"]
    ].copy()

    for target in BINARY_TARGETS:

        result = train_binary(
            train,
            test,
            holdout,
            features,
            target,
        )

        results[target] = result

        predictions[
            PROBABILITY_NAMES[target]
        ] = result["probability"]

    # ------------------------------------------------------------
    # Multiclass next-state
    # ------------------------------------------------------------

    state = train_next_state(
        train,
        test,
        holdout,
        features,
    )

    predictions["next_state_prediction"] = (
        state["prediction"]
    )

    for i, cls in enumerate(state["classes"]):

        safe_name = (
            str(cls)
            .lower()
            .replace("+", "plus")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

        predictions[
            f"next_state_probability_{safe_name}"
        ] = state["probability"][:, i]

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------

    prediction_path = DATA / "ml_predictions_test.csv"

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\nPredictions saved: {prediction_path}"
    )

    # ------------------------------------------------------------
    # Comparison report
    # ------------------------------------------------------------

    rows = []

    for target, result in results.items():

        v = result["validation"]
        h = result["holdout"]

        rows.append({
            "target": target,
            "model": result["model"],
            "validation_pr_auc": v["pr_auc"],
            "validation_roc_auc": v["roc_auc"],
            "validation_f1": v["f1"],
            "validation_brier": v["brier"],
            "holdout_pr_auc": h["pr_auc"],
            "holdout_roc_auc": h["roc_auc"],
            "holdout_f1": h["f1"],
            "holdout_brier": h["brier"],
            "threshold": result["threshold"],
        })

    rows.append({
        "target": "next_state",
        "model": state["model"],
        "validation_pr_auc": np.nan,
        "validation_roc_auc": np.nan,
        "validation_f1": state["validation"]["macro_f1"],
        "validation_brier": np.nan,
        "holdout_pr_auc": np.nan,
        "holdout_roc_auc": np.nan,
        "holdout_f1": state["holdout"]["macro_f1"],
        "holdout_brier": np.nan,
        "threshold": np.nan,
    })

    comparison_path = (
        REPORTS / "model_comparison.csv"
    )

    pd.DataFrame(rows).to_csv(
        comparison_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    with open(
        REPORTS / "phase2_reproducibility.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "seed": SEED,
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "sklearn": __import__(
                    "sklearn"
                ).__version__,
                "features": features,
                "validation": "2016-01 through 2017-12",
                "final_training_cutoff": "2017-12",
            },
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------

    report = []

    report.append(
        "# Phase 2 — ML Model Evaluation Report\n"
    )

    report.append(
        f"- Training rows: {len(train):,}\n"
        f"- Test rows: {len(test):,}\n"
        f"- Holdout rows: {len(holdout):,}\n"
        f"- Numeric features: {len(features)}\n"
        f"- Validation: 2016-01 to 2017-12\n"
        f"- Final training cutoff: 2017-12\n"
    )

    report.append("\n## Binary targets\n")

    for target, result in results.items():

        v = result["validation"]
        h = result["holdout"]

        report.append(
            f"\n### {target}\n"
            f"- Model: {result['model']}\n"
            f"- Threshold: {result['threshold']:.4f}\n"
            f"- Validation PR-AUC: {v['pr_auc']:.4f}\n"
            f"- Validation ROC-AUC: {v['roc_auc']:.4f}\n"
            f"- Validation F1: {v['f1']:.4f}\n"
            f"- Validation Brier: {v['brier']:.4f}\n"
            f"- Holdout PR-AUC: {h['pr_auc']:.4f}\n"
            f"- Holdout ROC-AUC: {h['roc_auc']:.4f}\n"
            f"- Holdout F1: {h['f1']:.4f}\n"
            f"- Holdout Brier: {h['brier']:.4f}\n"
        )

    report.append(
        "\n## Next state\n"
        f"- Validation macro-F1: "
        f"{state['validation']['macro_f1']:.4f}\n"
        f"- Validation weighted-F1: "
        f"{state['validation']['weighted_f1']:.4f}\n"
        f"- Holdout macro-F1: "
        f"{state['holdout']['macro_f1']:.4f}\n"
        f"- Holdout weighted-F1: "
        f"{state['holdout']['weighted_f1']:.4f}\n"
    )

    report.append(
        "\n## Leakage controls\n"
        "- Chronological validation was used.\n"
        "- 2018–2025 holdout was not used for model selection.\n"
        "- Target/censoring/future columns were excluded.\n"
        "- loan_id and monthly_reporting_period were excluded as predictors.\n"
    )

    report.append(
        "\n## Artifacts\n"
        f"- Predictions: `{prediction_path}`\n"
        f"- Comparison: `{comparison_path}`\n"
        "- Models: `models/`\n"
    )

    report_path = (
        REPORTS / "phase2_model_evaluation_report.md"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"Report: {report_path}")
    print(f"Predictions: {prediction_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()