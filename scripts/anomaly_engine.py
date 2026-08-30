from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SERVICER_FILE = DATA / "servicer_updates.csv"
GROUND_TRUTH_FILE = DATA / "servicer_updates_ground_truth.json"
RULES_FILE = DATA / "validation_rules.json"


def normalize_period(value):
    """
    Normalize Fannie-style MMYYYY periods while handling
    values that pandas has interpreted as integers.

    Examples:
        122012 -> 2012-12
        12015  -> 2015-01
    """

    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if "." in text:
        text = text.split(".")[0]

    # Already YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return pd.to_datetime(
            text,
            format="%Y-%m",
            errors="coerce",
        )

    # Six-digit MMYYYY
    if len(text) == 6:
        month = int(text[:2])
        year = int(text[2:])

        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return pd.Timestamp(
                year=year,
                month=month,
                day=1,
            )

    # Five-digit representation caused by leading zero removal.
    # Example 12015 -> 01/2015
    if len(text) == 5:
        month = int(text[0])
        year = int(text[1:])

        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return pd.Timestamp(
                year=year,
                month=month,
                day=1,
            )

    return pd.NaT


def parse_period_series(series):
    return series.map(normalize_period)


def load_ground_truth():
    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data


def extract_ground_truth_records(data):
    """
    Ground truth was generated in Phase 1B.
    Handle common dictionary/list layouts without
    assuming a single exact JSON structure.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "anomalies",
            "ground_truth",
            "records",
            "labels",
        ]:
            if key in data and isinstance(
                data[key],
                list,
            ):
                return data[key]

        # Mapping of record IDs -> anomaly information
        records = []

        for key, value in data.items():

            if isinstance(value, dict):
                record = dict(value)

                if "loan_id" not in record:
                    record["loan_id"] = key

                records.append(record)

        if records:
            return records

    return []


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def rule_balance_discrepancy(df):
    """
    Detect suspicious balance movements.

    A single servicing snapshot does not contain previous UPB,
    therefore this rule operates within loan history.
    """

    result = pd.Series(
        False,
        index=df.index,
    )

    ordered = df.sort_values(
        ["loan_id", "_period"]
    )

    previous_upb = (
        ordered.groupby("loan_id")[
            "current_actual_upb"
        ].shift(1)
    )

    current_upb = pd.to_numeric(
        ordered["current_actual_upb"],
        errors="coerce",
    )

    previous_upb = pd.to_numeric(
        previous_upb,
        errors="coerce",
    )

    # Negative balance is intrinsically invalid.
    negative_balance = (
        current_upb < 0
    )

    # Extremely large unexplained increase.
    increase_ratio = (
        current_upb / previous_upb
    )

    suspicious_increase = (
        previous_upb.gt(0)
        & current_upb.gt(0)
        & (increase_ratio > 1.25)
    )

    ordered_result = (
        negative_balance
        | suspicious_increase
    )

    result.loc[
        ordered.index
    ] = ordered_result.fillna(False).values

    return result


def rule_delinquency_status_lag(df):
    """
    Detect suspicious delinquency transitions.

    A status should not normally jump backward or forward
    in a way inconsistent with the immediately preceding
    servicing state without supporting evidence.
    """

    result = pd.Series(
        False,
        index=df.index,
    )

    ordered = df.sort_values(
        ["loan_id", "_period"]
    )

    status = pd.to_numeric(
        ordered[
            "current_delinquency_status"
        ],
        errors="coerce",
    )

    previous_status = (
        status.groupby(
            ordered["loan_id"]
        ).shift(1)
    )

    # Detect unusually large downward jumps while the loan
    # remains active.
    suspicious_reversal = (
        previous_status.ge(3)
        & status.eq(0)
    )

    # Detect jumps of more than 2 delinquency levels.
    suspicious_jump = (
        status.notna()
        & previous_status.notna()
        & (
            (status - previous_status).abs()
            >= 3
        )
    )

    ordered_result = (
        suspicious_reversal
        | suspicious_jump
    )

    result.loc[
        ordered.index
    ] = ordered_result.fillna(False).values

    return result


def rule_missing_modification(df):
    """
    Detect modification records where a loan's servicing history
    indicates modification-related behavior but the current
    modification flag is absent.

    This is deliberately conservative.
    """

    result = pd.Series(
        False,
        index=df.index,
    )

    ordered = df.sort_values(
        ["loan_id", "_period"]
    )

    modification = (
        ordered[
            "monthly_modification_flag"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("Y")
    )

    previous_modification = (
        modification.groupby(
            ordered["loan_id"]
        ).shift(1)
        .fillna(False)
        .astype(bool)
    )

    # A modification followed immediately by N is not itself
    # an anomaly; therefore we only flag cases where a previous
    # modification is followed by a suspicious state change
    # and no current modification indication.
    status = pd.to_numeric(
        ordered[
            "current_delinquency_status"
        ],
        errors="coerce",
    )

    previous_status = (
        status.groupby(
            ordered["loan_id"]
        ).shift(1)
    )

    missing_flag = (
        previous_modification
        & ~modification
        & status.notna()
        & previous_status.notna()
        & (
            (status - previous_status).abs()
            >= 2
        )
    )

    result.loc[
        ordered.index
    ] = missing_flag.fillna(False).values

    return result


def rule_invalid_values(df):
    """
    Basic schema/business validation.
    """

    result = pd.Series(
        False,
        index=df.index,
    )

    upb = pd.to_numeric(
        df["current_actual_upb"],
        errors="coerce",
    )

    rate = pd.to_numeric(
        df["current_interest_rate"],
        errors="coerce",
    )

    delinquency = pd.to_numeric(
        df["current_delinquency_status"],
        errors="coerce",
    )

    result |= upb.lt(0)
    result |= rate.lt(0)
    result |= rate.gt(25)
    result |= delinquency.lt(0)
    result |= delinquency.gt(99)

    return result


def build_anomaly_engine(df):
    print("\nRunning anomaly rules...")

    df["balance_discrepancy_flag"] = (
        rule_balance_discrepancy(df)
    )

    df["delinquency_status_lag_flag"] = (
        rule_delinquency_status_lag(df)
    )

    df["missing_modification_flag"] = (
        rule_missing_modification(df)
    )

    df["invalid_value_flag"] = (
        rule_invalid_values(df)
    )

    rule_columns = [
        "balance_discrepancy_flag",
        "delinquency_status_lag_flag",
        "missing_modification_flag",
        "invalid_value_flag",
    ]

    df["anomaly_rule_count"] = (
        df[rule_columns]
        .astype(int)
        .sum(axis=1)
    )

    df["anomaly_flag"] = (
        df["anomaly_rule_count"] > 0
    )

    def severity(count):
        if count >= 3:
            return "CRITICAL"
        if count == 2:
            return "HIGH"
        if count == 1:
            return "MEDIUM"
        return "NONE"

    df["anomaly_severity"] = (
        df["anomaly_rule_count"]
        .map(severity)
    )

    def explanation(row):

        reasons = []

        if row["balance_discrepancy_flag"]:
            reasons.append(
                "Balance discrepancy"
            )

        if row["delinquency_status_lag_flag"]:
            reasons.append(
                "Delinquency status transition anomaly"
            )

        if row["missing_modification_flag"]:
            reasons.append(
                "Possible missing modification"
            )

        if row["invalid_value_flag"]:
            reasons.append(
                "Invalid servicing value"
            )

        if not reasons:
            return "No anomaly detected"

        return "; ".join(reasons)

    df["anomaly_explanation"] = (
        df.apply(
            explanation,
            axis=1,
        )
    )

    return df


def build_ground_truth_table(df, records):
    """
    Attach the 200 synthetic ground-truth anomaly labels.

    Ground truth schema:
        loan_id
        reporting_period
        anomaly_type

    The servicing dataset uses monthly_reporting_period, while
    the ground-truth JSON uses reporting_period. Both are normalized
    to the internal _period key before joining.
    """

    truth = pd.DataFrame(records)

    if truth.empty:
        raise ValueError(
            "Ground-truth JSON contains no anomaly records."
        )

    print(
        f"\nGround-truth records loaded: {len(truth):,}"
    )

    truth.columns = [
        str(c).strip()
        for c in truth.columns
    ]

    required = {
        "loan_id",
        "reporting_period",
        "anomaly_type",
    }

    missing = required - set(truth.columns)

    if missing:
        raise ValueError(
            "Ground-truth is missing required columns: "
            + ", ".join(sorted(missing))
        )

    truth["loan_id"] = (
        truth["loan_id"]
        .astype(str)
        .str.strip()
    )

    truth["_period"] = parse_period_series(
        truth["reporting_period"]
    )

    if truth["_period"].isna().any():
        bad = int(truth["_period"].isna().sum())
        raise ValueError(
            f"{bad} ground-truth reporting periods could not be parsed."
        )

    truth["anomaly_type"] = (
        truth["anomaly_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    truth_keys = truth[
        [
            "loan_id",
            "_period",
            "anomaly_type",
        ]
    ].drop_duplicates(
        subset=[
            "loan_id",
            "_period",
        ]
    )

    if len(truth_keys) != len(truth):
        raise ValueError(
            "Duplicate loan_id + reporting_period keys found "
            "in ground truth."
        )

    df["loan_id"] = (
        df["loan_id"]
        .astype(str)
        .str.strip()
    )

    df = df.merge(
        truth_keys,
        on=[
            "loan_id",
            "_period",
        ],
        how="left",
        validate="many_to_one",
    )

    df["ground_truth_anomaly"] = (
        df["anomaly_type_y"]
        if "anomaly_type_y" in df.columns
        else df["anomaly_type"]
    )

    # The servicing file already has anomaly_type for synthetic
    # rows. Keep the ground-truth value separately so evaluation
    # does not depend on that source field.
    if "anomaly_type_y" in df.columns:
        df["ground_truth_type"] = df["anomaly_type_y"]
        if "anomaly_type_x" in df.columns:
            df["anomaly_type"] = df["anomaly_type_x"]
        df.drop(
            columns=["anomaly_type_y"],
            inplace=True,
        )
    else:
        df["ground_truth_type"] = df["anomaly_type"]

    df["ground_truth_anomaly"] = (
        df["ground_truth_type"]
        .notna()
    )

    # Multiple servicing rows can share the same loan_id + period
    # key. Ground truth labels are key-based, so validate the number
    # of unique matched keys rather than the number of dataframe rows.
    matched_keys = (
        df.loc[
            df["ground_truth_anomaly"],
            ["loan_id", "_period"]
        ]
        .drop_duplicates()
    )

    matched = len(matched_keys)

    print(
        f"Ground-truth anomalies matched: {matched:,}"
    )

    if matched != len(truth_keys):
        raise ValueError(
            f"Ground-truth matching failed: expected "
            f"{len(truth_keys):,} unique keys, "
            f"matched {matched:,} unique keys."
        )

    return df


def evaluate_detection(df):
    """
    Evaluate anomaly detection against all 10,200 servicing records.

    The 200 ground-truth records are the positive class.
    The remaining servicing records are treated as negatives.
    """

    if "ground_truth_anomaly" not in df.columns:
        raise ValueError(
            "ground_truth_anomaly column is missing."
        )

    y_true = (
        df["ground_truth_anomaly"]
        .astype(int)
    )

    y_pred = (
        df["anomaly_flag"]
        .astype(int)
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 70)

    print(
        f"Evaluated records: {len(df):,}"
    )

    print(
        f"True anomalies: {y_true.sum():,}"
    )

    print(
        f"Predicted anomalies: {y_pred.sum():,}"
    )

    print(
        f"True negatives: {matrix[0, 0]:,}"
    )

    print(
        f"False positives: {matrix[0, 1]:,}"
    )

    print(
        f"False negatives: {matrix[1, 0]:,}"
    )

    print(
        f"True positives: {matrix[1, 1]:,}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    return {
        "evaluated_records": int(len(df)),
        "true_anomalies": int(y_true.sum()),
        "predicted_anomalies": int(y_pred.sum()),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
    }


def main():

    print("=" * 70)
    print("INTain PHASE 3 — SERVICER ANOMALY ENGINE")
    print("=" * 70)

    print("\nLoading servicer data...")

    df = pd.read_csv(
        SERVICER_FILE,
        low_memory=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # Normalize period.
    df["_period"] = parse_period_series(
        df["monthly_reporting_period"]
    )

    invalid_periods = (
        df["_period"].isna().sum()
    )

    print(
        f"Invalid periods: {invalid_periods:,}"
    )

    # Normalize synthetic indicator.
    df["is_synthetic"] = (
        df["is_synthetic"]
        .map(normalize_bool)
    )

    print(
        f"Synthetic records: "
        f"{df['is_synthetic'].sum():,}"
    )

    # ------------------------------------------------------------
    # Load validation rules
    # ------------------------------------------------------------

    if RULES_FILE.exists():

        with open(
            RULES_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            rules = json.load(f)

        if isinstance(rules, dict):
            print(
                f"Validation rules loaded: "
                f"{len(rules):,}"
            )
        elif isinstance(rules, list):
            print(
                f"Validation rules loaded: "
                f"{len(rules):,}"
            )

    # ------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------

    ground_truth_raw = load_ground_truth()

    ground_truth_records = (
        extract_ground_truth_records(
            ground_truth_raw
        )
    )

    # ------------------------------------------------------------
    # Run rules
    # ------------------------------------------------------------

    df = build_anomaly_engine(
        df
    )

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    df = build_ground_truth_table(
        df,
        ground_truth_records,
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    evaluation = evaluate_detection(
        df
    )

    # ------------------------------------------------------------
    # Output anomaly dataset
    # ------------------------------------------------------------

    output_columns = [
        "loan_id",
        "monthly_reporting_period",
        "servicer_name",
        "current_actual_upb",
        "current_interest_rate",
        "current_delinquency_status",
        "monthly_modification_flag",
        "is_synthetic",
        "anomaly_type",
        "balance_discrepancy_flag",
        "delinquency_status_lag_flag",
        "missing_modification_flag",
        "invalid_value_flag",
        "anomaly_rule_count",
        "anomaly_flag",
        "anomaly_severity",
        "anomaly_explanation",
        "ground_truth_anomaly",
        "ground_truth_type",
    ]

    output = (
        DATA /
        "servicer_anomaly_results.csv"
    )

    df[
        output_columns
    ].to_csv(
        output,
        index=False,
    )

    print(
        f"\nAnomaly results saved: {output}"
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    severity_counts = (
        df["anomaly_severity"]
        .value_counts()
        .to_dict()
    )

    rule_counts = {
        column: int(
            df[column].sum()
        )
        for column in [
            "balance_discrepancy_flag",
            "delinquency_status_lag_flag",
            "missing_modification_flag",
            "invalid_value_flag",
        ]
    }

    summary = {
        "total_records": len(df),
        "synthetic_records": int(
            df["is_synthetic"].sum()
        ),
        "flagged_records": int(
            df["anomaly_flag"].sum()
        ),
        "severity_distribution": severity_counts,
        "rule_trigger_counts": rule_counts,
        "evaluation": evaluation,
    }

    with open(
        REPORTS /
        "phase3_anomaly_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    report = []

    report.append(
        "# Phase 3 — Servicer Anomaly Intelligence\n"
    )

    report.append(
        f"- Total servicing records: {len(df):,}\n"
        f"- Synthetic records: {int(df['is_synthetic'].sum()):,}\n"
        f"- Flagged records: {int(df['anomaly_flag'].sum()):,}\n"
    )

    report.append(
        "\n## Rule trigger counts\n"
    )

    for rule, count in rule_counts.items():
        report.append(
            f"- {rule}: {count:,}"
        )

    report.append(
        "\n## Severity distribution\n"
    )

    for severity, count in severity_counts.items():
        report.append(
            f"- {severity}: {count:,}"
        )

    if evaluation:

        report.append(
            "\n## Ground-truth evaluation\n"
        )

        report.append(
            f"- Precision: {evaluation['precision']:.4f}\n"
            f"- Recall: {evaluation['recall']:.4f}\n"
            f"- F1: {evaluation['f1']:.4f}\n"
        )

        report.append(
            "\nConfusion matrix:\n"
        )

        report.append(
            str(
                evaluation[
                    "confusion_matrix"
                ]
            )
        )

    report.append(
        "\n## Detection philosophy\n"
        "\nThe anomaly engine uses deterministic "
        "business-rule validation. It does not use "
        "machine learning to manufacture anomaly labels. "
        "Synthetic anomalies are used only for evaluation.\n"
    )

    (
        REPORTS /
        "phase3_anomaly_report.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        f"Report saved: "
        f"{REPORTS / 'phase3_anomaly_report.md'}"
    )

    print("\n" + "=" * 70)
    print("PHASE 3 ANOMALY ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()