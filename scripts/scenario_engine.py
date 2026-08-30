from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TEST_FILE = DATA_DIR / "loan_monthly_performance_test.csv"
SCENARIO_FILE = DATA_DIR / "macro_scenarios.csv"

SUMMARY_FILE = DATA_DIR / "scenario_risk_summary.csv"
RESULT_FILE = DATA_DIR / "scenario_risk_results.csv"
REPORT_FILE = REPORT_DIR / "phase4_scenario_report.md"


TARGET_MODELS = {
    "next_3m_delinquency_flag":
        MODEL_DIR / "next_3m_delinquency_flag" / "model.joblib",

    "next_6m_delinquency_flag":
        MODEL_DIR / "next_6m_delinquency_flag" / "model.joblib",

    "next_12m_default_flag":
        MODEL_DIR / "next_12m_default_flag" / "model.joblib",

    "next_12m_prepayment_flag":
        MODEL_DIR / "next_12m_prepayment_flag" / "model.joblib",
}


def load_data():

    print("=" * 70)
    print("INTain PHASE 4 — MACRO SCENARIO RISK ENGINE")
    print("=" * 70)

    print("\nLoading test dataset...")

    test = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Test loans: {test['loan_id'].nunique():,}"
    )

    print("\nLoading scenario definitions...")

    scenarios = pd.read_csv(
        SCENARIO_FILE
    )

    required_columns = {
        "scenario_name",
        "type",
        "parameter",
        "value",
        "description",
    }

    missing = required_columns - set(
        scenarios.columns
    )

    if missing:
        raise ValueError(
            "macro_scenarios.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Scenario parameter rows: "
        f"{len(scenarios):,}"
    )

    print(
        scenarios.to_string(index=False)
    )

    return test, scenarios


def load_models():

    print("\nLoading trained models...")

    models = {}

    for target, path in TARGET_MODELS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        models[target] = joblib.load(path)

        print(
            f"  Loaded {target}"
        )

    return models


def get_model_features(target):

    metadata_file = (
        MODEL_DIR /
        target /
        "metadata.json"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_file}"
        )

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    features = metadata.get(
        "features"
    )

    if not features:
        raise ValueError(
            f"No feature list found for {target}"
        )

    return features


def prepare_features(
    df,
    features
):

    missing = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing)
        )

    X = df[
        features
    ].copy()

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    return X


def build_scenario_parameters(
    scenario_df
):

    """
    Convert the 18-row parameter table into
    one dictionary per scenario.

    Example:

        Base
          default_multiplier = 1.0
          prepayment_multiplier = 1.0

        Adverse-Credit
          default_multiplier = 3.0
          prepayment_multiplier = 0.5

        High-Prepayment
          default_multiplier = 0.5
          prepayment_multiplier = 2.5
    """

    scenario_parameters = {}

    for scenario_name, group in (
        scenario_df.groupby(
            "scenario_name",
            sort=False
        )
    ):

        parameters = {}

        for _, row in group.iterrows():

            parameter = str(
                row["parameter"]
            ).strip()

            parameters[
                parameter
            ] = float(
                row["value"]
            )

        scenario_parameters[
            scenario_name
        ] = parameters

    return scenario_parameters


def get_multiplier(
    scenario_name,
    target,
    parameters
):

    """
    Scenario effects are applied to probabilities
    after model prediction.

    The actual macro_scenarios.csv multipliers are
    used for default and prepayment.

    Delinquency does not have an explicit parameter
    in the supplied scenario file, so its stress
    multiplier is derived transparently from the
    default multiplier.

    This is a scenario assumption, not a learned
    causal relationship.
    """

    default_multiplier = parameters.get(
        "default_multiplier",
        1.0
    )

    prepayment_multiplier = parameters.get(
        "prepayment_multiplier",
        1.0
    )

    if target == "next_12m_default_flag":

        return default_multiplier

    if target == "next_12m_prepayment_flag":

        return prepayment_multiplier

    if target in {
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
    }:

        # Convert default hazard stress into a
        # more moderate delinquency stress.
        return float(
            np.sqrt(
                default_multiplier
            )
        )

    raise ValueError(
        f"Unknown target: {target}"
    )


def apply_probability_multiplier(
    probabilities,
    multiplier
):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    return np.clip(
        probabilities * multiplier,
        0.0,
        1.0
    )


def generate_base_predictions(
    test,
    models
):

    print(
        "\nPreparing prediction features..."
    )

    feature_sets = {}

    for target in TARGET_MODELS:

        feature_sets[
            target
        ] = get_model_features(
            target
        )

    print(
        "\nGenerating base predictions..."
    )

    predictions = {}

    for target, model in models.items():

        X = prepare_features(
            test,
            feature_sets[target]
        )

        probability = model.predict_proba(
            X
        )[:, 1]

        predictions[
            target
        ] = probability

        print(
            f"  {target}: "
            f"mean={probability.mean():.6f}"
        )

    return predictions


def calculate_summary(
    scenario_name,
    predictions
):

    row = {
        "scenario": scenario_name
    }

    for target in TARGET_MODELS:

        values = predictions[
            target
        ]

        row[
            f"{target}_mean"
        ] = float(
            np.mean(values)
        )

        row[
            f"{target}_median"
        ] = float(
            np.median(values)
        )

        row[
            f"{target}_p90"
        ] = float(
            np.percentile(
                values,
                90
            )
        )

        row[
            f"{target}_p95"
        ] = float(
            np.percentile(
                values,
                95
            )
        )

        row[
            f"{target}_p99"
        ] = float(
            np.percentile(
                values,
                99
            )
        )

        row[
            f"{target}_high_risk_count"
        ] = int(
            np.sum(
                values >= 0.50
            )
        )

    return row


def run_scenarios(
    test,
    scenario_parameters,
    base_predictions
):

    summary_rows = []

    loan_rows = []

    for scenario_name, parameters in (
        scenario_parameters.items()
    ):

        print(
            f"\nRunning scenario: "
            f"{scenario_name}"
        )

        predictions = {}

        for target in TARGET_MODELS:

            multiplier = get_multiplier(
                scenario_name,
                target,
                parameters
            )

            predictions[
                target
            ] = apply_probability_multiplier(
                base_predictions[target],
                multiplier
            )

            print(
                f"  {target}: "
                f"multiplier={multiplier:.4f} "
                f"mean="
                f"{predictions[target].mean():.6f}"
            )

        summary_rows.append(
            calculate_summary(
                scenario_name,
                predictions
            )
        )

        for target in TARGET_MODELS:

            loan_rows.append(
                pd.DataFrame(
                    {
                        "loan_id":
                            test[
                                "loan_id"
                            ].values,

                        "monthly_reporting_period":
                            test[
                                "monthly_reporting_period"
                            ].values,

                        "scenario":
                            scenario_name,

                        "target":
                            target,

                        "predicted_probability":
                            predictions[
                                target
                            ],
                    }
                )
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    loan_df = pd.concat(
        loan_rows,
        ignore_index=True
    )

    return summary_df, loan_df


def create_report(
    summary_df,
    scenario_parameters
):

    lines = []

    lines.append(
        "# Phase 4 — Macro Scenario Risk Simulation"
    )

    lines.append("")

    lines.append(
        "The Phase 2 trained models provide the "
        "baseline probabilities. Phase 4 applies "
        "the explicit scenario multipliers from "
        "macro_scenarios.csv to perform deterministic "
        "stress testing."
    )

    lines.append("")

    lines.append(
        "## Scenario assumptions"
    )

    lines.append("")

    for scenario, parameters in (
        scenario_parameters.items()
    ):

        lines.append(
            f"### {scenario}"
        )

        for parameter, value in (
            parameters.items()
        ):

            lines.append(
                f"- {parameter}: {value}"
            )

        lines.append("")

    lines.append(
        "## Risk results"
    )

    lines.append("")

    for _, row in summary_df.iterrows():

        scenario = row[
            "scenario"
        ]

        lines.append(
            f"### {scenario}"
        )

        lines.append("")

        for target in TARGET_MODELS:

            mean = row[
                f"{target}_mean"
            ]

            p95 = row[
                f"{target}_p95"
            ]

            high_risk = row[
                f"{target}_high_risk_count"
            ]

            lines.append(
                f"- {target}: "
                f"mean={mean:.4%}, "
                f"P95={p95:.4%}, "
                f"high-risk count="
                f"{int(high_risk):,}"
            )

        lines.append("")

    base_rows = summary_df[
        summary_df["scenario"]
        .astype(str)
        .str.lower()
        .eq("base")
    ]

    if not base_rows.empty:

        base = base_rows.iloc[0]

        lines.append(
            "## Change versus Base"
        )

        lines.append("")

        for _, row in (
            summary_df.iterrows()
        ):

            if str(
                row["scenario"]
            ).lower() == "base":
                continue

            lines.append(
                f"### {row['scenario']}"
            )

            for target in TARGET_MODELS:

                base_mean = base[
                    f"{target}_mean"
                ]

                scenario_mean = row[
                    f"{target}_mean"
                ]

                if base_mean > 0:

                    change = (
                        scenario_mean /
                        base_mean
                        - 1.0
                    )

                    lines.append(
                        f"- {target}: "
                        f"{change:+.2%}"
                    )

            lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "These scenario results are stress-test "
        "outputs, not causal macroeconomic forecasts. "
        "The macro_scenarios.csv multipliers are explicit "
        "assumptions supplied for scenario analysis."
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"\nReport saved: "
        f"{REPORT_FILE}"
    )


def main():

    test, scenario_df = load_data()

    models = load_models()

    scenario_parameters = (
        build_scenario_parameters(
            scenario_df
        )
    )

    print(
        "\nScenario groups:"
    )

    for scenario, parameters in (
        scenario_parameters.items()
    ):

        print(
            f"  {scenario}: "
            f"{len(parameters)} parameters"
        )

    base_predictions = (
        generate_base_predictions(
            test,
            models
        )
    )

    summary_df, loan_df = run_scenarios(
        test,
        scenario_parameters,
        base_predictions
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    loan_df.to_csv(
        RESULT_FILE,
        index=False
    )

    create_report(
        summary_df,
        scenario_parameters
    )

    print(
        "\nScenario summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 4 SCENARIO ENGINE COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()