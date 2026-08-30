import math
import pandas as pd


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def clean_record(record):
    return {
        str(k): clean_value(v)
        for k, v in record.items()
    }


def probability(value):
    if value is None or pd.isna(value):
        return None

    return round(float(value) * 100, 2)


def risk_band(probability_pct):
    if probability_pct is None:
        return "Unknown"

    if probability_pct >= 20:
        return "Critical"

    if probability_pct >= 10:
        return "High"

    if probability_pct >= 5:
        return "Moderate"

    if probability_pct >= 2:
        return "Low"

    return "Very Low"


def build_risk_summary(predictions):
    if predictions.empty:
        return None

    # Predictions contain one row per loan observation.
    # Use the latest reporting period, not simply the dataframe's
    # original row order.
    predictions = predictions.copy()

    if "monthly_reporting_period" in predictions.columns:
        predictions["_period"] = pd.to_numeric(
            predictions["monthly_reporting_period"],
            errors="coerce",
        )
        predictions = predictions.sort_values("_period")

    row = predictions.iloc[-1]

    # IMPORTANT:
    # These are the actual column names produced by Phase 2.
    default_probability = probability(
        row.get("default_12m_probability")
    )

    delinquency_3m = probability(
        row.get("delinquency_3m_probability")
    )

    delinquency_6m = probability(
        row.get("delinquency_6m_probability")
    )

    prepayment = probability(
        row.get("prepayment_12m_probability")
    )

    next_state = row.get("next_state_prediction")

    return {
        "default_probability_12m": default_probability,
        "default_risk_band": risk_band(default_probability),
        "delinquency_probability_3m": delinquency_3m,
        "delinquency_probability_6m": delinquency_6m,
        "prepayment_probability_12m": prepayment,
        "next_state_prediction": clean_value(next_state),
    }


def build_risk_drivers(row):
    drivers = []

    delinquency_count = row.get("delinquency_count_12m")

    if delinquency_count is not None and not pd.isna(delinquency_count):
        if float(delinquency_count) > 0:
            drivers.append({
                "factor": "Recent delinquency activity",
                "value": int(delinquency_count),
                "severity": "High",
            })

    ever_delinquent = row.get("ever_delinquent")

    if ever_delinquent is not None and not pd.isna(ever_delinquent):
        if int(float(ever_delinquent)) == 1:
            drivers.append({
                "factor": "Historical delinquency",
                "value": "Yes",
                "severity": "Moderate",
            })

    max_dpd = row.get("max_delinquency_to_date")

    if max_dpd is not None and not pd.isna(max_dpd):
        max_dpd = float(max_dpd)

        if max_dpd >= 3:
            drivers.append({
                "factor": "Maximum delinquency history",
                "value": "90+ DPD",
                "severity": "Critical",
            })
        elif max_dpd >= 2:
            drivers.append({
                "factor": "Maximum delinquency history",
                "value": "60 DPD",
                "severity": "High",
            })
        elif max_dpd >= 1:
            drivers.append({
                "factor": "Maximum delinquency history",
                "value": "30 DPD",
                "severity": "Moderate",
            })

    ltv = row.get("original_ltv")

    if ltv is not None and not pd.isna(ltv):
        ltv = float(ltv)

        if ltv >= 90:
            drivers.append({
                "factor": "Original LTV",
                "value": round(ltv, 1),
                "severity": "High",
            })
        elif ltv >= 80:
            drivers.append({
                "factor": "Original LTV",
                "value": round(ltv, 1),
                "severity": "Moderate",
            })

    dti = row.get("dti_ratio")

    if dti is not None and not pd.isna(dti):
        dti = float(dti)

        if dti >= 45:
            drivers.append({
                "factor": "Debt-to-income ratio",
                "value": round(dti, 1),
                "severity": "High",
            })
        elif dti >= 36:
            drivers.append({
                "factor": "Debt-to-income ratio",
                "value": round(dti, 1),
                "severity": "Moderate",
            })

    fico = row.get("borrower_credit_score")

    if fico is not None and not pd.isna(fico):
        fico = float(fico)

        if fico < 660:
            drivers.append({
                "factor": "Borrower credit score",
                "value": int(fico),
                "severity": "High",
            })
        elif fico < 700:
            drivers.append({
                "factor": "Borrower credit score",
                "value": int(fico),
                "severity": "Moderate",
            })

    # Current delinquency status
    current_dpd = row.get("current_delinquency_status")

    if current_dpd is not None and not pd.isna(current_dpd):
        status = str(current_dpd).strip()

        if status in {"90", "90+", "3"}:
            drivers.append({
                "factor": "Current delinquency status",
                "value": "90+ DPD",
                "severity": "Critical",
            })
        elif status in {"60", "2"}:
            drivers.append({
                "factor": "Current delinquency status",
                "value": "60 DPD",
                "severity": "High",
            })
        elif status in {"30", "1"}:
            drivers.append({
                "factor": "Current delinquency status",
                "value": "30 DPD",
                "severity": "Moderate",
            })

    return drivers


def build_recommendation(risk, anomalies):
    if risk is None:
        return {
            "action": "Insufficient data",
            "explanation": "Insufficient risk data for a recommendation.",
        }

    default_risk = risk.get("default_probability_12m") or 0
    delinquency_3m = risk.get("delinquency_probability_3m") or 0
    delinquency_6m = risk.get("delinquency_probability_6m") or 0
    anomaly_count = len(anomalies)

    if default_risk >= 20:
        action = "Immediate review"
        explanation = (
            "The projected 12-month default risk is critically elevated."
        )

    elif default_risk >= 10:
        action = "Priority review"
        explanation = (
            "The projected 12-month default risk is high and "
            "should receive closer monitoring."
        )

    elif delinquency_3m >= 10 or delinquency_6m >= 15:
        action = "Monitor closely"
        explanation = (
            "Near-term delinquency risk is elevated even though "
            "12-month default risk is not critical."
        )

    elif anomaly_count > 0:
        action = "Servicing review"
        explanation = (
            "Credit risk indicators are not currently critical, "
            "but servicing anomalies require attention."
        )

    else:
        action = "Routine monitoring"
        explanation = (
            "The available risk indicators do not currently indicate "
            "a critical credit concern."
        )

    if anomaly_count:
        explanation += (
            f" {anomaly_count} servicing anomaly record(s) "
            "also require attention."
        )

    return {
        "action": action,
        "explanation": explanation,
    }