from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.data_service import DataService
from app.risk_service import (
    build_risk_summary,
    build_risk_drivers,
    build_recommendation,
    clean_record,
)


app = FastAPI(
    title="INTain Loan Intelligence API",
    version="1.0.0",
    description="Loan risk, servicing anomaly and scenario intelligence API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = DataService()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "INTain Loan Intelligence API",
    }


@app.get("/loans/search")
def search_loans(
    q: str = Query("", description="Loan ID search"),
    limit: int = Query(20, ge=1, le=100),
):
    performance_loan_ids = set(
        data.test["loan_id"].astype(str).unique()
    )

    result = data.static[
        data.static["loan_id"].astype(str).isin(performance_loan_ids)
    ].copy()

    if q:
        result = result[
            result["loan_id"].astype(str).str.contains(
                str(q),
                case=False,
                na=False,
            )
        ]

    result = result.head(limit)

    return {
        "count": len(result),
        "loans": [
            clean_record(row)
            for row in result.to_dict(orient="records")
        ],
    }


@app.get("/loans/{loan_id}")
def get_loan(loan_id: str):
    static = data.get_static(loan_id)
    rows = data.get_loan_rows(loan_id)

    if static is None or rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Loan {loan_id} was not found.",
        )

    latest = rows.iloc[-1]

    return {
        "loan_id": loan_id,
        "static": clean_record(static.to_dict()),
        "latest": clean_record(latest.to_dict()),
        "observation_count": len(rows),
    }


@app.get("/loans/{loan_id}/history")
def loan_history(loan_id: str):
    rows = data.get_loan_rows(loan_id)

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Loan {loan_id} was not found.",
        )

    columns = [
        "loan_id",
        "monthly_reporting_period",
        "current_actual_upb",
        "current_interest_rate",
        "current_delinquency_status",
        "current_state",
        "current_modified",
    ]

    columns = [c for c in columns if c in rows.columns]

    return {
        "loan_id": loan_id,
        "count": len(rows),
        "history": [
            clean_record(row)
            for row in rows[columns].to_dict(orient="records")
        ],
    }


@app.get("/loans/{loan_id}/risk")
def loan_risk(loan_id: str):
    rows = data.get_loan_rows(loan_id)
    predictions = data.get_predictions(loan_id)

    if rows.empty or predictions.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Risk information for loan {loan_id} was not found.",
        )

    latest = rows.iloc[-1]

    risk = build_risk_summary(predictions)
    drivers = build_risk_drivers(latest)

    anomalies = data.get_anomalies(loan_id)

    recommendation = build_recommendation(
        risk,
        anomalies,
    )

    return {
        "loan_id": loan_id,
        "risk": risk,
        "drivers": drivers,
        "recommendation": recommendation,
    }


@app.get("/loans/{loan_id}/anomalies")
def loan_anomalies(loan_id: str):
    anomalies = data.get_anomalies(loan_id)

    return {
        "loan_id": loan_id,
        "count": len(anomalies),
        "anomalies": [
            clean_record(row)
            for row in anomalies.to_dict(orient="records")
        ],
    }


@app.get("/loans/{loan_id}/scenarios")
def loan_scenarios(loan_id: str):
    scenarios = data.get_scenarios(loan_id)

    if scenarios.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario information for loan {loan_id} was not found.",
        )

    return {
        "loan_id": loan_id,
        "count": len(scenarios),
        "scenarios": [
            clean_record(row)
            for row in scenarios.to_dict(orient="records")
        ],
    }


@app.get("/scenarios/summary")
def scenario_summary():
    summary = data.scenario_summary

    return {
        "count": len(summary),
        "scenarios": [
            clean_record(row)
            for row in summary.to_dict(orient="records")
        ],
    }