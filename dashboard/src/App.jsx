import { useEffect, useMemo, useState } from "react";
import {
  getLoan,
  getLoanHistory,
  getLoanRisk,
  getLoanAnomalies,
  getLoanScenarios,
  getScenarioSummary,
} from "./api";
import "./App.css";

function displayValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "object") {
    return "—";
  }

  return String(value);
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return displayValue(value);
  }

  return number.toLocaleString("en-US", {
    maximumFractionDigits: 2,
  });
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "—";
  }

  return `${number.toFixed(2)}%`;
}

function formatScenarioPercent(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "—";
  }

  const percentage = Math.abs(number) <= 1 ? number * 100 : number;

  return `${percentage.toFixed(2)}%`;
}

function extractArray(result, key) {
  if (Array.isArray(result)) {
    return result;
  }

  if (result && Array.isArray(result[key])) {
    return result[key];
  }

  return [];
}

function riskClass(riskBand) {
  if (!riskBand) {
    return "";
  }

  const band = String(riskBand).toLowerCase();

  if (band.includes("critical") || band.includes("high")) {
    return "high";
  }

  if (band.includes("moderate") || band.includes("medium")) {
    return "medium";
  }

  return "low";
}

function App() {
  const [loanId, setLoanId] = useState("270296749341");

  const [loanData, setLoanData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [history, setHistory] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [scenarioSummary, setScenarioSummary] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadLoan(id) {
    const cleanId = String(id).trim();

    if (!cleanId) {
      setError("Please enter a loan ID.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      /*
       * IMPORTANT:
       * Load the loan directly instead of depending on
       * /loans/search response formatting.
       */
      const loan = await getLoan(cleanId);

      if (!loan || !loan.loan_id) {
        throw new Error(`Loan ${cleanId} was not found.`);
      }

      setLoanData(loan);

      /*
       * Load the remaining sections independently.
       * If one secondary endpoint fails, the main loan profile
       * should still remain visible.
       */
      const [
        riskResult,
        historyResult,
        anomaliesResult,
        scenariosResult,
      ] = await Promise.allSettled([
        getLoanRisk(cleanId),
        getLoanHistory(cleanId),
        getLoanAnomalies(cleanId),
        getLoanScenarios(cleanId),
      ]);

      setRiskData(
        riskResult.status === "fulfilled"
          ? riskResult.value
          : null
      );

      /*
       * API returns:
       * {
       *   loan_id: "...",
       *   count: ...,
       *   history: [...]
       * }
       *
       * Store only the actual history array.
       */
      setHistory(
        historyResult.status === "fulfilled"
          ? extractArray(historyResult.value, "history")
          : []
      );

      /*
       * API returns:
       * {
       *   loan_id: "...",
       *   count: ...,
       *   anomalies: [...]
       * }
       *
       * Store only the actual anomalies array.
       */
      setAnomalies(
        anomaliesResult.status === "fulfilled"
          ? extractArray(anomaliesResult.value, "anomalies")
          : []
      );

      /*
       * API returns:
       * {
       *   loan_id: "...",
       *   count: ...,
       *   scenarios: [...]
       * }
       *
       * Store only the actual scenarios array.
       */
      setScenarios(
        scenariosResult.status === "fulfilled"
          ? extractArray(scenariosResult.value, "scenarios")
          : []
      );
    } catch (err) {
      console.error("Loan loading error:", err);

      setLoanData(null);
      setRiskData(null);
      setHistory([]);
      setAnomalies([]);
      setScenarios([]);

      setError(
        err?.message ||
          "Unable to load this loan. Check that the loan ID exists."
      );
    } finally {
      setLoading(false);
    }
  }

  async function loadScenarioSummary() {
    try {
      const result = await getScenarioSummary();

      /*
       * API returns:
       * {
       *   count: ...,
       *   scenarios: [...]
       * }
       *
       * Store only the actual scenarios array.
       */
      setScenarioSummary(
        extractArray(result, "scenarios")
      );
    } catch (err) {
      console.error("Scenario summary error:", err);
      setScenarioSummary([]);
    }
  }

  useEffect(() => {
    loadScenarioSummary();
  }, []);

  useEffect(() => {
    if (loanId) {
      loadLoan(loanId);
    }
  }, []);

  function handleSubmit(event) {
    event.preventDefault();
    loadLoan(loanId);
  }

  const staticData = loanData?.static || {};
  const latestData = loanData?.latest || {};

  const risk = riskData?.risk || {};
  const drivers = Array.isArray(riskData?.drivers)
    ? riskData.drivers
    : [];

  const recommendation = riskData?.recommendation || null;

  const defaultRisk = risk.default_probability_12m;
  const delinquency3m = risk.delinquency_probability_3m;
  const delinquency6m = risk.delinquency_probability_6m;
  const prepayment = risk.prepayment_probability_12m;

  const riskBand = risk.default_risk_band || "Unknown";

  /*
   * Remove duplicate scenario names while preserving order.
   */
  const uniqueScenarios = useMemo(() => {
    if (!Array.isArray(scenarios)) {
      return [];
    }

    const seen = new Set();

    return scenarios.filter((item) => {
      if (!item || typeof item !== "object") {
        return false;
      }

      const name =
        item.scenario_name ??
        item.scenario ??
        item.scenarioName;

      if (!name) {
        return false;
      }

      if (seen.has(name)) {
        return false;
      }

      seen.add(name);
      return true;
    });
  }, [scenarios]);

  const staticFields = [
    ["Channel", staticData.channel],
    ["Seller", staticData.seller_name],
    ["Original Interest Rate", staticData.original_interest_rate],
    ["Original UPB", formatNumber(staticData.original_upb)],
    ["Original Loan Term", staticData.original_loan_term],
    ["Original LTV", staticData.original_ltv],
    ["Original CLTV", staticData.original_cltv],
    ["Borrowers", staticData.number_of_borrowers],
    ["DTI Ratio", staticData.dti_ratio],
    ["Borrower Credit Score", staticData.borrower_credit_score],
    ["First-Time Homebuyer", staticData.first_time_homebuyer_flag],
    ["Loan Purpose", staticData.loan_purpose],
    ["Property Type", staticData.property_type],
    ["Units", staticData.number_of_units],
    ["Occupancy", staticData.occupancy_status],
    ["Property State", staticData.property_state],
  ];

  const latestFields = [
    ["Reporting Period", latestData.monthly_reporting_period],
    ["Servicer", latestData.servicer_name],
    ["Current UPB", formatNumber(latestData.current_actual_upb)],
    ["Current Interest Rate", latestData.current_interest_rate],
    ["Delinquency Status", latestData.current_delinquency_status],
    ["Current State", latestData.current_state],
    ["Current Modified", latestData.current_modified],
    ["Loan Age", latestData.loan_age],
    ["Remaining Months", latestData.remaining_months_to_maturity],
    ["Adjusted Maturity", latestData.adjusted_months_to_maturity],
    ["Ever Delinquent", latestData.ever_delinquent],
    ["Max Delinquency", latestData.max_delinquency_to_date],
    ["12M Delinquency Count", latestData.delinquency_count_12m],
    ["Ever Modified", latestData.ever_modified],
    ["Observation Count", loanData?.observation_count],
  ];

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="brand">INTAIN</div>
          <div className="brand-sub">
            Loan Intelligence Platform
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          API Connected
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div>
            <div className="eyebrow">
              INTELLIGENT LOAN RISK ANALYTICS
            </div>

            <h1>Portfolio Risk Command Center</h1>

            <p>
              Predict loan-level risk, identify servicing anomalies
              and understand how macroeconomic scenarios affect the
              portfolio.
            </p>
          </div>

          <div className="hero-stat">
            <span>ACTIVE ENGINE</span>
            <strong>ML + Rules + Scenarios</strong>
          </div>
        </section>

        <section className="search-panel">
          <div>
            <h2>Loan Investigation</h2>
            <p>
              Search by loan ID to inspect its complete risk profile.
            </p>
          </div>

          <form
            className="search-form"
            onSubmit={handleSubmit}
          >
            <input
              value={loanId}
              onChange={(event) =>
                setLoanId(event.target.value)
              }
              placeholder="Enter loan ID"
              aria-label="Loan ID"
            />

            <button type="submit" disabled={loading}>
              {loading ? "Loading..." : "Search"}
            </button>
          </form>
        </section>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {loanData && (
          <>
            <section className="section-heading">
              <div>
                <div className="eyebrow">LOAN PROFILE</div>
                <h2>{loanData.loan_id}</h2>
              </div>

              <span
                className={`risk-badge ${riskClass(riskBand)}`}
              >
                {riskBand}
              </span>
            </section>

            <section className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">
                  12M Default Probability
                </div>

                <div className="metric-value">
                  {formatPercent(defaultRisk)}
                </div>

                <div className="metric-sub">
                  Calibrated risk estimate
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  3M Delinquency
                </div>

                <div className="metric-value">
                  {formatPercent(delinquency3m)}
                </div>

                <div className="metric-sub">
                  Probability of 30+ DPD
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  6M Delinquency
                </div>

                <div className="metric-value">
                  {formatPercent(delinquency6m)}
                </div>

                <div className="metric-sub">
                  Forward-looking risk
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  12M Prepayment
                </div>

                <div className="metric-value">
                  {formatPercent(prepayment)}
                </div>

                <div className="metric-sub">
                  Expected prepayment probability
                </div>
              </div>
            </section>

            <section className="dashboard-grid">
              <div className="panel">
                <div className="panel-title">
                  <div>
                    <h3>Loan Information</h3>
                    <p>
                      Original borrower and loan attributes
                    </p>
                  </div>
                </div>

                <div className="details">
                  {staticFields.map(([label, value]) => (
                    <div
                      className="detail-row"
                      key={label}
                    >
                      <span>{label}</span>
                      <strong>
                        {displayValue(value)}
                      </strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <div className="panel-title">
                  <div>
                    <h3>Risk Assessment</h3>
                    <p>
                      Model-generated forward-looking signals
                    </p>
                  </div>
                </div>

                <div className="risk-list">
                  <div>
                    <span>3-Month Delinquency</span>
                    <strong>
                      {formatPercent(delinquency3m)}
                    </strong>
                  </div>

                  <div>
                    <span>6-Month Delinquency</span>
                    <strong>
                      {formatPercent(delinquency6m)}
                    </strong>
                  </div>

                  <div>
                    <span>12-Month Default</span>
                    <strong>
                      {formatPercent(defaultRisk)}
                    </strong>
                  </div>

                  <div>
                    <span>12-Month Prepayment</span>
                    <strong>
                      {formatPercent(prepayment)}
                    </strong>
                  </div>

                  {risk.next_state_prediction && (
                    <div>
                      <span>Predicted Next State</span>
                      <strong>
                        {risk.next_state_prediction}
                      </strong>
                    </div>
                  )}
                </div>

                {drivers.length > 0 && (
                  <>
                    <div
                      className="panel-title"
                      style={{ marginTop: "24px" }}
                    >
                      <div>
                        <h3>Risk Drivers</h3>
                        <p>
                          Factors contributing to the assessment
                        </p>
                      </div>
                    </div>

                    <div className="table">
                      {drivers.map((driver, index) => (
                        <div
                          className="table-row"
                          key={index}
                        >
                          <strong>
                            {displayValue(driver.factor)}
                          </strong>

                          <span>
                            {displayValue(driver.value)}
                            {driver.severity
                              ? ` · ${driver.severity}`
                              : ""}
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {recommendation && (
                  <div
                    className="recommendation"
                    style={{ marginTop: "24px" }}
                  >
                    <strong>
                      {recommendation.action}
                    </strong>

                    <p>
                      {recommendation.explanation}
                    </p>
                  </div>
                )}
              </div>
            </section>

            <section className="dashboard-grid">
              <div className="panel">
                <div className="panel-title">
                  <div>
                    <h3>Servicing</h3>
                    <p>
                      Most recent servicing observation
                    </p>
                  </div>
                </div>

                <div className="details">
                  {latestFields.map(([label, value]) => (
                    <div
                      className="detail-row"
                      key={label}
                    >
                      <span>{label}</span>
                      <strong>
                        {displayValue(value)}
                      </strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <div className="panel-title">
                  <div>
                    <h3>Recommendation</h3>
                    <p>
                      Portfolio action based on current signals
                    </p>
                  </div>
                </div>

                {recommendation ? (
                  <div className="recommendation large">
                    <strong>
                      {recommendation.action}
                    </strong>

                    <p>
                      {recommendation.explanation}
                    </p>
                  </div>
                ) : (
                  <div className="empty-state">
                    No recommendation available.
                  </div>
                )}
              </div>
            </section>

            <section className="dashboard-grid">
              <div className="panel">
                <div className="panel-title">
                  <div>
                    <h3>Servicing Anomalies</h3>
                    <p>
                      Rule-based integrity and servicing checks
                    </p>
                  </div>

                  <span
                    className={`count ${
                      anomalies.length
                        ? "danger"
                        : "safe"
                    }`}
                  >
                    {anomalies.length}
                  </span>
                </div>

                {anomalies.length === 0 ? (
                  <div className="empty-state">
                    No detected anomalies for this loan.
                  </div>
                ) : (
                  <div className="table">
                    {anomalies.slice(0, 10).map(
                      (item, index) => (
                        <div
                          className="table-row"
                          key={index}
                        >
                          <strong>
                            {item.anomaly_type ||
                              item.type ||
                              "Detected anomaly"}
                          </strong>

                          <span>
                            {displayValue(
                              item.reporting_period ??
                                item.monthly_reporting_period
                            )}
                          </span>
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>

              <div className="panel">
                <div className="panel-title">
                  <div>
                    <h3>Macro Scenarios</h3>
                    <p>
                      Stress-test impact on this loan
                    </p>
                  </div>
                </div>

                {uniqueScenarios.length === 0 ? (
                  <div className="empty-state">
                    No scenario data available.
                  </div>
                ) : (
                  <div className="table">
                    {uniqueScenarios.map(
                      (item, index) => {
                        const name =
                          item.scenario_name ??
                          item.scenario ??
                          item.scenarioName ??
                          `Scenario ${index + 1}`;

                        const defaultProbability =
                          item.next_12m_default_flag ??
                          item.default_probability ??
                          item.default_probability_12m;

                        const riskLevel =
                          item.risk_level ??
                          item.default_risk_band;

                        return (
                          <div
                            className="table-row"
                            key={`${name}-${index}`}
                          >
                            <strong>{name}</strong>

                            <span>
                              {riskLevel ||
                                (defaultProbability != null
                                  ? formatScenarioPercent(
                                      defaultProbability
                                    )
                                  : "View")}
                            </span>
                          </div>
                        );
                      }
                    )}
                  </div>
                )}
              </div>
            </section>

            <section className="panel history-panel">
              <div className="panel-title">
                <div>
                  <h3>Loan Performance History</h3>
                  <p>
                    Historical monthly servicing observations
                  </p>
                </div>

                <span className="count">
                  {history.length}
                </span>
              </div>

              {history.length === 0 ? (
                <div className="empty-state">
                  No historical records available.
                </div>
              ) : (
                <div className="history-scroll">
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(history[0])
                          .slice(0, 7)
                          .map((key) => (
                            <th key={key}>
                              {key.replaceAll(
                                "_",
                                " "
                              )}
                            </th>
                          ))}
                      </tr>
                    </thead>

                    <tbody>
                      {history.slice(-12).map(
                        (row, index) => (
                          <tr key={index}>
                            {Object.values(row)
                              .slice(0, 7)
                              .map(
                                (value, cellIndex) => (
                                  <td key={cellIndex}>
                                    {displayValue(value)}
                                  </td>
                                )
                              )}
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </>
        )}

        <section className="section-heading scenario-heading">
          <div>
            <div className="eyebrow">
              PORTFOLIO VIEW
            </div>

            <h2>Macro Scenario Monitor</h2>

            <p>
              Portfolio-level stress testing from the
              scenario engine.
            </p>
          </div>
        </section>

        <section className="scenario-cards">
          {Array.isArray(scenarioSummary) ? (
            scenarioSummary
              .filter((scenario, index, array) => {
                const name =
                  scenario?.scenario ??
                  scenario?.scenario_name;

                return (
                  name &&
                  array.findIndex(
                    (x) =>
                      (x?.scenario ??
                        x?.scenario_name) === name
                  ) === index
                );
              })
              .slice(0, 3)
              .map((scenario, index) => (
                <div
                  className="scenario-card"
                  key={index}
                >
                  <span className="scenario-name">
                    {scenario.scenario ??
                      scenario.scenario_name}
                  </span>

                  <strong>
                    {scenario.next_12m_default_flag_mean !=
                    null
                      ? formatScenarioPercent(
                          scenario.next_12m_default_flag_mean
                        )
                      : "—"}
                  </strong>

                  <span>
                    Expected 12M default probability
                  </span>
                </div>
              ))
          ) : (
            scenarioSummary &&
            Object.entries(scenarioSummary)
              .slice(0, 3)
              .map(([key, value]) => (
                <div
                  className="scenario-card"
                  key={key}
                >
                  <span className="scenario-name">
                    {key}
                  </span>

                  <strong>
                    {typeof value === "number"
                      ? formatScenarioPercent(value)
                      : displayValue(value)}
                  </strong>
                </div>
              ))
          )}
        </section>

        <footer>
          <span>INTAIN Loan Intelligence</span>
          <span>Predict · Detect · Stress Test</span>
        </footer>
      </main>
    </div>
  );
}

export default App;