# Phase 4 — Macro Scenario Risk Simulation

The Phase 2 trained models provide the baseline probabilities. Phase 4 applies the explicit scenario multipliers from macro_scenarios.csv to perform deterministic stress testing.

## Scenario assumptions

### Base
- gdp_growth_pct: 2.0
- unemployment_rate_pct: 4.0
- hpi_growth_pct: 3.0
- mortgage_rate_30y_pct: 6.5
- prepayment_multiplier: 1.0
- default_multiplier: 1.0

### Adverse-Credit
- gdp_growth_pct: -1.5
- unemployment_rate_pct: 8.0
- hpi_growth_pct: -10.0
- mortgage_rate_30y_pct: 7.5
- prepayment_multiplier: 0.5
- default_multiplier: 3.0

### High-Prepayment
- gdp_growth_pct: 3.0
- unemployment_rate_pct: 3.5
- hpi_growth_pct: 6.0
- mortgage_rate_30y_pct: 4.5
- prepayment_multiplier: 2.5
- default_multiplier: 0.5

## Risk results

### Base

- next_3m_delinquency_flag: mean=3.9860%, P95=16.3882%, high-risk count=8,132
- next_6m_delinquency_flag: mean=4.9428%, P95=24.2585%, high-risk count=9,462
- next_12m_default_flag: mean=0.4888%, P95=0.0846%, high-risk count=1,083
- next_12m_prepayment_flag: mean=16.2301%, P95=53.1459%, high-risk count=14,778

### Adverse-Credit

- next_3m_delinquency_flag: mean=5.6611%, P95=28.3853%, high-risk count=10,872
- next_6m_delinquency_flag: mean=7.1797%, P95=42.0170%, high-risk count=13,073
- next_12m_default_flag: mean=0.7321%, P95=0.2538%, high-risk count=1,360
- next_12m_prepayment_flag: mean=8.1150%, P95=26.5730%, high-risk count=0

### High-Prepayment

- next_3m_delinquency_flag: mean=2.8185%, P95=11.5882%, high-risk count=6,769
- next_6m_delinquency_flag: mean=3.4951%, P95=17.1534%, high-risk count=7,164
- next_12m_default_flag: mean=0.2444%, P95=0.0423%, high-risk count=39
- next_12m_prepayment_flag: mean=35.2197%, P95=100.0000%, high-risk count=36,340

## Change versus Base

### Adverse-Credit
- next_3m_delinquency_flag: +42.03%
- next_6m_delinquency_flag: +45.26%
- next_12m_default_flag: +49.77%
- next_12m_prepayment_flag: -50.00%

### High-Prepayment
- next_3m_delinquency_flag: -29.29%
- next_6m_delinquency_flag: -29.29%
- next_12m_default_flag: -50.00%
- next_12m_prepayment_flag: +117.00%

## Interpretation

These scenario results are stress-test outputs, not causal macroeconomic forecasts. The macro_scenarios.csv multipliers are explicit assumptions supplied for scenario analysis.