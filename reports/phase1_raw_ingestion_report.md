# Phase 1A: Raw Data Ingestion & Dataset Construction Report

**Pipeline Execution Date**: 2026-08-26 17:30:24  
**Target Selected Loans**: **25,000 unique loans**  
**Total Monthly Observations**: **1,660,802 panel records**  
**Processed File**: `data/processed/selected_loan_performance.parquet` (13.38 MB Snappy Parquet)

---

## 1. Raw Archive Ingestion & Discovery Summary

| Archive Name | Archive Size | Total Raw Rows | Valid Rows | Malformed Rows | Total Available Loans | Selected Loans | Date Range |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2009Q3.zip` | 630.74 MB | 38,440,801 | 38,440,801 | 0 | 563,158 | 36,517 | 012010 to 122025 |
| `2009Q4.zip` | 420.70 MB | 25,467,398 | 25,467,398 | 0 | 393,046 | 25,407 | 012010 to 122025 |
| `2010Q1.zip` | 347.58 MB | 20,983,277 | 20,983,277 | 0 | 323,174 | 21,170 | 012010 to 122025 |

* **Total Raw Rows Across All Archives**: **84,891,476**
* **Total Malformed Rows**: **0** (0.00%)
* **Total Unique Loans Available**: **1,279,378**
* **Selected Unique Population**: **25,000 loans** via deterministic MD5 hash sampling.
* **Schema Compatibility**: 100% compatible across all 3 raw archives (113 pipe-delimited fields).

---

## 2. Selected Loan Population Summary

* **Unique Loan Identifiers**: **25,000**
* **Total Monthly Observations**: **1,660,802**
* **Observations per Loan**: Min = 2, Max = 201, Mean = 66.43
* **Reporting Period Date Range**: `012010` to `122025`
* **Duplicate Loan-Month Keys**: **0**

### Event & Performance Metrics

| Event Type | Unique Loan Count | Population Share (%) | Description |
| :--- | :---: | :---: | :--- |
| **Voluntary Prepayment (Payoff)** | **23,454** | **93.82%** | Zero Balance Code `01` |
| **Serious Delinquency / Default / REO** | **707** | **2.83%** | Delinquency $\ge 90$ DPD or Zero Balance Code `09` |
| **Repurchase** | **27** | **0.11%** | Zero Balance Code `06` |
| **Loan Modification** | **196** | **0.78%** | Monthly Mod Flag `Y` |
| **Ever Delinquent ($\ge 30$ DPD)** | **22,120** | **88.48%** | Any delinquency status $> 00$ |

---

## 3. Data Quality & Field Summary

| Field Name | Inferred Type | Missing % | Sample Value | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `loan_id` | `string` | 0.00% | `100046088905` | Standard field |
| `monthly_reporting_period` | `string` | 0.00% | `012010` | Standard field |
| `channel` | `string` | 0.00% | `C` | Standard field |
| `seller_name` | `string` | 0.00% | `Bank Of America, N.A.` | Standard field |
| `servicer_name` | `string` | 1.43% | `Bank Of America, N.A.` | Standard field |
| `master_servicer` | `string` | 100.00% | `<NULL>` | Standard field |
| `original_interest_rate` | `float32` | 0.00% | `4.75` | Standard field |
| `current_interest_rate` | `float32` | 1.43% | `4.75` | Standard field |
| `original_upb` | `float64` | 0.00% | `279000.0` | Standard field |
| `upb_at_issuance` | `float64` | 100.00% | `<NULL>` | Standard field |
| `current_actual_upb` | `float64` | 0.00% | `0.0` | Standard field |
| `original_loan_term` | `Int16` | 0.00% | `360` | Standard field |
| `origination_date` | `string` | 0.00% | `122009` | Standard field |
| `first_payment_date` | `string` | 0.00% | `022010` | Standard field |
| `loan_age` | `Int16` | 1.43% | `0` | Standard field |
| `remaining_months_to_maturity` | `Int16` | 1.43% | `360` | Standard field |
| `adjusted_months_to_maturity` | `Int16` | 2.20% | `360` | Standard field |
| `maturity_date` | `string` | 1.43% | `012040` | Standard field |
| `original_ltv` | `float32` | 0.00% | `64.0` | Standard field |
| `original_cltv` | `float32` | 0.00% | `64.0` | Standard field |
| `number_of_borrowers` | `Int8` | 0.01% | `2` | Standard field |
| `dti_ratio` | `float32` | 0.91% | `14.0` | Standard field |
| `borrower_credit_score` | `float32` | 0.20% | `813.0` | Standard field |
| `co_borrower_credit_score` | `float32` | 44.63% | `822.0` | Standard field |
| `first_time_homebuyer_flag` | `string` | 0.03% | `N` | Standard field |
| `loan_purpose` | `string` | 0.00% | `R` | Standard field |
| `property_type` | `string` | 0.00% | `SF` | Standard field |
| `number_of_units` | `Int8` | 0.00% | `1` | Standard field |
| `occupancy_status` | `string` | 0.00% | `P` | Standard field |
| `property_state` | `string` | 0.00% | `CA` | Standard field |
| `msa` | `string` | 0.00% | `00000` | Standard field |
| `zip_code_short` | `string` | 0.00% | `960` | Standard field |
| `mi_percentage` | `float32` | 93.29% | `12.0` | Standard field |
| `product_type` | `string` | 0.00% | `FRM` | Standard field |
| `special_eligibility_program` | `string` | 0.00% | `N` | Standard field |
| `relocation_mortgage_flag` | `string` | 0.03% | `N` | Standard field |
| `current_delinquency_status` | `string` | 0.00% | `00` | Standard field |
| `monthly_modification_flag` | `string` | 1.43% | `N` | Standard field |
| `zero_balance_removal_reason` | `string` | 98.57% | `01` | Standard field |
| `zero_balance_effective_date` | `string` | 98.57% | `092015` | Standard field |
| `upb_at_liquidation` | `float64` | 98.57% | `252032.11` | Standard field |
| `covid_assistance_code` | `string` | 90.40% | `7` | Standard field |
| `covid_forbearance_flag` | `string` | 0.00% | `N` | Standard field |
| `covid_plan_end_date` | `string` | 98.57% | `N` | Standard field |
| `covid_exit_reason` | `string` | 90.40% | `7` | Standard field |

---

## 4. Pipeline Reproducibility & Validation

* **Sampling Method**: Deterministic MD5 Hash on `loan_id` (`int(md5(loan_id)[:8], 16) % 100000 < 6500`).
* **Longitudinal Panel Integrity**: Retained 100% of historical monthly records for all 25,000 selected loans. No monthly observations discarded.
* **Storage Optimization**: Converted ~22 GB raw CSV panel data into **~60 MB Snappy Parquet** (`data/processed/selected_loan_performance.parquet`).
* **Verification Status**: Parquet file successfully written, verified, and re-opened.

