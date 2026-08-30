# Temporal Distribution Shift Audit

- Validation period: 2016-01 to 2017-12
- Validation rows: 161,449
- Holdout rows: 285,776
- Validation default rate: 0.201222%
- Holdout default rate: 0.102621%


## Largest numeric distribution shifts

                      feature    type  shift_score_ks    train_mean  holdout_mean  train_median  holdout_median  train_missing_pct  holdout_missing_pct
                     loan_age numeric        0.941768     85.641365    134.175677     85.000000      129.000000           1.499545             1.496277
             pct_life_elapsed numeric        0.667020      0.309811      0.454020      0.252778        0.405556           1.499545             1.496277
 remaining_months_to_maturity numeric        0.640257    225.278341    189.386085    269.000000      215.000000           1.499545             1.496277
  adjusted_months_to_maturity numeric        0.498167    206.439051    164.166892    260.000000      191.000000           2.979888             4.159202
             upb_pct_original numeric        0.458524      0.739783      0.619640      0.849250        0.710654           0.000000             0.000000
months_since_last_delinquency numeric        0.157649     30.226751     43.878826     24.000000       32.000000          85.093745            76.554014
           current_actual_upb numeric        0.120098 131615.017820 107732.193210 108126.810000    87213.250000           0.000000             0.000000
      max_delinquency_to_date numeric        0.085397      0.390990      0.792243      0.000000        0.000000           0.000000             0.000000
              ever_delinquent numeric        0.085397      0.149063      0.234460      0.000000        0.000000           0.000000             0.000000
           original_loan_term numeric        0.050668    308.023066    317.274162    360.000000      360.000000           0.000000             0.000000
       original_interest_rate numeric        0.028936      4.955223      4.983081      4.875000        4.875000           0.000000             0.000000
        current_interest_rate numeric        0.025552      4.948832      4.970188      4.875000        4.875000           1.499545             1.488928
                 original_upb numeric        0.024243 173337.536931 168437.265551 146000.000000   140000.000000           0.000000             0.000000
        borrower_credit_score numeric        0.022879    755.338143    753.243187    766.000000      763.000000           0.117065             0.136820
          number_of_borrowers numeric        0.020690      1.538789      1.517895      2.000000        2.000000           0.000000             0.000000
     co_borrower_credit_score numeric        0.019987    762.358941    760.924166    773.000000      771.000000          46.908312            48.983819
        delinquency_count_12m numeric        0.018820      0.243136      0.336354      0.000000        0.000000           0.000000             0.000000
                mi_percentage numeric        0.017155     20.472001     20.382180     25.000000       25.000000          93.706372            93.633825
                ever_modified numeric        0.012021      0.014958      0.026979      0.000000        0.000000           0.000000             0.000000
             current_modified numeric        0.011850      0.014803      0.026654      0.000000        0.000000           0.000000             0.000000